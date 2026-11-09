import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from src.agents.action_agent import ActionOrchestrationAgent
from src.agents.clinical_agent import ClinicalEvaluationAgent
from src.agents.signal_agent import SignalProcessingAgent
from src.simulation import run_langgraph_simulation
from src.dsp.filters import extract_vitals_from_phase


class RadarTests(unittest.TestCase):
    def test_known_rates(self):
        t = np.arange(1200) / 20
        for resp, cardiac, amplitude, br, bpm in [
            (0.25, 1.2, 0.5, 15, 72), (0.66, 2.1, 0.8, 40, 126)
        ]:
            signal = amplitude * np.sin(2*np.pi*resp*t) + 0.15*np.sin(2*np.pi*cardiac*t)
            result = extract_vitals_from_phase(signal)
            self.assertEqual((result["calculated_br"], result["calculated_bpm"]), (br, bpm))
            self.assertEqual(len(result["filtered_resp_signal"]), 1200)
            json.dumps(result, allow_nan=False)

    def test_seeded_apnea_windows(self):
        for seed in range(100):
            t = np.arange(600) / 20
            signal = 0.08*np.sin(2*np.pi*1.3*t) + np.random.default_rng(seed).normal(0, 0.03, 600)
            for start in range(0, 401, 40):
                result = extract_vitals_from_phase(signal[start:start+200])
                self.assertEqual(result["calculated_br"], 0, (seed, start))

    def test_invalid_signals(self):
        for signal in [[], [1]*27, [np.nan]*600, [np.inf]*600, [[0]*600]]:
            with self.assertRaises(ValueError):
                extract_vitals_from_phase(signal)
        with self.assertRaises(ValueError):
            SignalProcessingAgent(0)
        with self.assertRaises(ValueError):
            SignalProcessingAgent().process_frame_stream([0]*600, "adult")

    def test_clinical_validation_and_priority(self):
        agent = ClinicalEvaluationAgent()
        frame = {"demographic": "elderly", "heart_rate_bpm": 0, "respiration_rate_br": 0,
                 "max_amplitude": 8.5, "no_respiration_seconds": 20}
        result = agent.evaluate_telemetry(frame)
        self.assertEqual(result["status"], "CRITICAL")
        self.assertEqual(len(result["clinical_findings"]), 3)
        for bad in [{}, dict(frame, demographic="adult"), dict(frame, heart_rate_bpm=np.nan)]:
            with self.assertRaises(ValueError):
                agent.evaluate_telemetry(bad)
        for duration, expected in [(14, "WARNING"), (15, "CRITICAL")]:
            result = agent.evaluate_telemetry(dict(frame, demographic="infant",
                max_amplitude=0, heart_rate_bpm=126, no_respiration_seconds=duration))
            self.assertEqual(result["status"], expected)

    def test_actions_are_simulated_and_validated(self):
        agent = ActionOrchestrationAgent()
        for status in ["NORMAL", "WARNING", "CRITICAL"]:
            for demo in ["infant", "elderly"]:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = agent.execute_safety_protocol(
                        {"status": status, "demographic": demo, "clinical_findings": ["test"]}
                    )
                self.assertTrue(result.startswith("SIMULATED_ACTION:"))
                self.assertTrue(result.isascii())
        for changes in [{"status": "ERROR"}, {"demographic": None}, {"clinical_findings": "ABC"}]:
            with self.assertRaises(ValueError):
                agent.execute_safety_protocol({
                    "status": "NORMAL", "demographic": "infant", "clinical_findings": ["test"],
                    **changes,
                })

    def test_window_coverage_and_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenario.json"
            for length, windows in [(200, 1), (600, 11), (615, 12)]:
                path.write_text(json.dumps({"signal": [0]*length, "demographics": "infant"}))
                with contextlib.redirect_stdout(io.StringIO()):
                    results = run_langgraph_simulation(str(path), "infant")
                self.assertEqual(len(results), windows)
                self.assertAlmostEqual(results[-1]["no_respiration_seconds"], length/20)
                if length == 600:
                    self.assertEqual([r["evaluation"]["status"] for r in results[:4]],
                                     ["WARNING", "WARNING", "WARNING", "CRITICAL"])
            path.write_text(json.dumps({"signal": [0]*100 + [-8.5] + [0]*499}))
            with contextlib.redirect_stdout(io.StringIO()):
                results = run_langgraph_simulation(str(path), "elderly")
            self.assertEqual(sum(r["telemetry"]["fall_event_detected"] for r in results), 1)
            with self.assertRaises(FileNotFoundError):
                run_langgraph_simulation(str(Path(directory)/"missing.json"), "infant")

    def test_direct_launch(self):
        root = Path(__file__).resolve().parent.parent
        result = subprocess.run([sys.executable, "-m", "src.simulation", "--help"],
                                cwd=root, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_streamlit_playback(self):
        from streamlit.testing.v1 import AppTest
        root = Path(__file__).resolve().parent.parent
        ui = AppTest.from_file(str(root / "src" / "streamlit_app.py"), default_timeout=20).run()
        ui.radio[0].set_value("Mock file playback").run()
        self.assertFalse(ui.exception)
        ui.button[0].click().run()
        self.assertFalse(ui.exception)
        self.assertEqual(len(ui.session_state["results"]), 1)
        ui.run()
        self.assertEqual(len(ui.session_state["results"]), 1)
        ui.button[0].click().run()
        self.assertEqual(ui.session_state["results"][-1]["no_respiration_seconds"], 12)
        ui.button[1].click().run()
        self.assertEqual(len(ui.session_state["results"]), 0)
        ui.selectbox[0].select("Elderly fall").run()
        ui.button[0].click().run()
        self.assertFalse(ui.exception)
        self.assertEqual(ui.session_state["results"][-1]["evaluation"]["status"], "CRITICAL")

    def test_live_injections(self):
        from src.simulation import LiveSimulation
        with contextlib.redirect_stdout(io.StringIO()):
            infant = LiveSimulation("infant")
            self.assertEqual(infant.advance()["evaluation"]["status"], "NORMAL")
            infant.apnea = True
            results = [infant.advance() for _ in range(15)]
            self.assertEqual(results[-1]["evaluation"]["status"], "CRITICAL")
            self.assertEqual(len(infant.buffer), 200)
            infant.apnea = False
            for _ in range(5):
                result = infant.advance()
            self.assertEqual(result["evaluation"]["status"], "NORMAL")
            self.assertEqual(result["no_respiration_seconds"], 0)
            elderly = LiveSimulation("elderly")
            elderly.advance()
            elderly.pending_fall = True
            self.assertTrue(elderly.advance()["telemetry"]["fall_event_detected"])
            self.assertFalse(elderly.advance()["telemetry"]["fall_event_detected"])

    def test_live_ui_controls(self):
        from streamlit.testing.v1 import AppTest
        root = Path(__file__).resolve().parent.parent
        ui = AppTest.from_file(str(root/"src"/"streamlit_app.py"), default_timeout=20).run()
        ui.button[1].click().run()
        self.assertFalse(ui.exception)
        self.assertEqual(ui.session_state["live_source"].samples, 200)
        ui.run()
        self.assertEqual(ui.session_state["live_source"].samples, 200)
        ui.button[3].click().run()
        self.assertTrue(ui.session_state["live_source"].apnea)
        ui.button[3].click().run()
        self.assertFalse(ui.session_state["live_source"].apnea)
        ui.button[0].click().run()
        self.assertTrue(ui.session_state["live_running"])
        ui.button[0].click().run()
        self.assertFalse(ui.session_state["live_running"])
        ui.selectbox[0].select("Elderly Care").run()
        ui.button[3].click().run()
        ui.button[1].click().run()
        self.assertFalse(ui.exception)
        self.assertTrue(ui.session_state["live_source"].previous["telemetry"]["fall_event_detected"])


if __name__ == "__main__":
    unittest.main()
