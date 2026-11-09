import argparse
import json
from pathlib import Path
import numpy as np

from src.app import RadarState, app_graph, signal_agent
from src.dsp.filters import validate_signal
from src.utils.config import validate_demographic


class LiveSimulation:
    """Session-local synthetic source with a bounded rolling signal window."""

    def __init__(self, demographic: str):
        self.demographic = validate_demographic(demographic)
        self.apnea = False
        self.pending_fall = False
        self.samples = 0
        self.buffer: list[float] = []
        self.previous: RadarState | None = None
        self.rng = np.random.default_rng(42)

    def advance(self) -> RadarState:
        count = 200 if self.previous is None else 40
        t = np.arange(self.samples, self.samples + count) / 20
        infant = self.demographic == "infant"
        br, hr = (0.66, 2.1) if infant else (0.25, 1.2)
        resp_amp, cardiac_amp = (0.8, 0.15) if infant else (0.5, 0.1)
        if self.apnea:
            resp_amp, hr, cardiac_amp = 0.0, 1.3, 0.08
        signal = resp_amp * np.sin(2*np.pi*br*t) + cardiac_amp * np.sin(2*np.pi*hr*t)
        signal += self.rng.normal(0, 0.03, count)
        if self.pending_fall:
            signal[count // 2] = -8.5
        combined = self.buffer + signal.tolist()
        previous = None
        if self.previous is not None:
            previous = dict(self.previous)
            previous["covered_end"] = 200
        result = process_window(combined, self.demographic, 0 if previous is None else 40, previous)
        self.samples += count
        result["covered_end"] = self.samples
        self.buffer = combined[-200:]
        self.previous = result
        self.pending_fall = False
        return result


def load_scenario(scenario_file: str, demographic: str) -> list[float]:
    validate_demographic(demographic)
    with open(scenario_file, encoding="utf-8") as handle:
        package = json.load(handle)
    if not isinstance(package, dict) or "signal" not in package:
        raise ValueError("Scenario must contain a signal array.")
    if package.get("demographics", demographic) != demographic:
        raise ValueError("Scenario demographic does not match the requested cohort.")
    return validate_signal(package["signal"], signal_agent.fs).tolist()


def window_starts(length: int) -> list[int]:
    if length < 200:
        raise ValueError("Scenario requires at least 200 samples.")
    starts = list(range(0, length - 200 + 1, 40))
    if starts[-1] != length - 200:
        starts.append(length - 200)
    return starts


def process_window(signal: list[float], demographic: str, start: int,
                   previous: RadarState | None = None) -> RadarState:
    """Process one window, counting only newly covered samples."""
    covered_end = int(previous["covered_end"]) if previous is not None else 0
    end = start + 200
    if start not in window_starts(len(signal)) or end <= covered_end or start > covered_end:
        raise ValueError("Windows must be processed in chronological, overlapping order.")
    state: RadarState = {
        "raw_chunk": signal[start:end],
        "demographic": demographic,
        "no_respiration_seconds": previous["no_respiration_seconds"] if previous else 0.0,
        "advance_seconds": (end - covered_end) / signal_agent.fs,
        "fall_event_detected": any(abs(value) > 8.0 for value in signal[covered_end:end]),
    }
    result: RadarState = app_graph.invoke(state)
    result["covered_end"] = end
    return result


def run_langgraph_simulation(scenario_file: str, demographic: str) -> list[RadarState]:
    signal = load_scenario(scenario_file, demographic)
    results: list[RadarState] = []
    for start in window_starts(len(signal)):
        result = process_window(signal, demographic, start, results[-1] if results else None)
        results.append(result)
        print(f"Window [{start}:{start + 200}]: {result['action_result']}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unvalidated mock radar simulation.")
    parser.add_argument("mode", nargs="?", choices=("infant", "elderly", "aged"), default="infant")
    args = parser.parse_args()
    mode = "elderly" if args.mode == "aged" else args.mode
    filename = "infant_apnea.json" if mode == "infant" else "elderly_fall.json"
    path = Path(__file__).resolve().parent.parent / "data" / "simulated" / filename
    try:
        run_langgraph_simulation(str(path), mode)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Simulation failed: {error}\n")
