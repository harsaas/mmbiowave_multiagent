# src/agents/signal_agent.py
from src.dsp.filters import extract_vitals_from_phase
from src.utils.config import validate_demographic, validate_number

class SignalProcessingAgent:
    def __init__(self, sample_rate=20):
        self.fs = validate_number(sample_rate, "Sampling frequency")
        if self.fs <= 5:
            raise ValueError("Sampling frequency must exceed 5 Hz.")

    def process_frame_stream(self, raw_signal, demographic):
        """Transforms raw waves into clean structured JSON metrics"""
        validate_demographic(demographic)
        vitals = extract_vitals_from_phase(
            raw_signal, fs=self.fs,
            respiratory_highcut=1.0 if demographic == "infant" else 0.5,
            cardiac_lowcut=1.0 if demographic == "infant" else 0.8,
        )
        
        # Structure the payload for downstream agent evaluation
        telemetry = {
            "demographic": demographic,
            "heart_rate_bpm": vitals["calculated_bpm"],
            "respiration_rate_br": vitals["calculated_br"],
            "max_amplitude": vitals["max_displacement_amplitude"]
        }
        return telemetry
