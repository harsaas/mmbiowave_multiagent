import os
import json
import numpy as np

# Create the folder structure locally
os.makedirs(os.path.join("data", "raw"), exist_ok=True)
os.makedirs(os.path.join("data", "simulated"), exist_ok=True)

def generate_radar_stream(duration_seconds=30, fs=20, condition="normal"):
    """Simulates raw extracted phase values from a mmWave radar range bin."""
    t = np.arange(int(duration_seconds * fs)) / fs
    
    # Set up baseline physiological frequencies
    if condition == "infant_normal":
        br_freq, hr_freq = 0.66, 2.1  # 40 Breaths/min, 126 Beats/min
        resp_amp, cardiac_amp = 0.8, 0.15
        noise = np.random.normal(0, 0.05, len(t))
    elif condition == "infant_apnea":
        br_freq, hr_freq = 0.0, 1.3   # Zero breathing, dropping heart rate (Bradycardia)
        resp_amp, cardiac_amp = 0.0, 0.08
        noise = np.random.normal(0, 0.03, len(t))
    elif condition == "elderly_fall":
        # Simulates normal baseline that suddenly halts due to a rapid impact vector
        br_freq, hr_freq = 0.25, 1.2  # 15 Breaths/min, 72 Beats/min
        resp_amp, cardiac_amp = 0.5, 0.1
        noise = np.random.normal(0, 0.04, len(t))
    else:  # Adult Normal Resting
        br_freq, hr_freq = 0.25, 1.2
        resp_amp, cardiac_amp = 0.5, 0.1
        noise = np.random.normal(0, 0.04, len(t))

    # Generate waveforms
    resp_wave = resp_amp * np.sin(2 * np.pi * br_freq * t)
    cardiac_wave = cardiac_amp * np.sin(2 * np.pi * hr_freq * t)
    
    # Combine signals to mimic raw phase displacement data
    total_signal = resp_wave + cardiac_wave + noise
    return t.tolist(), total_signal.tolist()

# 1. Generate Raw Data Example (Mirroring an adult target file)
timestamps, signal = generate_radar_stream(duration_seconds=60, condition="normal")
raw_data = {"timestamps": timestamps, "raw_phase_displacement": signal}
with open(os.path.join("data", "raw", "raw_radar_target.json"), "w") as f:
    json.dump(raw_data, f, indent=4)

# 2. Generate Simulated Agent Scenarios
# Scenario A: Infant Apnea Crisis
t_infant, sig_infant = generate_radar_stream(duration_seconds=30, condition="infant_apnea")
with open(os.path.join("data", "simulated", "infant_apnea.json"), "w") as f:
    json.dump({"demographics": "infant", "age_months": 3, "signal": sig_infant}, f, indent=4)

# Scenario B: Elderly Fall State
t_elderly, sig_elderly = generate_radar_stream(duration_seconds=30, condition="elderly_fall")
# Emulate a massive sudden micro-Doppler deceleration spike right at frame 100
sig_elderly[100] = -8.5 
with open(os.path.join("data", "simulated", "elderly_fall.json"), "w") as f:
    json.dump({"demographics": "elderly", "age_years": 82, "signal": sig_elderly}, f, indent=4)

print("Local mock data initialized successfully!")
print(" -> Created data/raw/raw_radar_target.json")
print(" -> Created data/simulated/infant_apnea.json")
print(" -> Created data/simulated/elderly_fall.json")
