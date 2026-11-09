import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt

from src.utils.config import validate_number


def validate_signal(signal_array, fs):
    fs = validate_number(fs, "Sampling frequency")
    if fs <= 5.0:
        raise ValueError("Sampling frequency must exceed 5 Hz for the cardiac band.")
    data = np.asarray(signal_array, dtype=float)
    if data.ndim != 1 or not np.isfinite(data).all():
        raise ValueError("Signal must be a one-dimensional array of finite numbers.")
    if len(data) < int(np.ceil(10 * fs)):
        raise ValueError("Insufficient data: at least 10 seconds of samples are required.")
    return data


def butter_bandpass(lowcut, highcut, fs, order=4):
    """Return Butterworth transfer-function coefficients."""
    fs = validate_number(fs, "Sampling frequency")
    if fs == 0:
        raise ValueError("Sampling frequency must be greater than zero.")
    if not 0 < lowcut < highcut < fs / 2:
        raise ValueError("Bandpass cutoffs must satisfy 0 < lowcut < highcut < fs/2.")
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("Filter order must be a positive integer.")
    return butter(order, [lowcut, highcut], btype="bandpass", fs=fs)


def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Filter finite samples using numerically stable second-order sections."""
    butter_bandpass(lowcut, highcut, fs, order)
    data = np.asarray(data, dtype=float)
    if data.ndim != 1 or not np.isfinite(data).all():
        raise ValueError("Signal must be a one-dimensional array of finite numbers.")
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    padlen = 3 * (2 * len(sos) + 1)
    if len(data) <= padlen:
        raise ValueError(f"Insufficient data: filter requires more than {padlen} samples.")
    return sosfiltfilt(sos, data)


def _estimate_rate(data, fs, lowcut, highcut, minimum_amplitude):
    # Window the original signal: filter edge transients must not become rate peaks.
    window = np.hanning(len(data))
    amplitudes = 2 * np.abs(np.fft.rfft(detrend(data) * window)) / window.sum()
    frequencies = np.fft.rfftfreq(len(data), d=1 / fs)
    band = (frequencies >= lowcut) & (frequencies <= highcut)
    band_amplitudes = amplitudes[band]
    index = int(np.argmax(band_amplitudes))
    if band_amplitudes[index] < minimum_amplitude:
        return 0.0
    return float(round(float(frequencies[band][index]) * 60, 1))


def extract_vitals_from_phase(signal_array, fs=20, respiratory_highcut=1.0,
                             cardiac_lowcut=0.8):
    """Estimate rates for mock signals; zero means no component detected, not arrest."""
    data = validate_signal(signal_array, fs)
    resp_wave = apply_bandpass_filter(data, 0.1, respiratory_highcut, fs)
    cardiac_wave = apply_bandpass_filter(data, cardiac_lowcut, 2.5, fs)
    return {
        "calculated_br": _estimate_rate(data, fs, 0.1, respiratory_highcut, 0.05),
        "calculated_bpm": _estimate_rate(data, fs, cardiac_lowcut, 2.5, 0.01),
        "max_displacement_amplitude": float(np.max(np.abs(data))),
        "filtered_resp_signal": resp_wave.tolist(),
        "filtered_cardiac_signal": cardiac_wave.tolist(),
    }
