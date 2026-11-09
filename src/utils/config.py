import math
from numbers import Real


def validate_demographic(demographic):
    if not isinstance(demographic, str) or demographic not in PATIENT_THRESHOLDS:
        raise ValueError("Demographic must be 'infant' or 'elderly'.")
    return demographic


def validate_number(value, name, minimum=0.0):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number.")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}.")
    return value

PATIENT_THRESHOLDS = {
    "infant": {
        "min_br": 25.0,        # Neonates breathe much faster
        "max_br": 60.0,
        "min_bpm": 100.0,      # High resting heart rate is normal for infants
        "max_bpm": 160.0,
        "fall_detection_enabled": False,
        "apnea_limit_seconds": 15
    },
    "elderly": {
        "min_br": 12.0,        # Standard adult baselines
        "max_br": 20.0,
        "min_bpm": 50.0,
        "max_bpm": 100.0,
        "fall_detection_enabled": True, # Critical for senior care tracking
        "apnea_limit_seconds": 20
    }
}
