from src.utils.config import PATIENT_THRESHOLDS, validate_demographic, validate_number


class ClinicalEvaluationAgent:
    def evaluate_telemetry(self, telemetry):
        """Apply unvalidated cohort screening rules, not individual age-based diagnosis."""
        required = {"demographic", "respiration_rate_br", "heart_rate_bpm", "max_amplitude"}
        if not isinstance(telemetry, dict) or not required.issubset(telemetry):
            raise ValueError("Telemetry is missing required fields.")
        demo = validate_demographic(telemetry["demographic"])
        br = validate_number(telemetry["respiration_rate_br"], "Respiration rate")
        bpm = validate_number(telemetry["heart_rate_bpm"], "Heart rate")
        amp = validate_number(telemetry["max_amplitude"], "Amplitude")
        duration = validate_number(
            telemetry.get("no_respiration_seconds", 0), "No-respiration duration"
        )
        fall = telemetry.get("fall_event_detected", amp > 8.0)
        if not isinstance(fall, bool):
            raise ValueError("fall_event_detected must be a boolean.")
        thresholds = PATIENT_THRESHOLDS[demo]
        critical = False
        findings = []
        if thresholds["fall_detection_enabled"] and fall:
            critical = True
            findings.append("DISPLACEMENT SPIKE: potential fall; requires review.")
        if br == 0:
            if duration >= thresholds["apnea_limit_seconds"]:
                critical = True
                findings.append(
                    "PERSISTENT UNDETECTED RESPIRATION: possible apnea or signal loss; requires review."
                )
            else:
                findings.append("Respiration not detected; check signal quality and duration.")
        elif br < thresholds["min_br"] or br > thresholds["max_br"]:
            findings.append(f"Abnormal Respiration: {br} BR")
        if bpm < thresholds["min_bpm"] or bpm > thresholds["max_bpm"]:
            findings.append(f"Abnormal Cardiac Cycle: {bpm} BPM")
        status = "CRITICAL" if critical else "WARNING" if findings else "NORMAL"
        if not findings:
            findings.append("Estimates within configured cohort bounds.")
        return {
            "status": status,
            "demographic": demo,
            "vitals_evaluated": {"BPM": bpm, "BR": br},
            "clinical_findings": findings,
        }
