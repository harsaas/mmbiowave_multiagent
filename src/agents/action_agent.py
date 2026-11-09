from src.utils.config import validate_demographic


class ActionOrchestrationAgent:
    def execute_safety_protocol(self, evaluation):
        """Describe simulated actions; never claim an external operation occurred."""
        required = {"status", "demographic", "clinical_findings"}
        if not isinstance(evaluation, dict) or not required.issubset(evaluation):
            raise ValueError("Evaluation is missing required fields.")
        status = evaluation["status"]
        if status not in ("NORMAL", "WARNING", "CRITICAL"):
            raise ValueError("Unsupported evaluation status.")
        demo = validate_demographic(evaluation["demographic"])
        findings = evaluation["clinical_findings"]
        if not isinstance(findings, list) or not findings or not all(
            isinstance(finding, str) and finding.strip() for finding in findings
        ):
            raise ValueError("Clinical findings must be a nonempty list of strings.")
        print(f"[Action Agent] Status={status} | Cohort={demo} | {', '.join(findings)}")
        if status == "CRITICAL":
            action = "parent/crib alert" if demo == "infant" else "nurse-station alert"
            return f"SIMULATED_ACTION: Would request {action}; no alert dispatched."
        if status == "WARNING":
            return "SIMULATED_ACTION: Would flag estimates for review; no database write."
        return "SIMULATED_ACTION: Would retain baseline estimates; no database write."
