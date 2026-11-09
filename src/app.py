from pathlib import Path
import sys
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

# Allow the documented direct-file launch as well as `python -m src.app`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.action_agent import ActionOrchestrationAgent
from src.agents.clinical_agent import ClinicalEvaluationAgent
from src.agents.signal_agent import SignalProcessingAgent


class RadarState(TypedDict, total=False):
    raw_chunk: list[float]
    demographic: str
    telemetry: dict[str, Any]
    evaluation: dict[str, Any]
    action_result: str
    no_respiration_seconds: float
    advance_seconds: float
    fall_event_detected: bool
    covered_end: int


signal_agent = SignalProcessingAgent(sample_rate=20)
clinical_agent = ClinicalEvaluationAgent()
action_agent = ActionOrchestrationAgent()


def signal_processing_node(state: RadarState) -> dict[str, Any]:
    telemetry = signal_agent.process_frame_stream(state["raw_chunk"], state["demographic"])
    duration = 0.0
    if telemetry["respiration_rate_br"] == 0:
        duration = state.get("no_respiration_seconds", 0.0) + state.get(
            "advance_seconds", len(state["raw_chunk"]) / signal_agent.fs
        )
    telemetry["no_respiration_seconds"] = duration
    if "fall_event_detected" in state:
        telemetry["fall_event_detected"] = state["fall_event_detected"]
    return {"telemetry": telemetry, "no_respiration_seconds": duration}


def clinical_evaluation_node(state: RadarState) -> dict[str, Any]:
    return {"evaluation": clinical_agent.evaluate_telemetry(state["telemetry"])}


def action_orchestration_node(state: RadarState) -> dict[str, Any]:
    return {"action_result": action_agent.execute_safety_protocol(state["evaluation"])}


workflow = StateGraph(RadarState)
workflow.add_node("signal_processor", signal_processing_node)
workflow.add_node("clinical_evaluator", clinical_evaluation_node)
workflow.add_node("action_orchestrator", action_orchestration_node)
workflow.add_edge(START, "signal_processor")
workflow.add_edge("signal_processor", "clinical_evaluator")
workflow.add_edge("clinical_evaluator", "action_orchestrator")
workflow.add_edge("action_orchestrator", END)
app_graph = workflow.compile()
