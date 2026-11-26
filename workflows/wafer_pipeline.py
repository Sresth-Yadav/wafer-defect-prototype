# workflows/wafer_pipeline.py
"""
Agentic LangGraph StateGraph pipeline for wafer-defect inspection.

Graph topology:
  START -> ingest -> infer -> explain
    ├── (confidence < 0.75)           -> human_review -> diagnose -> aggregate -> END
    ├── (severity in {High, Medium})  -> diagnose -> aggregate -> END
    └── (Low severity & high conf.)   -> aggregate -> END
"""
import sys
import json
from pathlib import Path
from typing import TypedDict, Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langgraph.graph import StateGraph, START, END

from mcp.preprocess import preprocess_image
from mcp.inference import run_inference
from mcp.explain import generate_gradcam

LOW_CONFIDENCE_THRESHOLD = 0.75


class WaferState(TypedDict, total=False):
    path: str
    image: Any
    inference: Dict[str, Any]
    gradcam_b64: str
    severity: Dict[str, Any]
    needs_review: bool
    review_reason: str
    diagnostic_report: str
    route_taken: List[str]
    output: Dict[str, Any]


def build_graph(enable_llm_diagnosis: bool = True):
    """Construct and compile the conditional LangGraph StateGraph."""
    builder = StateGraph(WaferState)

    def ingest(state: WaferState) -> Dict[str, Any]:
        image = preprocess_image(state["path"])
        return {"image": image, "route_taken": ["ingest"]}

    def infer(state: WaferState) -> Dict[str, Any]:
        results = run_inference(state["image"])
        route = list(state.get("route_taken", [])) + ["infer"]
        return {"inference": results, "route_taken": route}

    def explain(state: WaferState) -> Dict[str, Any]:
        explain_out = generate_gradcam(state["image"], state["inference"])
        route = list(state.get("route_taken", [])) + ["explain"]
        return {
            "gradcam_b64": explain_out["gradcam_b64"],
            "severity": explain_out["severity"],
            "route_taken": route,
        }

    def route_after_explain(state: WaferState) -> str:
        conf = float(state.get("inference", {}).get("confidence", 0.0))
        sev = state.get("severity", {}).get("severity_label", "Low")
        if conf < LOW_CONFIDENCE_THRESHOLD:
            return "human_review"
        if sev in {"High", "Medium"}:
            return "diagnose"
        return "aggregate"

    def human_review(state: WaferState) -> Dict[str, Any]:
        conf = float(state.get("inference", {}).get("confidence", 0.0))
        label = state.get("inference", {}).get("label", "unknown")
        route = list(state.get("route_taken", [])) + ["human_review"]
        return {
            "needs_review": True,
            "review_reason": (
                f"Low confidence ({conf * 100:.1f}% < {LOW_CONFIDENCE_THRESHOLD * 100:.0f}%) "
                f"for predicted class '{label}'. Flagged for manual engineering verification."
            ),
            "route_taken": route,
        }

    def diagnose(state: WaferState) -> Dict[str, Any]:
        route = list(state.get("route_taken", [])) + ["diagnose"]
        inf = state.get("inference", {})
        sev = state.get("severity", {})
        label = inf.get("label", "unknown")
        conf = inf.get("confidence", 0.0)
        sev_label = sev.get("severity_label", "Unknown")
        area = sev.get("affected_area_pct", 0.0)

        report = (
            f"Automated Summary: Detected '{label}' ({conf * 100:.1f}% confidence) with "
            f"{sev_label} severity ({area}% Grad-CAM affected area). Recommend inspecting "
            f"upstream lithography/etching/polishing stages for '{label}' root causes."
        )

        if enable_llm_diagnosis:
            try:
                from mcp.chatbot import answer_question
                prompt = (
                    "Provide a concise 3-bullet fab diagnostic summary for this wafer inspection: "
                    "(1) What this defect pattern indicates, (2) Likely fabrication/process root cause, "
                    "and (3) Recommended engineering action."
                )
                report = answer_question(prompt, {"inference": inf, "severity": sev})
            except Exception:
                # Graceful offline/no-key fallback
                pass

        return {"diagnostic_report": report, "route_taken": route}

    def aggregate(state: WaferState) -> Dict[str, Any]:
        route = list(state.get("route_taken", [])) + ["aggregate"]
        return {
            "route_taken": route,
            "output": {
                "inference": state.get("inference", {}),
                "severity": state.get("severity", {}),
                "gradcam_b64": state.get("gradcam_b64", ""),
                "needs_review": state.get("needs_review", False),
                "review_reason": state.get("review_reason", ""),
                "diagnostic_report": state.get("diagnostic_report", ""),
                "route_taken": route,
            },
        }

    builder.add_node("ingest", ingest)
    builder.add_node("infer", infer)
    builder.add_node("explain", explain)
    builder.add_node("human_review", human_review)
    builder.add_node("diagnose", diagnose)
    builder.add_node("aggregate", aggregate)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "infer")
    builder.add_edge("infer", "explain")
    builder.add_conditional_edges(
        "explain",
        route_after_explain,
        {
            "human_review": "human_review",
            "diagnose": "diagnose",
            "aggregate": "aggregate",
        },
    )
    builder.add_edge("human_review", "diagnose")
    builder.add_edge("diagnose", "aggregate")
    builder.add_edge("aggregate", END)

    return builder.compile()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m workflows.wafer_pipeline <image_path>")
        sys.exit(1)

    pipeline = build_graph()
    result = pipeline.invoke({"path": sys.argv[1]})
    out = dict(result.get("output", {}))
    if "gradcam_b64" in out:
        out["gradcam_b64"] = "<base64-png - not shown in CLI>"
    print(json.dumps({"output": out}, indent=2))
