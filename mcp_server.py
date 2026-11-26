# mcp_server.py
"""
Real Model Context Protocol (MCP) server using FastMCP.
Exposes the wafer inspection pipeline, EDA/model metrics, and Groq AI assistant
as standard MCP tools for Claude Desktop, Cursor, or Antigravity.

Run with:
    python mcp_server.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    from mcp.server.fastmcp import FastMCP as MCPServer

from workflows.wafer_pipeline import build_graph
from mcp.chatbot import answer_question

mcp = MCPServer("WaferAI-Inspector")
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph(enable_llm_diagnosis=True)
    return _graph


@mcp.tool()
def inspect_wafer(image_path: str) -> str:
    """
    Run the full LangGraph wafer defect inspection pipeline on a local image path.
    Returns predicted defect class, confidence, severity, route taken, and diagnostic summary.
    """
    result = _get_graph().invoke({"path": image_path})
    out = dict(result.get("output", {}))
    # Omit raw base64 image from text tool output to keep context compact
    if "gradcam_b64" in out:
        out["gradcam_available"] = bool(out["gradcam_b64"])
        out.pop("gradcam_b64", None)
    return json.dumps(out, indent=2)


@mcp.tool()
def get_model_metrics() -> str:
    """
    Return stored dataset EDA statistics and ResNet-18 validation metrics
    (overall accuracy, macro/weighted F1, per-class scores, and confusion matrix).
    """
    metrics_file = ROOT / "reports" / "metrics_eda.json"
    if not metrics_file.exists():
        return json.dumps({"error": "metrics_eda.json not found. Run scripts/evaluate.py first."})
    return metrics_file.read_text(encoding="utf-8")


@mcp.tool()
def ask_wafer_expert(question: str, context_json: str = "{}") -> str:
    """
    Ask the Groq-backed semiconductor AI assistant a question about wafer inspection
    results or domain concepts (Grad-CAM, ResNet-18, defect types, yield impact).
    """
    try:
        payload = json.loads(context_json) if context_json else {}
    except json.JSONDecodeError:
        payload = {}
    return answer_question(question, payload)


if __name__ == "__main__":
    mcp.run()
