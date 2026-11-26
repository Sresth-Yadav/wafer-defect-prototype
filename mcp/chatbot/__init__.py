"""
Chatbot MCP package – answer questions about wafer inspection results and semiconductor concepts.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from groq import Groq

ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(ENV_PATH)

# Current Groq model available for this API key
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        load_dotenv(ENV_PATH)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in .env or environment")
        _client = Groq(api_key=api_key)
    return _client

SYSTEM_PROMPT = (
    "You are an expert AI assistant for semiconductor wafer defect inspection. "
    "Your job is to answer user questions about defect detection results, "
    "severity scoring, Grad-CAM visualisations, and semiconductor wafer-process concepts "
    "(such as silk_spot, inclusion, waist folding, deburring, rolled_pit, ResNet-18, U-Net, YOLO, IoU, mAP, and yield impact). "
    "Explain technical terms clearly for engineers and researchers. "
    "If an inspection payload is provided, reference its exact predicted class, confidence, class probabilities, and severity metrics. "
    "If no inspection payload is provided yet, still answer general wafer, computer-vision, or pipeline questions helpfully; "
    "only say you don't have enough information if the user asks specifically about an unprovided image's results."
)


def _format_payload(payload: Dict[str, Any]) -> str:
    """Convert the pipeline JSON payload into a concise description for the LLM."""
    if not payload:
        return "No wafer image has been inspected yet in this session."

    # Support both wrapped {"output": {...}} and flat {"inference": {...}} payloads
    data = payload.get("output", payload)
    parts = []

    inference = data.get("inference", {})
    if inference:
        parts.append(f"Inference results: {json.dumps(inference, indent=2)}")

    severity = data.get("severity", {})
    if severity:
        parts.append(f"Severity assessment: {json.dumps(severity, indent=2)}")

    if data.get("gradcam_b64") or data.get("gradcam"):
        parts.append("Grad-CAM visual explanation heatmap is available.")

    return "\n".join(parts) if parts else "Empty inspection payload."


def answer_question(question: str, payload: Dict[str, Any]) -> str:
    """Send a user question plus the inspection payload to Groq and return the answer."""
    payload_summary = _format_payload(payload)
    user_message = f"Question: {question}\n\nInspection Context:\n{payload_summary}"

    response = _get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.25,
        max_tokens=700,
    )
    return response.choices[0].message.content.strip()
