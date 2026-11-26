# ui/app.py
"""
Streamlit UI for the Wafer Defect Detection prototype.

Run with:
    cd "C:/Users/asus/Desktop/Major 1/wafer-defect-prototype"
    streamlit run ui/app.py
"""
import sys
import base64
import tempfile
import os
from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image

# Add project root to path so MCP packages are importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env locally; on Streamlit Cloud use secrets
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# Inject Streamlit Cloud secrets into env if available
try:
    import streamlit as _st
    if "GROQ_API_KEY" in _st.secrets:
        os.environ["GROQ_API_KEY"] = _st.secrets["GROQ_API_KEY"]
except Exception:
    pass

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wafer Defect Detector",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Wafer Defect Detection – AI Inspection Dashboard")
st.caption(
    "Powered by **LangGraph** | **MCP** | **Groq Mistral** | **Grad-CAM** | "
    "Built by an AI Engineering student"
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
    st.markdown("---")
    st.markdown(
        "**Tech stack**\n"
        "- LangGraph pipeline\n"
        "- MCP packages (preprocess, inference, explain, chatbot)\n"
        "- Groq Mixtral-8x7b chatbot\n"
        "- Grad-CAM + severity scoring\n"
        "- FastAPI backend\n"
        "- Streamlit frontend"
    )

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab_inspect, tab_chat, tab_history = st.tabs(
    ["🖼️ Inspect Image", "🤖 AI Chatbot", "📋 Inspection History"]
)

# ─── Session state ───────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# ─── Helper ──────────────────────────────────────────────────────────────────
def run_pipeline(image_path: str):
    """Import and call the LangGraph pipeline inline."""
    from mcp.preprocess import preprocess_image
    from mcp.inference import run_inference
    from mcp.explain import generate_gradcam

    img_arr = preprocess_image(image_path)
    inference_result = run_inference(img_arr)
    explain_result = generate_gradcam(img_arr, inference_result)

    return {
        "inference": inference_result,
        "gradcam_b64": explain_result.get("gradcam_b64", ""),
        "severity": explain_result.get("severity", {}),
    }


# ─── Tab 1: Inspect ───────────────────────────────────────────────────────────
with tab_inspect:
    uploaded = st.file_uploader(
        "Upload a wafer image (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"]
    )

    if uploaded:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            orig_img = Image.open(uploaded).convert("RGB")
            st.image(orig_img, use_container_width=True)

        # Save to temp file for the pipeline
        with tempfile.NamedTemporaryFile(suffix="." + uploaded.name.split(".")[-1], delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        with st.spinner("⏳ Running AI inspection pipeline..."):
            try:
                result = run_pipeline(tmp_path)
                st.session_state.last_result = result
            except FileNotFoundError as e:
                st.error(f"❌ Model not found. Please run training first:\n\n`python scripts/train.py`\n\n{e}")
                result = None
            except Exception as e:
                st.error(f"❌ Pipeline error: {e}")
                result = None
        os.unlink(tmp_path)

        if result:
            inference = result["inference"]
            severity = result["severity"]
            gradcam_b64 = result["gradcam_b64"]

            with col2:
                st.subheader("Grad-CAM Overlay")
                if gradcam_b64:
                    grad_bytes = base64.b64decode(gradcam_b64)
                    grad_img = Image.open(__import__("io").BytesIO(grad_bytes))
                    st.image(grad_img, use_container_width=True)

            # Metrics
            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🏷️ Predicted Class", inference["label"])
            m2.metric(
                "🎯 Confidence",
                f"{inference['confidence'] * 100:.1f}%",
                delta=("Above threshold" if inference["confidence"] > confidence_threshold else "Below threshold"),
            )
            m3.metric("⚠️ Severity", severity.get("severity_label", "N/A"))
            m4.metric("🔴 Affected Area", f"{severity.get('affected_area_pct', 0):.1f}%")

            # All class scores
            with st.expander("📊 All class scores"):
                scores = inference.get("all_scores", {})
                for cls_name, score in sorted(scores.items(), key=lambda x: -x[1]):
                    st.progress(float(score), text=f"{cls_name}: {score * 100:.1f}%")

            # Save to history
            st.session_state.history.append({
                "file": uploaded.name,
                "label": inference["label"],
                "confidence": inference["confidence"],
                "severity": severity.get("severity_label", "N/A"),
                "affected_pct": severity.get("affected_area_pct", 0),
            })
            st.success("✅ Inspection complete! Go to the **AI Chatbot** tab to ask questions about the result.")

# ─── Tab 2: Chatbot ───────────────────────────────────────────────────────────
with tab_chat:
    st.subheader("🤖 Ask the AI Inspection Assistant (powered by Groq Mistral)")
    st.caption(
        "Ask anything about the inspection result, wafer defects, Grad-CAM, severity scoring, etc."
    )

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask a question about the inspection or wafer concepts...")

    if prompt:
        # Add user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        payload = st.session_state.last_result or {}

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    from mcp.chatbot import answer_question
                    answer = answer_question(prompt, payload)
                except Exception as e:
                    answer = f"❌ Chatbot error: {e}\n\nMake sure your GROQ_API_KEY is set in `.env`."
            st.markdown(answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})

    if st.button("🗑️ Clear chat"):
        st.session_state.chat_messages = []
        st.rerun()

# ─── Tab 3: History ───────────────────────────────────────────────────────────
with tab_history:
    st.subheader("📋 Inspection History (this session)")

    if not st.session_state.history:
        st.info("No inspections yet. Upload an image in the **Inspect Image** tab.")
    else:
        import pandas as pd
        df = pd.DataFrame(st.session_state.history)
        df.index += 1
        st.dataframe(df, use_container_width=True)

        # Summary stats
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Inspected", len(df))
        defects = df[df["label"].str.lower().str.contains("defect")].shape[0]
        c2.metric("Defects Found", defects)
        c3.metric("Avg Confidence", f"{df['confidence'].mean() * 100:.1f}%")

        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()
