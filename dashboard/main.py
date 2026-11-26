"""
FastAPI server exposing the wafer-defect LangGraph pipeline, EDA/metrics API,
Groq AI chatbot, and the React inspection dashboard.
"""

import os
import io
import json
import base64
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
SAMPLES_DIR = ROOT / "samples"
REPORTS_FILE = ROOT / "reports" / "metrics_eda.json"
TMP_DIR = ROOT / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="WaferAI — Semiconductor Inspection & Agentic Analytics", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from workflows.wafer_pipeline import build_graph
from mcp.chatbot import answer_question

graph = build_graph()


@app.get("/", response_class=HTMLResponse)
async def serve_react_ui():
    """Serve the React inspection dashboard."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>React UI not found at dashboard/static/index.html</h1>", status_code=404)
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "ResNet-18", "llm": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")}


@app.get("/api/metrics")
async def get_metrics_and_eda():
    """Return stored EDA statistics and model evaluation metrics."""
    if not REPORTS_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="metrics_eda.json not found. Run: python scripts/evaluate.py",
        )
    return JSONResponse(content=json.loads(REPORTS_FILE.read_text(encoding="utf-8")))


@app.get("/api/samples")
async def list_samples():
    """List built-in sample wafer images for 1-click demo inspection."""
    if not SAMPLES_DIR.exists():
        return {"samples": []}
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    items = [
        {"name": p.name, "label": p.stem.replace("_", " ")}
        for p in sorted(SAMPLES_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in exts
    ]
    return {"samples": items}


@app.get("/api/samples/{sample_name}")
async def get_sample_image(sample_name: str):
    """Serve a built-in sample wafer image."""
    target = (SAMPLES_DIR / sample_name).resolve()
    if not target.is_file() or target.parent != SAMPLES_DIR.resolve():
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(target)


@app.post("/run", summary="Upload a wafer image and run the LangGraph inspection pipeline")
async def run_pipeline(file: UploadFile = File(...)):
    tmp_path = TMP_DIR / (file.filename or "upload.png")
    try:
        with tmp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store uploaded file: {e}")
    finally:
        await file.close()

    try:
        state_out = graph.invoke({"path": str(tmp_path)})
        result = {"output": state_out.get("output", {})}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

    return JSONResponse(content=result)


@app.post("/chat", summary="Ask the Groq AI assistant about inspection results or wafer concepts")
async def chat_endpoint(payload: dict):
    question = payload.get("question")
    inspection_payload = payload.get("payload", {})
    if not question:
        raise HTTPException(status_code=400, detail="'question' field is required.")
    try:
        answer = answer_question(question, inspection_payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {e}")
    return {"answer": answer}


def _build_pdf_bytes(payload: dict) -> bytes:
    """Generate a structured inspection PDF using ReportLab if installed, else Pillow."""
    data = payload.get("output", payload)
    inf = data.get("inference", {})
    sev = data.get("severity", {})
    filename = payload.get("filename", "wafer_image.png")
    route = " -> ".join(data.get("route_taken", ["ingest", "infer", "explain", "aggregate"]))
    diag = data.get("diagnostic_report", "No diagnostic notes.")
    gradcam_b64 = data.get("gradcam_b64", "")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader

        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        w, h = A4
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, h - 50, "WaferAI - Semiconductor Inspection Report")
        c.setFont("Helvetica", 10)
        c.drawString(40, h - 70, f"Generated: {ts}   |   File: {filename}")
        c.drawString(40, h - 85, f"LangGraph Route: {route}")

        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, h - 115, f"Predicted Class: {inf.get('label', 'N/A')}  ({inf.get('confidence', 0)*100:.2f}%)")
        c.drawString(40, h - 135, f"Severity: {sev.get('severity_label', 'N/A')}  |  Affected Area: {sev.get('affected_area_pct', 0)}%")

        if gradcam_b64:
            img_bytes = base64.b64decode(gradcam_b64)
            c.drawImage(ImageReader(io.BytesIO(img_bytes)), 40, h - 380, width=220, height=220, preserveAspectRatio=True)

        c.setFont("Helvetica-Bold", 11)
        c.drawString(40, h - 410, "Automated Diagnostic Summary:")
        c.setFont("Helvetica", 9)
        y = h - 428
        for line in diag.splitlines():
            for chunk_start in range(0, len(line), 95):
                if y < 50:
                    break
                c.drawString(40, y, line[chunk_start:chunk_start + 95])
                y -= 13
        c.showPage()
        c.save()
        return buf.getvalue()
    except ImportError:
        from PIL import Image, ImageDraw
        page = Image.new("RGB", (1240, 1754), (248, 250, 252))
        draw = ImageDraw.Draw(page)
        draw.rectangle([0, 0, 1240, 140], fill=(15, 23, 42))
        draw.text((50, 45), "WaferAI - Semiconductor Defect Inspection Report", fill=(255, 255, 255))
        draw.text((50, 85), f"Generated: {ts}   |   File: {filename}", fill=(148, 163, 184))

        y = 180
        lines = [
            f"Predicted Defect Class : {inf.get('label', 'N/A')}",
            f"Model Confidence       : {inf.get('confidence', 0) * 100:.2f}%",
            f"Severity Level         : {sev.get('severity_label', 'N/A')} (score: {sev.get('severity_score', 0)})",
            f"Grad-CAM Affected Area : {sev.get('affected_area_pct', 0)}%",
            f"LangGraph Route        : {route}",
            f"Human Review Flagged   : {data.get('needs_review', False)}",
        ]
        for ln in lines:
            draw.text((50, y), ln, fill=(15, 23, 42))
            y += 36

        if gradcam_b64:
            try:
                g_img = Image.open(io.BytesIO(base64.b64decode(gradcam_b64))).convert("RGB").resize((420, 420))
                page.paste(g_img, (50, y + 20))
                y += 460
            except Exception:
                pass

        draw.text((50, y + 10), "Automated Diagnostic Summary:", fill=(15, 23, 42))
        y += 45
        for raw_line in diag.splitlines():
            for i in range(0, len(raw_line), 95):
                if y > 1680:
                    break
                draw.text((50, y), raw_line[i:i + 95], fill=(51, 65, 85))
                y += 26

        buf = io.BytesIO()
        page.save(buf, format="PDF")
        return buf.getvalue()


@app.post("/export-pdf", summary="Export an inspection result as a downloadable PDF report")
async def export_pdf(payload: dict):
    try:
        pdf_bytes = _build_pdf_bytes(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation error: {e}")
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=wafer_inspection_report.pdf"},
    )
