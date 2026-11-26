# tests/test_pipeline.py
"""
Unit tests for the wafer-defect pipeline components.
Run with:  pytest tests/ -v
"""
import sys
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ─── Preprocess ───────────────────────────────────────────────────────────────
def test_preprocess_returns_correct_shape(tmp_path):
    """preprocess_image should return a float32 (224,224,3) ImageNet-normalized array."""
    from PIL import Image
    img_path = tmp_path / "test.png"
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    img.save(str(img_path))

    from mcp.preprocess import preprocess_image
    result = preprocess_image(str(img_path))

    assert result.shape == (224, 224, 3), f"Expected (224,224,3) got {result.shape}"
    assert result.dtype == np.float32


def test_preprocess_raises_on_missing_file():
    from mcp.preprocess import preprocess_image
    with pytest.raises(FileNotFoundError):
        preprocess_image("nonexistent_path/image.png")


# ─── Dataset Loader ───────────────────────────────────────────────────────────
def test_dataset_loader(tmp_path):
    """WaferDefectDataset should load images and return (tensor, int) pairs."""
    from PIL import Image
    # Create dummy class folders
    for cls in ["defect", "clean"]:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        img.save(str(cls_dir / "img001.png"))

    from mcp.preprocess.dataloader import WaferDefectDataset
    ds = WaferDefectDataset(str(tmp_path))
    assert len(ds) == 2

    img_tensor, label = ds[0]
    import torch
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 256, 256)
    assert label in {0, 1}


# ─── Inference ────────────────────────────────────────────────────────────────
def test_inference_returns_correct_keys(tmp_path):
    """run_inference should return a dict with label, confidence, all_scores."""
    import torch
    from mcp.inference import SmallUNet

    # Create a dummy model and save it
    model = SmallUNet(num_classes=2)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    torch.save(model.state_dict(), models_dir / "defect_classifier.pt")
    (models_dir / "class_names.txt").write_text("clean\ndefect")

    # Monkey-patch the model path in the inference module
    import mcp.inference as inf_mod
    original_path = inf_mod.MODEL_PATH
    original_cls = inf_mod.CLASS_FILE
    inf_mod.MODEL_PATH = models_dir / "defect_classifier.pt"
    inf_mod.CLASS_FILE = models_dir / "class_names.txt"
    inf_mod._model = None  # reset lazy cache
    inf_mod._class_names = None

    dummy_image = np.zeros((256, 256, 3), dtype=np.float32)
    result = inf_mod.run_inference(dummy_image)

    # Restore original path
    inf_mod.MODEL_PATH = original_path
    inf_mod.CLASS_FILE = original_cls
    inf_mod._model = None
    inf_mod._class_names = None

    assert "label" in result
    assert "confidence" in result
    assert "all_scores" in result
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0


# ─── Chatbot ──────────────────────────────────────────────────────────────────
def test_chatbot_returns_string():
    """answer_question should return a non-empty string given a mock Groq response."""
    mock_choice = MagicMock()
    mock_choice.message.content = "The confidence score indicates model certainty."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    with patch("mcp.chatbot._get_client") as mock_get_client:
        mock_get_client.return_value.chat.completions.create.return_value = mock_response
        from mcp.chatbot import answer_question
        answer = answer_question(
            "What does the confidence score mean?",
            {"inference": {"label": "defect", "confidence": 0.9}}
        )
        assert isinstance(answer, str)
        assert len(answer) > 0


# ─── Conditional LangGraph Routing, MCP Server & PDF Export ──────────────────
def test_langgraph_conditional_routing(tmp_path):
    """Verify conditional LangGraph routes low-confidence predictions through human_review."""
    from PIL import Image
    from workflows.wafer_pipeline import build_graph

    img_path = tmp_path / "wafer.png"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(str(img_path))

    fake_inf = {"label": "rolled_pit", "confidence": 0.42, "all_scores": {"rolled_pit": 0.42}}
    fake_exp = {"gradcam_b64": "ZmFrZQ==", "severity": {"severity_score": 0.5, "affected_area_pct": 20.0, "severity_label": "Medium"}}

    with patch("workflows.wafer_pipeline.run_inference", return_value=fake_inf), \
         patch("workflows.wafer_pipeline.generate_gradcam", return_value=fake_exp):
        graph = build_graph(enable_llm_diagnosis=False)
        res = graph.invoke({"path": str(img_path)})

    out = res["output"]
    assert out["needs_review"] is True
    assert out["route_taken"] == ["ingest", "infer", "explain", "human_review", "diagnose", "aggregate"]
    assert "rolled_pit" in out["diagnostic_report"]


def test_mcp_server_metrics_tool():
    """Verify FastMCP server exposes get_model_metrics and returns valid JSON."""
    import json
    from mcp_server import get_model_metrics

    raw = get_model_metrics()
    data = json.loads(raw)
    assert "model" in data or "error" in data


def test_pdf_export_bytes():
    """Verify PDF report generator returns valid %PDF bytes."""
    from dashboard.main import _build_pdf_bytes

    pdf = _build_pdf_bytes({
        "filename": "sample.png",
        "inference": {"label": "deburring", "confidence": 0.99},
        "severity": {"severity_label": "High", "severity_score": 0.88, "affected_area_pct": 75.0},
        "route_taken": ["ingest", "infer", "explain", "diagnose", "aggregate"],
        "diagnostic_report": "Test diagnostic summary.",
    })
    assert pdf.startswith(b"%PDF")
