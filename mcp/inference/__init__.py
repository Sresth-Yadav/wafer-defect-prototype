# mcp/inference/__init__.py
"""
Inference MCP – loads a trained SmallUNet model and returns predictions.
"""
import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


# ─── Model definition (ResNet-18 transfer learning) ──────────────────────────
from torchvision import models as tv_models

def _build_resnet18(num_classes: int) -> nn.Module:
    """Build the same ResNet-18 architecture used in training."""
    model = tv_models.resnet18(weights=None)   # no download needed – we load saved weights
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )
    return model

# Keep SmallUNet as a fallback alias so old code doesn't break
SmallUNet = _build_resnet18


# ─── Lazy model loader ────────────────────────────────────────────────────────
_model: Optional[SmallUNet] = None
_class_names: Optional[list] = None
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "defect_classifier.pt"
CLASS_FILE = Path(__file__).parent.parent.parent / "models" / "class_names.txt"


def _load_model() -> SmallUNet:
    global _model, _class_names
    if _model is not None:
        return _model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Please run:  python scripts/train.py"
        )

    # Load class names if available
    if CLASS_FILE.exists():
        _class_names = CLASS_FILE.read_text().strip().splitlines()
        num_classes = len(_class_names)
    else:
        num_classes = 2
        _class_names = [str(i) for i in range(num_classes)]

    _model = _build_resnet18(num_classes=num_classes)
    _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    _model.eval()
    return _model


# ─── Public API ───────────────────────────────────────────────────────────────
def run_inference(image: np.ndarray) -> Dict[str, Any]:
    """
    Run inference on a preprocessed image.

    Args:
        image: float32 NumPy array of shape (256, 256, 3) with values in [0,1].
    Returns:
        dict with keys:
          - label        (str)   : predicted class name
          - confidence   (float) : probability of the predicted class
          - all_scores   (dict)  : {class_name: probability} for all classes
    """
    model = _load_model()
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

    with torch.no_grad():
        logits = model(tensor)                              # (1, num_classes)
        probs = torch.softmax(logits, dim=1).squeeze(0)    # (num_classes,)

    pred_idx = int(probs.argmax())
    confidence = float(probs[pred_idx])
    label = _class_names[pred_idx] if _class_names else str(pred_idx)

    all_scores = {
        (_class_names[i] if _class_names else str(i)): float(probs[i])
        for i in range(len(probs))
    }

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "all_scores": all_scores,
    }
