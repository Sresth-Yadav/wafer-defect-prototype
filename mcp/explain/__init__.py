# mcp/explain/__init__.py
"""
Explain MCP – generates a Grad-CAM heatmap overlay and severity score.
"""
import cv2
import base64
import numpy as np
import torch
from typing import Any, Dict


def _compute_gradcam(model, tensor: torch.Tensor) -> np.ndarray:
    """
    Compute Grad-CAM heatmap on the last convolutional layer of the bottleneck.
    Returns a (H, W) float32 array in [0, 1].
    """
    activations = []
    gradients = []

    # We hook the last conv layer in ResNet's layer4
    target_layer = model.layer4[1].conv2

    fwd_handle = target_layer.register_forward_hook(
        lambda m, inp, out: activations.append(out.detach())
    )
    bwd_handle = target_layer.register_backward_hook(
        lambda m, gi, go: gradients.append(go[0].detach())
    )

    model.zero_grad()
    logits = model(tensor)                      # (1, num_classes)
    pred_class = logits.argmax(dim=1).item()
    score = logits[0, pred_class]
    score.backward()

    fwd_handle.remove()
    bwd_handle.remove()

    if not activations or not gradients:
        # Fallback: return a blank heatmap
        return np.zeros((256, 256), dtype=np.float32)

    act = activations[0][0]       # (C, H, W)
    grad = gradients[0][0]        # (C, H, W)
    weights = grad.mean(dim=(1, 2))                       # (C,)
    cam = (weights[:, None, None] * act).sum(0).cpu().numpy()  # (H, W)
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.astype(np.float32)


def _compute_severity(heatmap: np.ndarray, confidence: float) -> Dict[str, Any]:
    """
    Derive a simple severity score from the heatmap and model confidence.
    Returns a dict with:
      - severity_score (float 0-1)
      - affected_area_pct (float 0-100)
      - severity_label (str: Low / Medium / High)
    """
    threshold = 0.5
    affected = float((heatmap > threshold).mean()) * 100.0  # percentage of pixels
    severity_score = round(0.5 * confidence + 0.5 * (affected / 100.0), 4)

    if severity_score < 0.35:
        label = "Low"
    elif severity_score < 0.65:
        label = "Medium"
    else:
        label = "High"

    return {
        "severity_score": severity_score,
        "affected_area_pct": round(affected, 2),
        "severity_label": label,
    }


def generate_gradcam(image: np.ndarray, inference: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a Grad-CAM overlay and severity assessment.

    Args:
        image:     float32 NumPy array (256, 256, 3) in [0, 1].
        inference: dict returned by run_inference().

    Returns:
        dict with keys:
          - gradcam_b64  (str): base64-encoded PNG of the overlay.
          - severity     (dict): severity_score, affected_area_pct, severity_label.
    """
    # Import here to avoid circular imports with inference module
    from mcp.inference import _load_model

    model = _load_model()
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    # Generate Grad-CAM
    heatmap = _compute_gradcam(model, tensor)                      # (H', W')
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))  # (256, 256)

    # Colorize heatmap and overlay on original image
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    img_uint8 = np.uint8(255 * image)                              # RGB uint8
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap_color, 0.4, 0)

    # Encode overlay to base64 PNG
    _, buffer = cv2.imencode(".png", overlay)
    b64 = base64.b64encode(buffer).decode("utf-8")

    # Severity
    severity = _compute_severity(heatmap_resized, inference.get("confidence", 0.5))

    return {"gradcam_b64": b64, "severity": severity}
