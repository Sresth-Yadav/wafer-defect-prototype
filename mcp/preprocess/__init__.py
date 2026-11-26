# mcp/preprocess/__init__.py
"""
Preprocess MCP – real implementation using OpenCV.
"""
import os
import cv2
import numpy as np
from pathlib import Path


IMG_SIZE = (224, 224)  # ResNet-18 expects 224x224

# ImageNet normalisation to match training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load an image, convert to RGB, resize to 256x256, normalize to [0, 1].

    Args:
        image_path: path to any .png / .jpg / .jpeg / .tif image.
    Returns:
        NumPy float32 array of shape (256, 256, 3) with values in [0, 1].
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    # Apply ImageNet normalisation (same as training transforms)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    img  = (img - mean) / std
    return img
