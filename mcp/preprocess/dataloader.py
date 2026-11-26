# mcp/preprocess/dataloader.py
"""
PyTorch Dataset class for the wafer-defect dataset.
Structure expected:
  Dataset/images/images/<class_name>/<img_file>
If no class labels exist, all images are assigned label -1 (inference only).
"""
import os
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class WaferDefectDataset(Dataset):
    """
    Reads all images from a root directory (recursively).
    The parent folder name is used as the class label.
    """

    def __init__(
        self,
        root_dir: str,
        img_size: Tuple[int, int] = (256, 256),
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root_dir)
        self.img_size = img_size
        self.transform = transform

        # Gather all image paths and infer class labels from folder names
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        all_paths = [p for p in self.root.rglob("*") if p.suffix.lower() in exts]

        # Build class → index map
        class_names = sorted({p.parent.name for p in all_paths})
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        self.samples: List[Tuple[Path, int]] = [
            (p, self.class_to_idx[p.parent.name]) for p in all_paths
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        else:
            img = img.resize(self.img_size, Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            img = torch.from_numpy(arr).permute(2, 0, 1)  # C, H, W

        return img, label

    def class_names(self) -> List[str]:
        return [self.idx_to_class[i] for i in range(len(self.class_to_idx))]
