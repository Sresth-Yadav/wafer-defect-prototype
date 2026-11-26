# scripts/benchmark_encoders.py
"""
Vision Encoder & Loss Ablation Script.
Compares backbones (resnet18, convnext_tiny, swin_t) with Focal Loss +
WeightedRandomSampler to improve minority-class recall (e.g. rolled_pit).

Usage:
    python scripts/benchmark_encoders.py --backbone resnet18 --focal
    python scripts/benchmark_encoders.py --backbone convnext_tiny --focal
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

from mcp.preprocess.dataloader import WaferDefectDataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FocalLoss(nn.Module):
    """Multi-class Focal Loss to focus training on hard/minority examples."""

    def __init__(self, alpha=None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def build_backbone(name: str, num_classes: int) -> nn.Module:
    if name == "resnet18":
        m = models.resnet18(weights=None)
        local_w = ROOT / "models" / "resnet18_imagenet.pth"
        if local_w.exists():
            m.load_state_dict(torch.load(str(local_w), map_location="cpu"))
        m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, num_classes))
        return m
    if name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m
    if name == "swin_t":
        m = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        m.head = nn.Linear(m.head.in_features, num_classes)
        return m
    raise ValueError(f"Unsupported backbone: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "convnext_tiny", "swin_t"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--focal", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    ds_train = WaferDefectDataset(str(ROOT / "Dataset"), transform=tf_train)
    ds_val = WaferDefectDataset(str(ROOT / "Dataset"), transform=tf_val)
    labels = [s[1] for s in ds_train.samples]
    num_classes = len(ds_train.class_names())

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(range(len(labels)), labels))

    train_labels = [labels[i] for i in train_idx]
    counts = Counter(train_labels)
    sample_weights = [1.0 / counts[y] for y in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(Subset(ds_train, train_idx), batch_size=32, sampler=sampler)
    val_loader = DataLoader(Subset(ds_val, val_idx), batch_size=32, shuffle=False)

    model = build_backbone(args.backbone, num_classes).to(device)
    criterion = FocalLoss(gamma=2.0) if args.focal else nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
                trues.extend(y.tolist())
        macro_f1 = f1_score(trues, preds, average="macro")
        acc = sum(p == t for p, t in zip(preds, trues)) / len(trues) * 100
        best_f1 = max(best_f1, macro_f1)
        print(f"Epoch {epoch:02d}/{args.epochs} | Val Acc: {acc:.2f}% | Macro F1: {macro_f1:.4f}")

    out_path = ROOT / "reports" / "ablation_results.json"
    existing = json.loads(out_path.read_text()) if out_path.exists() else []
    existing.append({
        "backbone": args.backbone,
        "loss": "FocalLoss(gamma=2)" if args.focal else "CrossEntropy",
        "sampler": "WeightedRandomSampler",
        "best_macro_f1": round(best_f1, 4),
    })
    out_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"Saved ablation entry to {out_path}")


if __name__ == "__main__":
    main()
