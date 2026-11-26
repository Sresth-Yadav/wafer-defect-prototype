# scripts/evaluate.py
"""
Full EDA + Model Evaluation script.
Computes dataset EDA statistics and held-out validation metrics,
prints a summary, and saves everything to reports/metrics_eda.json.

Run: python scripts/evaluate.py
"""
import os
import sys
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

TORCH_CACHE = str(ROOT / "models" / ".cache")
os.makedirs(TORCH_CACHE, exist_ok=True)
os.environ["TORCH_HOME"] = TORCH_CACHE

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

from mcp.preprocess.dataloader import WaferDefectDataset
from mcp.inference import _build_resnet18, CLASS_FILE, MODEL_PATH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

val_tf = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def compute_eda(full_ds: WaferDefectDataset, class_names: list) -> dict:
    """Compute real Exploratory Data Analysis stats on the dataset."""
    labels = [s[1] for s in full_ds.samples]
    counts = Counter(labels)
    total = len(labels)

    widths, heights, aspect_ratios, file_sizes_kb = [], [], [], []
    formats = Counter()

    for img_path, _ in full_ds.samples:
        formats[img_path.suffix.lower()] += 1
        file_sizes_kb.append(img_path.stat().st_size / 1024.0)
        with Image.open(img_path) as im:
            w, h = im.size
            widths.append(w)
            heights.append(h)
            aspect_ratios.append(w / h if h > 0 else 0.0)

    class_distribution = [
        {
            "class_name": name,
            "count": int(counts[idx]),
            "percentage": round(counts[idx] / total * 100, 2),
        }
        for idx, name in enumerate(class_names)
    ]

    return {
        "total_images": total,
        "num_classes": len(class_names),
        "class_distribution": class_distribution,
        "file_formats": dict(formats),
        "image_dimensions": {
            "width_min": int(min(widths)),
            "width_max": int(max(widths)),
            "width_mean": round(float(np.mean(widths)), 1),
            "height_min": int(min(heights)),
            "height_max": int(max(heights)),
            "height_mean": round(float(np.mean(heights)), 1),
            "aspect_ratio_mean": round(float(np.mean(aspect_ratios)), 2),
            "avg_file_size_kb": round(float(np.mean(file_sizes_kb)), 1),
        },
    }


def main():
    class_names = CLASS_FILE.read_text().strip().splitlines()
    num_classes = len(class_names)
    print(f"Classes: {class_names}")

    full_ds = WaferDefectDataset(str(ROOT / "Dataset"), img_size=IMG_SIZE, transform=val_tf)
    all_labels = [s[1] for s in full_ds.samples]

    print("Computing dataset EDA...")
    eda_stats = compute_eda(full_ds, class_names)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(range(len(all_labels)), all_labels))
    val_ds = Subset(full_ds, val_idx)
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = _build_resnet18(num_classes)
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location="cpu"))
    model.to(DEVICE).eval()
    print(f"Model loaded from: {MODEL_PATH}\n")

    all_preds, all_true = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_true.extend(labels.tolist())

    correct = sum(p == t for p, t in zip(all_preds, all_true))
    total = len(all_true)
    acc = correct / total * 100

    report_dict = classification_report(
        all_true, all_preds, target_names=class_names, digits=4, output_dict=True
    )
    cm = confusion_matrix(all_true, all_preds).tolist()

    print(f"{'=' * 50}")
    print(f"  Overall Accuracy: {acc:.2f}%  ({correct}/{total} correct)")
    print(f"{'=' * 50}\n")
    print(classification_report(all_true, all_preds, target_names=class_names, digits=3))
    print("Confusion Matrix (rows=true, cols=predicted):")
    print(np.array(cm))

    # Save structured EDA + evaluation metrics to reports/metrics_eda.json
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_file = reports_dir / "metrics_eda.json"

    payload = {
        "model": {
            "architecture": "ResNet-18 (Transfer Learning)",
            "input_size": "224x224",
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "overall_accuracy": round(acc, 2),
            "correct": correct,
            "total_val": total,
            "macro_avg": {
                "precision": round(report_dict["macro avg"]["precision"], 4),
                "recall": round(report_dict["macro avg"]["recall"], 4),
                "f1_score": round(report_dict["macro avg"]["f1-score"], 4),
            },
            "weighted_avg": {
                "precision": round(report_dict["weighted avg"]["precision"], 4),
                "recall": round(report_dict["weighted avg"]["recall"], 4),
                "f1_score": round(report_dict["weighted avg"]["f1-score"], 4),
            },
            "per_class": [
                {
                    "class_name": name,
                    "precision": round(report_dict[name]["precision"], 4),
                    "recall": round(report_dict[name]["recall"], 4),
                    "f1_score": round(report_dict[name]["f1-score"], 4),
                    "support": int(report_dict[name]["support"]),
                }
                for name in class_names
            ],
            "confusion_matrix": cm,
            "class_names": class_names,
        },
        "eda": eda_stats,
    }

    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved EDA & evaluation metrics to: {out_file}")


if __name__ == "__main__":
    main()
