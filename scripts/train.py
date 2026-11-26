# scripts/train.py
"""
Improved training script using:
  - ResNet18 Transfer Learning (pretrained on ImageNet)
  - Stratified train/val split (consistent splits)
  - Class-weighted CrossEntropyLoss (handles imbalance)
  - CosineAnnealingLR scheduler
  - Strong augmentation

Usage:
    cd "C:/Users/asus/Desktop/Major 1/wafer-defect-prototype"
    python scripts/train.py
"""
import os
import sys
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

# Redirect PyTorch hub/weights cache to our project folder (avoids permission issues)
TORCH_CACHE = str(ROOT / "models" / ".cache")
os.makedirs(TORCH_CACHE, exist_ok=True)
os.environ["TORCH_HOME"] = TORCH_CACHE
os.environ["XDG_CACHE_HOME"] = TORCH_CACHE

from mcp.preprocess.dataloader import WaferDefectDataset

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_ROOT   = ROOT / "Dataset"
MODEL_DIR   = ROOT / "models"
EPOCHS      = 20
BATCH_SIZE  = 32
LR          = 1e-4          # lower LR for fine-tuning pretrained weights
IMG_SIZE    = (224, 224)    # ResNet expects 224x224
VAL_SPLIT   = 0.2
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ─────────────────────────────────────────────────────────────────────────────

# ImageNet normalisation (required for pretrained ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_tf = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def build_resnet18(num_classes: int) -> nn.Module:
    """
    Load a pretrained ResNet-18 from local disk (avoids network download).
    Pretrained weights must be saved at: models/resnet18_imagenet.pth
    
    To download the weights once, run from a terminal WITH internet access:
        python -c "
        import torch
        from torchvision.models import resnet18, ResNet18_Weights
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        torch.save(m.state_dict(), r'models/resnet18_imagenet.pth')
        print('Saved!')
        "
    """
    LOCAL_WEIGHTS = ROOT / "models" / "resnet18_imagenet.pth"

    # Build the model architecture (no download)
    model = models.resnet18(weights=None)

    if LOCAL_WEIGHTS.exists():
        print(f"Loading pretrained weights from: {LOCAL_WEIGHTS}")
        state = torch.load(str(LOCAL_WEIGHTS), map_location="cpu")
        model.load_state_dict(state)
        print("Pretrained weights loaded ✅")
    else:
        print("⚠️  No pretrained weights found — training from scratch.")
        print(f"   Expected: {LOCAL_WEIGHTS}")
        print("   Run the download command shown in the script docstring.")

    # Freeze all layers, then unfreeze last block + head
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )
    return model


def compute_class_weights(labels, num_classes, device):
    """Inverse-frequency class weights to penalise majority classes."""
    counts = Counter(labels)
    total = len(labels)
    weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32).to(device)


def main():
    print(f"Using device: {DEVICE}")

    # ── Dataset with transforms applied per-split ─────────────────────────
    # Build both transform variants from the same file list
    full_train = WaferDefectDataset(str(DATA_ROOT), img_size=IMG_SIZE, transform=train_tf)
    full_val   = WaferDefectDataset(str(DATA_ROOT), img_size=IMG_SIZE, transform=val_tf)

    class_names  = full_train.class_names()
    num_classes  = len(class_names)
    all_labels   = [s[1] for s in full_train.samples]

    print(f"Classes ({num_classes}): {class_names}")
    print(f"Total samples: {len(full_train)}")

    # Class distribution
    counts = Counter(all_labels)
    for idx, name in enumerate(class_names):
        print(f"  {name:<20}: {counts[idx]:>4} samples ({counts[idx]/len(all_labels)*100:.1f}%)")

    # ── Stratified split ─────────────────────────────────────────────────
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT, random_state=42)
    train_idx, val_idx = next(sss.split(range(len(all_labels)), all_labels))

    train_ds = Subset(full_train, train_idx)
    val_ds   = Subset(full_val,   val_idx)

    print(f"\nTrain: {len(train_ds)}  |  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────
    model = build_resnet18(num_classes).to(DEVICE)
    print(f"\nModel: ResNet-18 (pretrained ImageNet) → {num_classes} classes")

    # Class-weighted loss
    class_weights = compute_class_weights(all_labels, num_classes, DEVICE)
    print(f"Class weights: {[round(w.item(), 2) for w in class_weights]}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Only update unfrozen params
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_acc = 0.0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Epoch':<8}{'Loss':>8}{'Train Acc':>12}{'Val Acc':>10}{'LR':>12}")
    print("─" * 52)

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = train_correct = train_total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * imgs.size(0)
            train_correct += (out.argmax(1) == labels).sum().item()
            train_total   += imgs.size(0)
        scheduler.step()

        # Validate
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total   += imgs.size(0)

        t_acc = train_correct / train_total * 100
        v_acc = val_correct   / val_total   * 100 if val_total else 0
        avg_l = train_loss    / train_total
        cur_lr = scheduler.get_last_lr()[0]

        print(f"{epoch:>3}/{EPOCHS:<4}  {avg_l:>8.4f}  {t_acc:>9.1f}%  {v_acc:>7.1f}%  {cur_lr:>11.2e}")

        # Save best model
        if v_acc >= best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), MODEL_DIR / "defect_classifier.pt")
            (MODEL_DIR / "class_names.txt").write_text("\n".join(class_names))

    print(f"\n✅ Training complete! Best val accuracy: {best_val_acc:.1f}%")
    print(f"   Model: {MODEL_DIR / 'defect_classifier.pt'}")
    print(f"   Classes: {MODEL_DIR / 'class_names.txt'}")


if __name__ == "__main__":
    main()
