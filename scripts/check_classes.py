import sys
from pathlib import Path
from collections import Counter

ROOT = Path(r'C:/Users/asus/Desktop/Major 1/wafer-defect-prototype')
sys.path.insert(0, str(ROOT))

from mcp.preprocess.dataloader import WaferDefectDataset

ds = WaferDefectDataset(str(ROOT / 'Dataset'))
labels = [s[1] for s in ds.samples]
class_counts = Counter(labels)
print(f"{'Class':<30} {'Count':>6} {'%':>6}")
print("-" * 44)
for idx, count in sorted(class_counts.items(), key=lambda x: -x[1]):
    name = ds.idx_to_class[idx]
    pct = count / len(labels) * 100
    print(f"{name:<30} {count:>6}  {pct:>5.1f}%")
print(f"\nTotal: {len(labels)} images, {len(class_counts)} classes")
