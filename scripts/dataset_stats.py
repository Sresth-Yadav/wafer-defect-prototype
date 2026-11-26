import os
from pathlib import Path
import PIL.Image as Image
import numpy as np

root = Path(r'C:/Users/asus/Desktop/Major 1/wafer-defect-prototype/Dataset')
image_paths = [p for p in root.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}]
print('Total images:', len(image_paths))
if not image_paths:
    raise SystemExit('No images found')

widths, heights = [], []
for p in image_paths:
    try:
        with Image.open(p) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
    except Exception as e:
        print('Failed', p, e)

print('Width: min', min(widths), 'max', max(widths), 'mean', np.mean(widths))
print('Height: min', min(heights), 'max', max(heights), 'mean', np.mean(heights))
