"""
select_and_convert.py
=====================
For binary human segmentation dataset preparation:

1. Reads all segmentation masks from SegmentationClass/
2. Computes the percentage of human pixels (label == 1) per image
3. Selects the top 1000 images with the highest human pixel percentage
4. Converts those JPEG images to grayscale  -> output: output/images/<name>.jpg
5. Converts their mask images to binary BW  -> output: output/masks/<name>.png
   (0 = background, 255 = human)
6. Saves a CSV report with human pixel percentages

Assumptions
-----------
- Masks are grayscale PNGs with integer class labels (uint8).
- Label 1 = person / human  (PASCAL VOC convention; colour (192,128,128)).
- Label 0 = background.
- JPEGs and masks share the same stem name
  (e.g. 000001_001.jpg <-> 000001_001.png).
"""

import os
import csv
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# -- Paths ---------------------------------------------------------------------
BASE_DIR      = Path(__file__).parent
JPEG_DIR      = BASE_DIR / "JPEGImages"
MASK_DIR      = BASE_DIR / "SegmentationClass"
OUT_IMG_DIR   = BASE_DIR / "output" / "images"
OUT_MASK_DIR  = BASE_DIR / "output" / "masks"
REPORT_CSV    = BASE_DIR / "output" / "human_pixel_report.csv"

# -- Parameters ----------------------------------------------------------------
HUMAN_LABEL   = 1          # integer label for "person" in the segmentation masks
TOP_N         = 1000       # number of images to select

# -- Create output directories -------------------------------------------------
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

# -- Step 1: Scan all masks and compute human pixel percentage -----------------
print("Scanning all masks ...")

mask_files = sorted(MASK_DIR.glob("*.png"))
total = len(mask_files)
if total == 0:
    sys.exit("No PNG masks found in SegmentationClass/. Aborting.")

records = []   # list of (stem, human_pct)

for i, mask_path in enumerate(mask_files, 1):
    stem = mask_path.stem                # e.g. "000001_001"
    jpeg_path = JPEG_DIR / f"{stem}.jpg"

    if not jpeg_path.exists():
        # No corresponding image - skip
        continue

    mask_arr = np.array(Image.open(mask_path))
    total_pixels  = mask_arr.size
    human_pixels  = int((mask_arr == HUMAN_LABEL).sum())
    human_pct     = human_pixels / total_pixels * 100.0

    records.append((stem, human_pct))

    if i % 200 == 0 or i == total:
        print(f"  {i}/{total}  processed ...")

print(f"\nTotal valid image-mask pairs found: {len(records)}")

# -- Step 2: Sort and pick top N -----------------------------------------------
records.sort(key=lambda x: x[1], reverse=True)

top_records = records[:TOP_N]
print(f"\nTop {TOP_N} human-pixel-percentage range:")
print(f"  Highest : {top_records[0][1]:.2f}%  ({top_records[0][0]})")
print(f"  Lowest  : {top_records[-1][1]:.2f}%  ({top_records[-1][0]})")

# -- Step 3: Save CSV report ---------------------------------------------------
with open(REPORT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["rank", "stem", "human_pct"])
    for rank, (stem, pct) in enumerate(top_records, 1):
        writer.writerow([rank, stem, f"{pct:.4f}"])

print(f"\nReport saved -> {REPORT_CSV}")

# -- Step 4 & 5: Convert selected images and masks ----------------------------
print(f"\nConverting top {TOP_N} images and masks ...")

skipped_img  = 0
skipped_mask = 0

for idx, (stem, pct) in enumerate(top_records, 1):
    jpeg_path = JPEG_DIR  / f"{stem}.jpg"
    mask_path = MASK_DIR  / f"{stem}.png"

    # Convert JPEG to grayscale
    try:
        img = Image.open(jpeg_path).convert("L")          # "L" = 8-bit grayscale
        out_img_path = OUT_IMG_DIR / f"{stem}.jpg"
        img.save(out_img_path, "JPEG", quality=95)
    except Exception as e:
        print(f"  [WARN] Could not convert image {stem}: {e}")
        skipped_img += 1

    # Convert mask to binary black/white
    try:
        mask_arr  = np.array(Image.open(mask_path))
        # 255 where human (label==1), 0 everywhere else
        bw_arr    = np.where(mask_arr == HUMAN_LABEL, 255, 0).astype(np.uint8)
        bw_mask   = Image.fromarray(bw_arr, mode="L")
        out_mask_path = OUT_MASK_DIR / f"{stem}.png"
        bw_mask.save(out_mask_path)
    except Exception as e:
        print(f"  [WARN] Could not convert mask {stem}: {e}")
        skipped_mask += 1

    if idx % 100 == 0 or idx == TOP_N:
        print(f"  {idx}/{TOP_N} done ...")

# -- Summary -------------------------------------------------------------------
print("\n" + "="*60)
print("DONE!")
print(f"  Grayscale images saved -> {OUT_IMG_DIR}")
print(f"  Binary masks saved     -> {OUT_MASK_DIR}")
print(f"  CSV report saved       -> {REPORT_CSV}")
print(f"  Skipped images : {skipped_img}")
print(f"  Skipped masks  : {skipped_mask}")
print("="*60)
