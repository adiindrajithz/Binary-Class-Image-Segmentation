import csv
import sys
from pathlib import Path
from PIL import Image
import numpy as np

BASE_DIR = Path(__file__).parent
JPEG_DIR = BASE_DIR / "JPEGImages"
MASK_DIR = BASE_DIR / "SegmentationClass"
OUT_IMG_DIR = BASE_DIR / "output" / "images"
OUT_MASK_DIR = BASE_DIR / "output" / "masks"
REPORT_CSV = BASE_DIR / "output" / "human_pixel_report.csv"

HUMAN_LABEL = 1
TOP_N = 1000

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

print("Scanning all masks ...")

mask_files = sorted(MASK_DIR.glob("*.png"))
total = len(mask_files)
if total == 0:
    sys.exit("No PNG masks found in SegmentationClass/. Aborting.")

records = []

for i, mask_path in enumerate(mask_files, 1):
    stem = mask_path.stem
    jpeg_path = JPEG_DIR / f"{stem}.jpg"

    if not jpeg_path.exists():
        continue

    mask_arr = np.array(Image.open(mask_path))
    total_pixels = mask_arr.size
    human_pixels = int((mask_arr == HUMAN_LABEL).sum())
    human_pct = human_pixels / total_pixels * 100.0

    records.append((stem, human_pct))

    if i % 200 == 0 or i == total:
        print(f"  {i}/{total}  processed ...")

print(f"\nTotal valid image-mask pairs found: {len(records)}")

records.sort(key=lambda x: x[1], reverse=True)

top_records = records[:TOP_N]
print(f"\nTop {TOP_N} human-pixel-percentage range:")
print(f"  Highest : {top_records[0][1]:.2f}%  ({top_records[0][0]})")
print(f"  Lowest  : {top_records[-1][1]:.2f}%  ({top_records[-1][0]})")

with open(REPORT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["rank", "stem", "human_pct"])
    for rank, (stem, pct) in enumerate(top_records, 1):
        writer.writerow([rank, stem, f"{pct:.4f}"])

print(f"\nReport saved -> {REPORT_CSV}")
print(f"\nConverting top {TOP_N} images and masks ...")

skipped_img = 0
skipped_mask = 0

for idx, (stem, pct) in enumerate(top_records, 1):
    jpeg_path = JPEG_DIR / f"{stem}.jpg"
    mask_path = MASK_DIR / f"{stem}.png"

    try:
        img = Image.open(jpeg_path).convert("L")
        out_img_path = OUT_IMG_DIR / f"{stem}.jpg"
        img.save(out_img_path, "JPEG", quality=95)
    except Exception as e:
        print(f"  [WARN] Could not convert image {stem}: {e}")
        skipped_img += 1

    try:
        mask_arr = np.array(Image.open(mask_path))
        bw_arr = np.where(mask_arr == HUMAN_LABEL, 255, 0).astype(np.uint8)
        bw_mask = Image.fromarray(bw_arr, mode="L")
        out_mask_path = OUT_MASK_DIR / f"{stem}.png"
        bw_mask.save(out_mask_path)
    except Exception as e:
        print(f"  [WARN] Could not convert mask {stem}: {e}")
        skipped_mask += 1

    if idx % 100 == 0 or idx == TOP_N:
        print(f"  {idx}/{TOP_N} done ...")

print("\n" + "=" * 60)
print("DONE!")
print(f"  Grayscale images saved -> {OUT_IMG_DIR}")
print(f"  Binary masks saved     -> {OUT_MASK_DIR}")
print(f"  CSV report saved       -> {REPORT_CSV}")
print(f"  Skipped images : {skipped_img}")
print(f"  Skipped masks  : {skipped_mask}")
print("=" * 60)
