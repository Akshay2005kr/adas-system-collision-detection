"""
Dataset Validator for YOLOv8
==============================
Checks everything before you start training:

  ✓ dataset.yaml  exists and is readable
  ✓ All split folders exist  (train / val / test)
  ✓ Every image  can be opened  (not corrupted)
  ✓ Every image  has a label file
  ✓ No orphan labels  (label with no matching image)
  ✓ Every label line is valid YOLO format  (5 numbers per line)
  ✓ All coords are normalised  0.0 – 1.0
  ✓ No bounding box is too tiny  (< 1 % of image)
  ✓ No bounding box goes outside the image
  ✓ All images are JPG  (uniform format check)
  ✓ Image size consistency report
  ✓ Class distribution per split
  ✓ Final PASS / FAIL summary

Run:
    pip install Pillow PyYAML
    python validate_dataset.py
"""

from pathlib import Path
from PIL import Image
import yaml
from collections import defaultdict

# ── Settings ──────────────────────────────────────────────
DATASET_ROOT  = Path("CLEAN_POTHOLE_DATASET")
YAML_FILE     = DATASET_ROOT / "dataset.yaml"
SPLITS        = ["train", "val", "test"]
MIN_BOX_SIZE  = 0.01   # boxes smaller than 1% of image side are flagged
# ──────────────────────────────────────────────────────────

PASS = "  [PASS]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"
INFO = "  [INFO]"

issues   = []   # critical errors
warnings = []   # non-critical

def fail(msg):
    print(f"{FAIL} {msg}")
    issues.append(msg)

def warn(msg):
    print(f"{WARN} {msg}")
    warnings.append(msg)

def ok(msg):
    print(f"{PASS} {msg}")

def info(msg):
    print(f"{INFO} {msg}")

# ══════════════════════════════════════════════════════════
print("=" * 60)
print("  YOLOV8 DATASET VALIDATOR")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. dataset.yaml
# ─────────────────────────────────────────────
print("\n[1/6] Checking dataset.yaml ...")

if not YAML_FILE.exists():
    fail(f"dataset.yaml not found at {YAML_FILE}")
    print("\nCannot continue without dataset.yaml. Exiting.")
    exit(1)

with open(YAML_FILE, "r") as f:
    cfg = yaml.safe_load(f)

for key in ("train", "val", "nc", "names"):
    if key not in cfg:
        fail(f"dataset.yaml missing key: '{key}'")
    else:
        ok(f"dataset.yaml has '{key}' = {cfg[key]}")

num_classes = cfg.get("nc", 0)
class_names = cfg.get("names", {})
info(f"Classes ({num_classes}): {class_names}")

# ─────────────────────────────────────────────
# 2. Folder structure
# ─────────────────────────────────────────────
print("\n[2/6] Checking folder structure ...")

for split in SPLITS:
    img_dir = DATASET_ROOT / "images" / split
    lbl_dir = DATASET_ROOT / "labels" / split
    if img_dir.exists():
        ok(f"images/{split}  exists")
    else:
        fail(f"images/{split}  MISSING")
    if lbl_dir.exists():
        ok(f"labels/{split}  exists")
    else:
        fail(f"labels/{split}  MISSING")

# ─────────────────────────────────────────────
# 3. Image + label checks per split
# ─────────────────────────────────────────────
print("\n[3/6] Checking images and labels ...")

valid_exts    = {".jpg", ".jpeg", ".png"}
split_stats   = {}
class_counts  = defaultdict(lambda: defaultdict(int))  # split → class_id → count
size_set      = set()

total_images  = 0
total_corrupt = 0
total_no_lbl  = 0
total_orphan  = 0
total_bad_lbl = 0
total_bad_box = 0

for split in SPLITS:
    img_dir = DATASET_ROOT / "images" / split
    lbl_dir = DATASET_ROOT / "labels" / split

    if not img_dir.exists():
        continue

    images = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in valid_exts
    )
    labels = {p.stem for p in lbl_dir.iterdir() if p.suffix == ".txt"} if lbl_dir.exists() else set()

    n_images   = len(images)
    n_corrupt  = 0
    n_no_lbl   = 0
    n_bad_lbl  = 0
    n_bad_box  = 0
    non_jpg    = 0

    print(f"\n  — {split.upper()}  ({n_images} images) —")

    for img_path in images:
        # Format check
        if img_path.suffix.lower() != ".jpg":
            warn(f"{split}/{img_path.name}  is not .jpg  ({img_path.suffix})")
            non_jpg += 1

        # Open image
        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                w, h = im.size
                size_set.add((w, h))
        except Exception as e:
            fail(f"Corrupted image: {split}/{img_path.name}  ({e})")
            n_corrupt += 1
            continue

        # Label exists?
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            fail(f"No label for: {split}/{img_path.name}")
            n_no_lbl += 1
            continue

        # Parse label
        with open(lbl_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        if len(lines) == 0:
            warn(f"Empty label file: {split}/{lbl_path.name}")

        for line_no, line in enumerate(lines, 1):
            parts = line.split()

            # Must have exactly 5 values
            if len(parts) != 5:
                fail(f"Bad format (expected 5 values, got {len(parts)}): "
                     f"{split}/{lbl_path.name}  line {line_no}: '{line}'")
                n_bad_lbl += 1
                continue

            try:
                cls, cx, cy, bw, bh = int(parts[0]), float(parts[1]), \
                                       float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                fail(f"Non-numeric value: {split}/{lbl_path.name}  line {line_no}: '{line}'")
                n_bad_lbl += 1
                continue

            # Class ID valid?
            if cls < 0 or cls >= num_classes:
                fail(f"Invalid class id {cls} (nc={num_classes}): "
                     f"{split}/{lbl_path.name}  line {line_no}")
                n_bad_lbl += 1

            # Coords in 0-1?
            coords_ok = all(0.0 <= v <= 1.0 for v in (cx, cy, bw, bh))
            if not coords_ok:
                fail(f"Coords out of range [0,1]: {split}/{lbl_path.name}  "
                     f"line {line_no}: {cx:.3f} {cy:.3f} {bw:.3f} {bh:.3f}")
                n_bad_box += 1

            # Box too small?
            if bw < MIN_BOX_SIZE or bh < MIN_BOX_SIZE:
                warn(f"Very small box (w={bw:.4f} h={bh:.4f}): "
                     f"{split}/{lbl_path.name}  line {line_no}")
                n_bad_box += 1

            # Box outside image?
            x1, y1 = cx - bw / 2, cy - bh / 2
            x2, y2 = cx + bw / 2, cy + bh / 2
            if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
                warn(f"Box clips image edge: {split}/{lbl_path.name}  "
                     f"line {line_no}  ({x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f})")
                n_bad_box += 1

            class_counts[split][cls] += 1

    # Orphan labels (labels with no matching image)
    image_stems = {p.stem for p in images}
    orphans = labels - image_stems
    for o in orphans:
        warn(f"Orphan label (no image): {split}/labels/{o}.txt")
        total_orphan += 1

    # Split summary
    if n_corrupt == 0 and n_no_lbl == 0 and n_bad_lbl == 0:
        ok(f"{split}: all {n_images} images OK")
    else:
        info(f"{split}: {n_corrupt} corrupt  |  {n_no_lbl} missing labels  "
             f"|  {n_bad_lbl} bad label lines  |  {n_bad_box} bad boxes")

    if non_jpg > 0:
        warn(f"{split}: {non_jpg} non-JPG images found — run convert_to_jpg.py")

    split_stats[split] = n_images
    total_images  += n_images
    total_corrupt += n_corrupt
    total_no_lbl  += n_no_lbl
    total_bad_lbl += n_bad_lbl
    total_bad_box += n_bad_box

# ─────────────────────────────────────────────
# 4. Image size report
# ─────────────────────────────────────────────
print("\n[4/6] Image size report ...")

if len(size_set) == 1:
    ok(f"All images are the same size: {next(iter(size_set))}")
elif len(size_set) <= 5:
    warn(f"Mixed image sizes ({len(size_set)} unique): {size_set}")
    info("YOLOv8 handles mixed sizes via imgsz — this is acceptable.")
else:
    warn(f"Many different image sizes ({len(size_set)} unique).")
    info("Consider resizing to a uniform size for faster training.")

# ─────────────────────────────────────────────
# 5. Class distribution
# ─────────────────────────────────────────────
print("\n[5/6] Class distribution ...")

for split in SPLITS:
    if split not in class_counts:
        continue
    total_boxes = sum(class_counts[split].values())
    print(f"\n  {split.upper()} — {total_boxes} total bounding boxes")
    for cls_id, count in sorted(class_counts[split].items()):
        name = class_names.get(cls_id, f"class_{cls_id}") if isinstance(class_names, dict) \
               else (class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}")
        pct  = count / total_boxes * 100 if total_boxes > 0 else 0
        bar  = "█" * int(pct / 2)
        print(f"    Class {cls_id} ({name:15s}): {count:5d}  {pct:5.1f}%  {bar}")

# ─────────────────────────────────────────────
# 6. Final summary
# ─────────────────────────────────────────────
print("\n[6/6] Final summary ...")
print("=" * 60)
print(f"  Total images checked : {total_images}")
print(f"  Corrupt images       : {total_corrupt}")
print(f"  Missing labels       : {total_no_lbl}")
print(f"  Orphan labels        : {total_orphan}")
print(f"  Bad label lines      : {total_bad_lbl}")
print(f"  Bad bounding boxes   : {total_bad_box}")
print(f"  Warnings             : {len(warnings)}")
print(f"  Critical errors      : {len(issues)}")
print("=" * 60)

if len(issues) == 0:
    print("\n  ✓  PASS — dataset is clean and ready for YOLOv8 training!")
    print("\n  Run:")
    print(f"    yolo train model=yolov8n.pt \\")
    print(f"      data={YAML_FILE.resolve()} \\")
    print(f"      epochs=100 imgsz=640 \\")
    print(f"      hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \\")
    print(f"      flipud=0.3 fliplr=0.5 mosaic=1.0 degrees=10 scale=0.5")
else:
    print(f"\n  ✗  FAIL — fix the {len(issues)} error(s) above before training.")
    print("\n  Issues found:")
    for i, issue in enumerate(issues, 1):
        print(f"    {i}. {issue}")

print("=" * 60)