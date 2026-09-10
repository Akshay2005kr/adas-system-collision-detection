"""
Dataset Auto-Fixer
===================
Fixes all 3 warning types found by validate_dataset.py:

  1. EMPTY LABEL FILES  → deletes image + label pair
     (image has no pothole annotation = useless for training)

  2. BOX CLIPS EDGE     → clamps coords to [0.0, 1.0]
     (box drawn slightly outside image boundary → clip it in)

  3. VERY SMALL BOXES   → removes that annotation line
     (boxes smaller than MIN_BOX_SIZE are noise / too far away)

Run:
    python fix_dataset.py
"""

from pathlib import Path

# ── Settings ──────────────────────────────────────────────
DATASET_ROOT = Path("CLEAN_POTHOLE_DATASET")
SPLITS       = ["train", "val", "test"]
MIN_BOX_SIZE = 0.01      # remove boxes smaller than 1% of image side
# ──────────────────────────────────────────────────────────

stats = {
    "empty_removed"   : 0,
    "boxes_clamped"   : 0,
    "boxes_removed"   : 0,
    "files_modified"  : 0,
}

print("=" * 55)
print("  DATASET AUTO-FIXER")
print("=" * 55)

for split in SPLITS:
    img_dir = DATASET_ROOT / "images" / split
    lbl_dir = DATASET_ROOT / "labels" / split

    if not img_dir.exists() or not lbl_dir.exists():
        continue

    print(f"\n[{split.upper()}]")

    label_files = sorted(lbl_dir.glob("*.txt"))

    for lbl_path in label_files:

        # ── 1. EMPTY LABEL FILE ────────────────────────────
        raw = lbl_path.read_text().strip()
        if raw == "":
            # Find and delete matching image
            deleted_img = False
            for ext in (".jpg", ".jpeg", ".png"):
                img_path = img_dir / (lbl_path.stem + ext)
                if img_path.exists():
                    img_path.unlink()
                    deleted_img = True
                    break
            lbl_path.unlink()
            print(f"  [EMPTY REMOVED] {lbl_path.name}"
                  + ("  +image" if deleted_img else "  (no image found)"))
            stats["empty_removed"] += 1
            continue

        # ── 2 & 3. FIX BOX COORDS ─────────────────────────
        lines     = raw.splitlines()
        new_lines = []
        file_changed = False

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                new_lines.append(line)   # leave malformed lines alone
                continue

            cls                    = parts[0]
            cx, cy, bw, bh         = map(float, parts[1:])

            # ── 3. Remove tiny boxes ──────────────────────
            if bw < MIN_BOX_SIZE or bh < MIN_BOX_SIZE:
                print(f"  [TINY REMOVED]  {lbl_path.name}  "
                      f"w={bw:.4f} h={bh:.4f}")
                stats["boxes_removed"] += 1
                file_changed = True
                continue

            # ── 2. Clamp edge-clipping boxes ─────────────
            # Recalculate edges, clamp, recalculate center+size
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            x1c = max(0.0, min(1.0, x1))
            y1c = max(0.0, min(1.0, y1))
            x2c = max(0.0, min(1.0, x2))
            y2c = max(0.0, min(1.0, y2))

            if (x1c, y1c, x2c, y2c) != (x1, y1, x2, y2):
                new_cx = (x1c + x2c) / 2
                new_cy = (y1c + y2c) / 2
                new_bw = x2c - x1c
                new_bh = y2c - y1c

                # After clamping, box might become too small → remove it
                if new_bw < MIN_BOX_SIZE or new_bh < MIN_BOX_SIZE:
                    print(f"  [CLAMP→TINY REMOVED] {lbl_path.name}  "
                          f"after clamp: w={new_bw:.4f} h={new_bh:.4f}")
                    stats["boxes_removed"] += 1
                    file_changed = True
                    continue

                print(f"  [CLAMPED]       {lbl_path.name}  "
                      f"({x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}) "
                      f"→ ({x1c:.2f},{y1c:.2f},{x2c:.2f},{y2c:.2f})")
                cx, cy, bw, bh = new_cx, new_cy, new_bw, new_bh
                stats["boxes_clamped"] += 1
                file_changed = True

            new_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # Write back only if something changed
        if file_changed:
            lbl_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
            stats["files_modified"] += 1

            # If all boxes were removed → delete image + label
            if len(new_lines) == 0:
                lbl_path.unlink()
                for ext in (".jpg", ".jpeg", ".png"):
                    img_path = img_dir / (lbl_path.stem + ext)
                    if img_path.exists():
                        img_path.unlink()
                        break
                print(f"  [ALL BOXES GONE → REMOVED] {lbl_path.name}")
                stats["empty_removed"] += 1
                stats["files_modified"] -= 1

print("\n" + "=" * 55)
print("  FIX COMPLETE")
print("=" * 55)
print(f"  Empty image+label pairs removed : {stats['empty_removed']}")
print(f"  Bounding boxes clamped to edge  : {stats['boxes_clamped']}")
print(f"  Tiny boxes removed              : {stats['boxes_removed']}")
print(f"  Label files modified            : {stats['files_modified']}")
print("=" * 55)
print("\nNow re-run validator to confirm:")
print("  python validate_dataset.py")