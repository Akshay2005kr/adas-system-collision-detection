"""
YOLOv8 Speed Breaker Detector — Training Script
================================================
Trains a YOLOv8 model on your speedbreaker_only_dataset.

Run:
  python train_speedbreaker.py

Requirements:
  pip install ultralytics
"""

import os
import sys
from pathlib import Path

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

# Path to your dataset (the folder you just created)
DATA_YAML     = "./speedbreaker_only_dataset/data.yaml"

# Model size — choose one:
#   yolov8n.pt  → Nano   (fastest, smallest, good for Raspberry Pi / mobile)
#   yolov8s.pt  → Small  (good balance ✓ recommended for your dataset size)
#   yolov8m.pt  → Medium (better accuracy, needs more GPU RAM)
#   yolov8l.pt  → Large  (best accuracy, needs strong GPU)
MODEL         = "yolov8s.pt"

# Output folder for trained weights + results
PROJECT       = "./runs/speedbreaker"
RUN_NAME      = "v1"

# Training settings
EPOCHS        = 100       # 100 is good for ~1600 images
IMGSZ         = 640       # standard YOLOv8 input size
BATCH         = 16        # lower to 8 if you get CUDA out-of-memory
WORKERS       = 4         # data loader threads (set 0 on Windows if errors)
PATIENCE      = 20        # stop early if no improvement for 20 epochs
LR0           = 0.01      # initial learning rate
LRF           = 0.01      # final learning rate factor
MOMENTUM      = 0.937
WEIGHT_DECAY  = 0.0005
WARMUP_EPOCHS = 3

# Augmentation (helps a lot with small datasets)
AUGMENT       = True
DEGREES       = 5.0       # random rotation ±5° (road images shouldn't flip too much)
TRANSLATE     = 0.1       # random translation
SCALE         = 0.5       # random scale
FLIPUD        = 0.1       # vertical flip (occasional, simulates different camera angles)
FLIPLR        = 0.5       # horizontal flip (common augmentation)
MOSAIC        = 1.0       # mosaic augmentation (very helpful for small datasets)
MIXUP         = 0.1       # mix two images together slightly

# Device: 'cuda' for GPU, 'cpu' for CPU (GPU is strongly recommended)
DEVICE        = "0"       # use "cpu" if no NVIDIA GPU

# ─────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────

def check_dataset(yaml_path: str):
    """Quick sanity check before training starts."""
    import yaml
    p = Path(yaml_path)
    if not p.exists():
        print(f"\n  ERROR: data.yaml not found at {p}")
        print("  Make sure you ran fix_speedbreaker_only.py first.")
        sys.exit(1)

    with open(p) as f:
        data = yaml.safe_load(f)

    dataset_root = Path(data.get("path", p.parent))
    train_dir = dataset_root / "train" / "images"
    valid_dir = dataset_root / "valid" / "images"

    train_count = len(list(train_dir.glob("*"))) if train_dir.exists() else 0
    valid_count = len(list(valid_dir.glob("*"))) if valid_dir.exists() else 0

    print(f"\n  Dataset check:")
    print(f"    data.yaml   : {p.resolve()}")
    print(f"    Classes     : {data.get('names')}  (nc={data.get('nc')})")
    print(f"    Train images: {train_count}")
    print(f"    Valid images: {valid_count}")

    if train_count == 0:
        print("\n  ERROR: No training images found!")
        sys.exit(1)

    print(f"\n  All good — starting training...\n")


def main():
    print("\n" + "="*60)
    print("  YOLOv8 Speed Breaker Training")
    print("="*60)

    # Dependency check
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n  ERROR: ultralytics not installed.")
        print("  Run: pip install ultralytics")
        sys.exit(1)

    # Dataset sanity check
    check_dataset(DATA_YAML)

    # Load model (downloads weights automatically if not cached)
    print(f"  Loading model : {MODEL}")
    model = YOLO(MODEL)

    # ── Train ─────────────────────────────────────────────────
    print(f"  Epochs        : {EPOCHS}")
    print(f"  Image size    : {IMGSZ}")
    print(f"  Batch size    : {BATCH}")
    print(f"  Device        : {DEVICE}")
    print(f"  Output        : {PROJECT}/{RUN_NAME}")
    print()

    results = model.train(
        data          = DATA_YAML,
        epochs        = EPOCHS,
        imgsz         = IMGSZ,
        batch         = BATCH,
        workers       = WORKERS,
        patience      = PATIENCE,
        lr0           = LR0,
        lrf           = LRF,
        momentum      = MOMENTUM,
        weight_decay  = WEIGHT_DECAY,
        warmup_epochs = WARMUP_EPOCHS,
        degrees       = DEGREES,
        translate     = TRANSLATE,
        scale         = SCALE,
        flipud        = FLIPUD,
        fliplr        = FLIPLR,
        mosaic        = MOSAIC,
        mixup         = MIXUP,
        augment       = AUGMENT,
        project       = PROJECT,
        name          = RUN_NAME,
        device        = DEVICE,
        exist_ok      = True,
        pretrained    = True,
        verbose       = True,
        save          = True,
        save_period   = 10,    # save checkpoint every 10 epochs
        plots         = True,  # save training curves
        val           = True,
    )

    # ── Validate best model ───────────────────────────────────
    best_weights = Path(PROJECT) / RUN_NAME / "weights" / "best.pt"
    print("\n" + "="*60)
    print("  Training complete! Running validation on best model...")
    print("="*60)

    if best_weights.exists():
        best_model = YOLO(str(best_weights))
        metrics = best_model.val(data=DATA_YAML, imgsz=IMGSZ)

        print(f"\n  RESULTS (best.pt):")
        print(f"    mAP@50      : {metrics.box.map50:.4f}")
        print(f"    mAP@50:95   : {metrics.box.map:.4f}")
        print(f"    Precision   : {metrics.box.mp:.4f}")
        print(f"    Recall      : {metrics.box.mr:.4f}")

    print(f"\n  Saved weights  : {best_weights.resolve()}")
    print(f"  Training plots : {(Path(PROJECT) / RUN_NAME).resolve()}")
    print()
    print("  To run inference on a new image:")
    print(f"    yolo predict model={best_weights} source=your_image.jpg")
    print()
    print("  To export to ONNX (for deployment):")
    print(f"    yolo export model={best_weights} format=onnx")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()