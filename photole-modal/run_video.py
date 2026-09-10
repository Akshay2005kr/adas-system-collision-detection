import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO

# ── Settings ──────────────────────────────────────────────
MODEL_PATH           = "runs/detect/train/weights/best.pt"
CONF_THRESH          = 0.25
IOU_THRESH           = 0.45
IMG_SIZE             = 640
REAL_POTHOLE_WIDTH_M = 0.60   # average real pothole width in metres
FOCAL_LENGTH_PX      = 457.0  # calibrate for your camera

WHITE = (255, 255, 255)
BLACK = (0,   0,   0)
BLUE  = (255, 140, 0)         # box colour (orange-blue in BGR)
# ──────────────────────────────────────────────────────────


def estimate_distance(box_width_px):
    if box_width_px < 2:
        return -1.0
    return (REAL_POTHOLE_WIDTH_M * FOCAL_LENGTH_PX) / box_width_px


def shadow_text(img, text, pos, scale=0.55, color=WHITE, thickness=1):
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, BLACK, thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x,   y),   cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness,   cv2.LINE_AA)


def run(source, save=False, show=True):
    model = YOLO(MODEL_PATH)
    print(f"\nModel  : {MODEL_PATH}")
    print(f"Source : {source}")
    print("Press Q to quit\n")

    cap = cv2.VideoCapture(int(source) if source == "0" else source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open: {source}")
        return

    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if save:
        out = Path(source).stem + "_out.mp4" if source != "0" else "webcam_out.mp4"
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
        print(f"Saving → {out}")

    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        results = model(frame, conf=CONF_THRESH, iou=IOU_THRESH,
                        imgsz=IMG_SIZE, verbose=False)[0]
        n_det = 0

        for box in results.boxes:
            conf         = float(box.conf)
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            bw_px        = x2 - x1
            dist_m       = estimate_distance(bw_px)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), BLUE, 2)

            # Label:  pothole 87%  |  3.2 m
            tag = f"pothole {conf:.0%}  |  {dist_m:.1f} m" \
                  if dist_m >= 0 else f"pothole {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), BLUE, -1)
            shadow_text(frame, tag, (x1 + 4, y1 - 4), scale=0.52, color=WHITE)

            # Distance text below box
            if dist_m >= 0:
                shadow_text(frame, f"{dist_m:.1f} m",
                            ((x1+x2)//2 - 22, y2 + 18),
                            scale=0.55, color=WHITE)

            print(f"  [{frame_no:05d}]  conf={conf:.2f}  dist={dist_m:.2f} m")
            n_det += 1

        # Simple counter top-left
        shadow_text(frame, f"Potholes: {n_det}", (10, 28), scale=0.60, color=WHITE)

        if writer:
            writer.write(frame)
        if show:
            cv2.imshow("Pothole Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped.")
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Frames: {frame_no}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",        help="Video path or 0 for webcam")
    ap.add_argument("--save",   action="store_true", help="Save output video")
    ap.add_argument("--noshow", action="store_true", help="No display window")
    a = ap.parse_args()
    run(source=a.source, save=a.save, show=not a.noshow)