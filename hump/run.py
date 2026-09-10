import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO

# ── Settings ──────────────────────────────────────────────
MODEL_PATH = r"runs\detect\train\weights\best.pt"
CONF_THRESH = 0.25
IOU_THRESH = 0.45
IMG_SIZE = 640

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
# ──────────────────────────────────────────────────────────


def shadow_text(img, text, pos, scale=0.55, color=WHITE, thickness=1):
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, BLACK, thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def run(source, save=False, show=True):
    model = YOLO(MODEL_PATH)

    print(f"\nModel  : {MODEL_PATH}")
    print(f"Source : {source}")
    print("Press Q to quit\n")

    cap = cv2.VideoCapture(0 if source == "0" else source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open: {source}")
        return

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

        count = 0

        for box in results.boxes:
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)

            label = f"speed_breaker {conf:.0%}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), GREEN, -1)

            shadow_text(frame, label, (x1 + 3, y1 - 5))

            count += 1

        # Display count
        shadow_text(frame, f"Speed Breakers: {count}", (10, 30), scale=0.6)

        if writer:
            writer.write(frame)

        if show:
            cv2.imshow("Speed Breaker Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\nDone. Frames: {frame_no}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="Video path or 0 for webcam")
    ap.add_argument("--save", action="store_true", help="Save output video")
    ap.add_argument("--noshow", action="store_true", help="No display window")
    args = ap.parse_args()

    run(source=args.source, save=args.save, show=not args.noshow)