"""
Inference script for BDD100k YOLOv8n-seg model
(lane / road-area segmentation, trained via bdd100k_40k_yolov8n_seg run)

Usage:
    python infer_seg.py --source /path/to/image_or_video --weights /path/to/linerodebest.pt
"""

import argparse
import os
import time
import cv2
import numpy as np
from ultralytics import YOLO


def overlay_masks(result, alpha=0.5, colors=None, prev_mask_probs=None, smooth=0.6):
    """
    Draw colored segmentation masks on top of the original image.

    prev_mask_probs: dict {class_id: float32 mask array} from the previous
        frame, used for temporal smoothing (EMA) to reduce flicker.
        Pass None for single images (no smoothing needed).
    smooth: how much weight to give the previous frame's mask (0-1).
        Higher = smoother/more stable but slightly laggier.

    Returns: (rendered_frame, updated_mask_probs_dict)
    """
    img = result.orig_img.copy()
    overlay = img.copy()

    if colors is None:
        colors = {
            0: (0, 255, 0),   # class 0 -> green
            1: (0, 0, 255),   # class 1 -> red
        }

    h, w = img.shape[:2]
    new_mask_probs = {}

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        # collect this frame's raw probability masks per class (merge duplicates by max)
        curr_probs = {}
        for mask, cls in zip(masks, classes):
            cls = int(cls)
            mask = cv2.resize(mask, (w, h))
            if cls in curr_probs:
                curr_probs[cls] = np.maximum(curr_probs[cls], mask)
            else:
                curr_probs[cls] = mask

        # temporal smoothing with previous frame, per class
        all_classes = set(curr_probs.keys()) | set((prev_mask_probs or {}).keys())
        for cls in all_classes:
            curr = curr_probs.get(cls, np.zeros((h, w), dtype=np.float32))
            if prev_mask_probs is not None and cls in prev_mask_probs:
                blended = smooth * prev_mask_probs[cls] + (1 - smooth) * curr
            else:
                blended = curr
            new_mask_probs[cls] = blended

            color = colors.get(cls, (255, 255, 255))
            colored_mask = np.zeros_like(img)
            colored_mask[blended > 0.5] = color
            overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
    else:
        new_mask_probs = prev_mask_probs or {}

    return overlay, new_mask_probs


def run_image(model, source, out_dir, imgsz, conf, iou):
    # save=False here -> we build our own clean overlay below instead of
    # ultralytics' default plot (which draws boxes+labels, not wanted for
    # a mask-only lane-line / drivable-area model)
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        save=False,
        project=out_dir,
        name="predict",
        exist_ok=True,
    )
    r = results[0]
    r.names = {0: "lane_line", 1: "drivable_area"}
    print("Class names:", r.names)

    out, _ = overlay_masks(r)
    out_path = os.path.join(out_dir, "predict", "overlay.jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, out)
    print(f"Saved overlay to: {out_path}")


def run_video(model, source, out_dir, imgsz, conf, iou):
    # save=False -> ultralytics won't draw its own boxes/labels.
    # We build a clean mask-only overlay video ourselves (lane line +
    # drivable area masks, no boxes, no "car"/"other" text).
    results_gen = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        stream=True,
        save=False,
    )

    save_dir = os.path.join(out_dir, "video_predict")
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "overlay.mp4")

    writer = None
    frame_count = 0
    window_name = "Lane Line / Drivable Area Segmentation - LIVE"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    prev_time = time.time()
    prev_mask_probs = None

    for r in results_gen:
        r.names = {0: "lane_line", 1: "drivable_area"}
        frame, prev_mask_probs = overlay_masks(r, prev_mask_probs=prev_mask_probs, smooth=0.6)

        # live FPS counter
        curr_time = time.time()
        fps_live = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time
        cv2.putText(
            frame, f"FPS: {fps_live:.1f}  Frame: {frame_count}",
            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2
        )

        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = 30  # adjust if you know the source video's real fps
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        writer.write(frame)

        # live preview window — press 'q' to stop early
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopped early by user (q pressed).")
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed frame {frame_count}...")

    cv2.destroyAllWindows()

    if writer is not None:
        writer.release()

    print(f"Done. Processed {frame_count} frames.")
    print(f"Video output saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8n-seg inference (lane / road area)")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained .pt weights")
    parser.add_argument("--source", type=str, required=True, help="Image, video path, or 0 for webcam")
    parser.add_argument("--out", type=str, default="./infer_out", help="Output directory")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--video", action="store_true", help="Set this flag if source is a video/webcam")
    args = parser.parse_args()

    model = YOLO(args.weights)

    # Model was trained with mislabeled class names (car/other).
    # Fix them here directly since the original dataset yaml isn't available.
    # model.names is often a read-only property on newer ultralytics versions,
    # so set it on the underlying torch model instead.
    try:
        model.names = {0: "lane_line", 1: "drivable_area"}
    except AttributeError:
        model.model.names = {0: "lane_line", 1: "drivable_area"}

    os.makedirs(args.out, exist_ok=True)

    if args.video or str(args.source) == "0":
        run_video(model, args.source, args.out, args.imgsz, args.conf, args.iou)
    else:
        run_image(model, args.source, args.out, args.imgsz, args.conf, args.iou)


if __name__ == "__main__":
    main()