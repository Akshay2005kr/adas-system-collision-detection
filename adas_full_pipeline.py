"""
adas_full_pipeline.py  (v2 — bug-fixed integration)
=================================
All five ADAS modules combined, with four improvements:

  1. Pothole / hump boxes now ALWAYS show an estimated distance + real
     size (cm) — the old code only used a strict ground-plane formula
     that silently returned nothing whenever a box sat above the frame's
     vertical center or landed outside a hard 0.5-80m range. There's now
     a fallback heuristic (box-height based) so something is always
     shown, marked with "~" when it's the rougher estimate.
  2. Brake bar is smooth and physics-informed: TTC + distance curve,
     a real stopping-distance check using actual speed, and the ML
     collision model's CONTINUOUS probability (not just a hard label)
     blended in — smoothed per-vehicle-track so single noisy frames
     can't yank the target (which is what caused the old "jumps to 40
     then snaps to 0" behaviour).
  3. Direction arrow (STRAIGHT / LEFT / RIGHT) drawn inside the drivable
     area, Tesla-FSD style — now triggers on closing speed (low TTC) as
     well as raw distance, so it reacts before a fast car gets close.
  4. After the run, a self-contained report.html is generated with:
       • Interactive Chart.js graphs (distance, speed, TTC, risk, brake timeline)
       • Photo gallery of DANGER / WARNING / POTHOLE / HUMP / BRAKE snapshots
       • Pothole/hump/brake-event counts in the summary cards
       • Chronological event timeline with metadata
"""

import cv2
import os
import time
import numpy as np
from ultralytics import YOLO

# --- existing modules ---
from feature_extractor import extract_features
from collision_model import CollisionModel
from live_graph import LiveDashboard
from data_logger import DataLogger
from alert_system import AlertSystem

# --- seg helpers ---
from seg_utils import build_mask, blend_mask
from tesla_view import build_known_mask, render_painted_background

# --- 3D renderer ---
from car_renderer_3d import Scene3D
from render_3d import render_3d_view

# --- smooth brake, direction arrow, web report, trajectory ---
from brake_predictor import BrakePredictor
from direction_predictor import DirectionPredictor
from web_report_generator import WebReportGenerator
from trajectory_planner import TrajectoryPlanner

# ============================================================
# 1.  MODEL PATHS  (>>> fix these 6 paths <<<)
# ============================================================
MAIN_MODEL_PATH     = "modals/car_best.pt"
POTHOLE_MODEL_PATH  = "modals/bestphotole.pt"
HUMP_MODEL_PATH     = "modals/besthumb.pt"
LANE_MODEL_PATH     = "modals/linerodebest.pt"
DRIVABLE_MODEL_PATH = "bdd100k_40k_yolov8n_seg_results/weights/best.pt"
ANIMAL_MODEL_PATH   = "modals/animal_best.pt"

model          = YOLO(MAIN_MODEL_PATH)
pothole_model  = YOLO(POTHOLE_MODEL_PATH)
hump_model     = YOLO(HUMP_MODEL_PATH)
lane_model     = YOLO(LANE_MODEL_PATH)
drivable_model = YOLO(DRIVABLE_MODEL_PATH)

# Load animal model with graceful fallback
try:
    animal_model = YOLO(ANIMAL_MODEL_PATH)
    print(f"[animal model]     {animal_model.names}")
except Exception as e:
    animal_model = None
    print(f"[WARNING] Animal model not loaded: {e}. Continuing without animal detection.")

collision_model = CollisionModel("modals/best_model.pkl")
dashboard       = LiveDashboard()
logger          = DataLogger("logs/session_log.csv")
alert           = AlertSystem()

# --- smooth-brake / direction / report / trajectory instances ---
brake_pred   = BrakePredictor(alpha_rise=0.18, alpha_fall=0.25)
dir_pred     = DirectionPredictor(alpha=0.22)
web_reporter = WebReportGenerator("reports")
traj_planner = TrajectoryPlanner(alpha=0.25)

print("[main model]     ", model.names)
print("[lane model]     ", lane_model.names)
print("[drivable model] ", drivable_model.names)

# ============================================================
# 2.  CONFIDENCE / IOU THRESHOLDS
# ============================================================
POTHOLE_CONF, POTHOLE_IOU = 0.40, 0.50
HUMP_CONF,    HUMP_IOU    = 0.40, 0.50
LANE_CONF                 = 0.35
DRIVABLE_CONF             = 0.35
DETECT_EVERY_N = 2
SEG_EVERY_N    = 3

# ============================================================
# 3.  CALIBRATION  (>>> tune these for your camera <<<)
# ============================================================
FOCAL_LENGTH    = 800     # pixels (calibrated for your lens)
CAMERA_HEIGHT_M = 1.30    # metres — camera above ground (hood mount ≈ 1.0-1.4)

REAL_WIDTHS = {
    "person": 0.5, "rider": 0.6, "car": 1.8,
    "bus": 2.5, "truck": 2.5, "bike": 0.7, "motor": 0.7,
}

# Animal width mapping (configurable, with safe fallbacks)
ANIMAL_WIDTHS = {
    "dog": 0.4, "cat": 0.25, "cow": 0.6, "horse": 0.7,
    "sheep": 0.5, "pig": 0.4, "goat": 0.35, "deer": 0.5,
    "elephant": 2.5, "bear": 0.6, "zebra": 0.5, "giraffe": 0.8,
}
DEFAULT_ANIMAL_WIDTH = 0.4  # fallback for unknown animal classes

CLASS_COLORS = {
    "car":           (255, 204, 102),
    "truck":         (102, 178, 255),
    "bus":           (204, 153, 255),
    "person":        (153, 255, 153),
    "rider":         (255, 153, 204),
    "bike":          (153, 255, 255),
    "motor":         (153, 204, 255),
    "pothole":       (255, 102, 255),
    "hump":          (0, 255, 255),
    "traffic sign":  (102, 255, 255),
    "traffic light": (102, 255, 255),
}

DRIVABLE_COLOR = (0, 200, 0)
LANE_COLOR     = (0, 0, 255)

TESLA_MODE          = False
TESLA_INPAINT_RADIUS = 8
TESLA_DOWNSCALE      = 0.5

# ============================================================
# 4.  POTHOLE / HUMP  distance + size helpers
# ============================================================
def _ground_distance(y_bottom: float, frame_h: int, box_height_px: float | None = None):
    """
    Estimate ground distance (metres) for a pothole/hump box.

    Returns (distance_m, is_estimate):
      - Primary: pinhole ground-plane formula from the box's bottom pixel
        (distance = CAMERA_HEIGHT * FOCAL_LENGTH / (y_bottom - horizon_y)).
        Precise when the camera calibration matches your rig, but
        undefined at/above the horizon and unreliable outside a sane
        road range.
      - Fallback (is_estimate=True): if the ground-plane formula isn't
        usable, approximate distance from the box's pixel HEIGHT instead,
        assuming a typical pothole/hump vertical footprint as seen by a
        low, forward-facing camera. Rougher, but means the HUD always
        has a number to show instead of going blank — which was the
        actual bug (potholes/humps showing no info at all).
      - (None, False) only if there's truly nothing to go on (zero-height box).
    """
    horizon_y = frame_h / 2.0
    dy = y_bottom - horizon_y
    if dy >= 5:
        dist = (CAMERA_HEIGHT_M * FOCAL_LENGTH) / dy
        if 0.5 <= dist <= 80:
            return dist, False

    if box_height_px and box_height_px > 1:
        ASSUMED_FEATURE_HEIGHT_M = 0.12  # rough pothole/hump vertical extent on camera
        dist = (ASSUMED_FEATURE_HEIGHT_M * FOCAL_LENGTH) / box_height_px
        dist = max(1.0, min(dist, 60.0))
        return dist, True

    return None, False


def _ground_size_cm(x1, y1, x2, y2, dist_m: float) -> tuple[int, int]:
    """
    Given a bounding box and its estimated ground distance, return
    real-world width × height in centimetres.
    Width is the more reliable dimension (no perspective foreshortening).
    Height is the ground-plane 'length towards the camera' (less precise).
    """
    px_w = x2 - x1
    px_h = y2 - y1
    w_m = (px_w * dist_m) / FOCAL_LENGTH
    h_m = (px_h * dist_m) / FOCAL_LENGTH
    return max(1, round(w_m * 100)), max(1, round(h_m * 100))


# ============================================================
# 5.  SMOOTH DETECTOR  (for pothole / hump)
# ============================================================
def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(1e-6, area_a + area_b - inter)


class SmoothDetector:
    def __init__(self, max_age=6, iou_thresh=0.35):
        self.tracks = []
        self.max_age = max_age
        self.iou_thresh = iou_thresh

    def update(self, new_boxes):
        matched = set()
        for nb in new_boxes:
            best_iou, best_idx = 0.0, -1
            for i, t in enumerate(self.tracks):
                if i in matched:
                    continue
                iou = _iou(nb, t['box'])
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_idx != -1 and best_iou > self.iou_thresh:
                self.tracks[best_idx]['box'] = nb
                self.tracks[best_idx]['age'] = 0
                matched.add(best_idx)
            else:
                self.tracks.append({'box': nb, 'age': 0})
        survivors = []
        for i, t in enumerate(self.tracks):
            if i not in matched:
                t['age'] += 1
            if t['age'] <= self.max_age:
                survivors.append(t)
        self.tracks = survivors
        return [t['box'] for t in self.tracks]


pothole_tracker = SmoothDetector(max_age=6, iou_thresh=0.35)
hump_tracker    = SmoothDetector(max_age=6, iou_thresh=0.35)

last_drivable_mask = None
last_lane_mask     = None
track_history       = {}
risk_score_history  = {}   # NEW: per-track EMA of the ML model's continuous risk score


def _label_from_score(score: float) -> str:
    """Derive the discrete SAFE/WARNING/DANGER label from the SMOOTHED
    continuous risk score, instead of re-predicting the discrete label
    separately from the raw single-frame features. Using one smoothed
    source of truth for both the label and the brake floor is what kills
    the frame-to-frame flicker that was causing the brake bar to jump."""
    if score >= 0.60:
        return "DANGER"
    elif score >= 0.25:
        return "WARNING"
    return "SAFE"


# ============================================================
# 6.  DESKTOP MODE EXECUTION FUNCTION
# ============================================================
def run_desktop_mode():
    """Run the desktop ADAS pipeline with video input and OpenCV display."""
    
    # ============================================================
    # 6.1  OPEN VIDEO SOURCE
    # ============================================================
    VIDEO_SOURCE = "pothole.mp4"       # >>> change to your file / 0 for webcam <<<
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[error] Cannot open video source: {VIDEO_SOURCE}")
        return
    
    frame_count = 0
    
    SHOW_3D_WINDOW    = True
    SAVE_3D_VIDEO     = True
    THREED_OUTPUT_PATH = "reports/3d_view.mp4"
    scene_3d  = None
    writer_3d = None
    
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 1:
        source_fps = 30.0
    frame_interval = 1.0 / source_fps
    
    session_start = time.time()
    
    # ============================================================
    # 7.  MAIN LOOP
    # ============================================================
    while True:
        loop_start   = time.time()
        ret, frame   = cap.read()
        if not ret:
            break
    
        frame_count   += 1
        current_time   = time.time()
        h, w           = frame.shape[:2]
        session_t      = current_time - session_start   # seconds since start
    
        # ---- run all five models ----
        results = model.track(frame, persist=True, conf=0.35, iou=0.45,
                               tracker="bytetrack.yaml", verbose=False)
    
        run_det = (frame_count % DETECT_EVERY_N == 0)
        run_seg = (frame_count % SEG_EVERY_N    == 0)
    
        raw_potholes, raw_humps = [], []
        if run_det:
            for r in pothole_model(frame, conf=POTHOLE_CONF, iou=POTHOLE_IOU, verbose=False):
                for box in r.boxes:
                    raw_potholes.append(tuple(map(float, box.xyxy[0])))
            for r in hump_model(frame, conf=HUMP_CONF, iou=HUMP_IOU, verbose=False):
                for box in r.boxes:
                    raw_humps.append(tuple(map(float, box.xyxy[0])))
    
        potholes = pothole_tracker.update(raw_potholes)
        humps    = hump_tracker.update(raw_humps)
    
        if run_seg:
            drivable_results   = drivable_model(frame, conf=DRIVABLE_CONF, verbose=False)
            last_drivable_mask = build_mask(drivable_results, frame.shape)
            lane_results       = lane_model(frame, conf=LANE_CONF, verbose=False)
            last_lane_mask     = build_mask(lane_results, frame.shape)
    
        drivable_mask = last_drivable_mask
        lane_mask     = last_lane_mask
    
        vehicles = []
        animals  = []
        signs    = []
    
        # ---- PASS 1: vehicles / persons ----
        for r in results:
            for box in r.boxes:
                cls  = int(box.cls[0])
                name = model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
    
                if name in ("pothole", "hump", "lane", "drivable area"):
                    continue
    
                if name in ("traffic sign", "traffic light"):
                    signs.append((name, x1, y1, x2, y2))
                    continue
    
                if name not in REAL_WIDTHS:
                    continue
    
                pixel_width  = max(x2 - x1, 1)
                raw_distance = (REAL_WIDTHS[name] * FOCAL_LENGTH) / pixel_width
    
                speed_kmh, ttc, risk = 0.0, float("inf"), "SAFE"
                final_distance = raw_distance
                risk_score = 0.0
    
                if box.id is not None:
                    track_id = int(box.id[0])
                    if track_id in track_history:
                        prev   = track_history[track_id]
                        sm_d   = prev["dist"] * 0.8 + raw_distance * 0.2
                        dt     = current_time - prev["time"]
                        sm_spd = 0.0
                        if dt > 0:
                            raw_spd_ms = (prev["dist"] - sm_d) / dt
                            sm_spd     = prev.get("speed", 0) * 0.8 + raw_spd_ms * 0.2
                            if sm_spd > 0.3:
                                speed_kmh = sm_spd * 3.6
                                ttc       = sm_d / sm_spd
                        final_distance = sm_d
                        track_history[track_id] = {"dist": sm_d, "time": current_time, "speed": sm_spd}
                    else:
                        track_history[track_id] = {"dist": raw_distance, "time": current_time, "speed": 0}
    
                    features = extract_features(
                        track_id, (x1, y1, x2, y2), frame.shape, final_distance, current_time)
    
                    # ---- NEW: continuous ML risk score, smoothed per-track ----
                    raw_score  = collision_model.predict_proba(features[0])
                    prev_score = risk_score_history.get(track_id, raw_score)
                    risk_score = 0.35 * raw_score + 0.65 * prev_score
                    risk_score_history[track_id] = risk_score
                    risk = _label_from_score(risk_score)
    
                if final_distance > 12:
                    risk = "SAFE"
                    risk_score = min(risk_score, 0.05)
                elif final_distance > 5 and risk == "DANGER":
                    risk = "WARNING"
                    risk_score = min(risk_score, 0.55)
    
                risk_color  = {"DANGER": (0, 0, 255), "WARNING": (0, 165, 255), "SAFE": (0, 255, 0)}[risk]
                class_color = CLASS_COLORS.get(name, (255, 255, 255))
    
                vehicles.append({
                    "name": name, "box": (x1, y1, x2, y2),
                    "distance": final_distance, "speed": speed_kmh, "ttc": ttc,
                    "class_color": class_color, "risk_color": risk_color, "risk": risk,
                    "risk_score": risk_score,
                })

    # ---- ANIMAL DETECTION (if model loaded) ----
    if animal_model is not None:
        try:
            animal_results = animal_model(frame, conf=0.35, iou=0.45, verbose=False)
            for r in animal_results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    name = animal_model.names[cls] if animal_model.names else f"animal_{cls}"
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Get animal width from mapping or use default
                    animal_width = ANIMAL_WIDTHS.get(name.lower(), DEFAULT_ANIMAL_WIDTH)
                    
                    pixel_width = max(x2 - x1, 1)
                    raw_distance = (animal_width * FOCAL_LENGTH) / pixel_width
                    
                    speed_kmh, ttc, risk = 0.0, float("inf"), "SAFE"
                    final_distance = raw_distance
                    risk_score = 0.0
                    
                    if box.id is not None:
                        track_id = int(box.id[0])
                        if track_id in track_history:
                            prev = track_history[track_id]
                            sm_d = prev["dist"] * 0.8 + raw_distance * 0.2
                            dt = current_time - prev["time"]
                            sm_spd = 0.0
                            if dt > 0:
                                raw_spd_ms = (prev["dist"] - sm_d) / dt
                                sm_spd = prev.get("speed", 0) * 0.8 + raw_spd_ms * 0.2
                                if sm_spd > 0.3:
                                    speed_kmh = sm_spd * 3.6
                                    ttc = sm_d / sm_spd
                            final_distance = sm_d
                            track_history[track_id] = {"dist": sm_d, "time": current_time, "speed": sm_spd}
                        else:
                            track_history[track_id] = {"dist": raw_distance, "time": current_time, "speed": 0}
                        
                        # Use simplified risk for animals (no ML model needed)
                        if final_distance < 8:
                            risk = "DANGER"
                            risk_score = 0.7
                        elif final_distance < 15:
                            risk = "WARNING"
                            risk_score = 0.4
                        else:
                            risk = "SAFE"
                            risk_score = 0.1
                    
                    risk_color = {"DANGER": (0, 0, 255), "WARNING": (0, 165, 255), "SAFE": (0, 255, 0)}[risk]
                    class_color = CLASS_COLORS.get(name.lower(), (153, 255, 153))
                    
                    animals.append({
                        "name": name, "box": (x1, y1, x2, y2),
                        "distance": final_distance, "speed": speed_kmh, "ttc": ttc,
                        "class_color": class_color, "risk_color": risk_color, "risk": risk,
                        "risk_score": risk_score, "type": "animal"
                    })
        except Exception as e:
            print(f"[WARNING] Animal detection error: {e}")

    logger.log(frame_count, current_time, vehicles)

    if frame_count % 30 == 0 or frame_count == 1:
        print(f"[frame {frame_count}] vehicles={len(vehicles)} animals={len(animals)} "
              f"potholes={len(potholes)} humps={len(humps)}")

    # ---- 3D view ----
    if SHOW_3D_WINDOW or SAVE_3D_VIDEO:
        if scene_3d is None:
            scene_3d = Scene3D(frame.shape, FOCAL_LENGTH)
            if SAVE_3D_VIDEO:
                out_dir_3d = os.path.dirname(THREED_OUTPUT_PATH)
                if out_dir_3d:
                    os.makedirs(out_dir_3d, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer_3d = cv2.VideoWriter(
                    THREED_OUTPUT_PATH, fourcc, source_fps,
                    (frame.shape[1], frame.shape[0]))

        canvas_3d = render_3d_view(
            vehicles, frame.shape, FOCAL_LENGTH, scene_3d,
            drivable_mask=drivable_mask, lane_mask=lane_mask,
            potholes=potholes, humps=humps, signs=signs, animals=animals)

        if SHOW_3D_WINDOW:
            cv2.imshow("3D View - Tesla Style", canvas_3d)
        if writer_3d is not None:
            writer_3d.write(canvas_3d)

        closest_vehicle = min(vehicles, key=lambda v: v["distance"], default=None)
        closest_animal = min(animals, key=lambda a: a["distance"], default=None) if animals else None
    
        # Determine overall closest obstacle (vehicle or animal)
        all_obstacles = []
        if closest_vehicle:
            all_obstacles.append(closest_vehicle)
        if closest_animal:
            all_obstacles.append(closest_animal)
    
        closest_obstacle = min(all_obstacles, key=lambda x: x["distance"], default=None)
        risk_now = closest_obstacle["risk"] if closest_obstacle else "SAFE"
        alert.check(risk_now)

        if closest_obstacle:
            dashboard.update(
                closest_obstacle["distance"], closest_obstacle["speed"],
                closest_obstacle["ttc"], closest_obstacle["risk"])

        # ============================================================
        # ①  SMOOTH BRAKE PREDICTION (now speed- and ML-score-aware)
        # ============================================================
        if closest_obstacle:
            brake_pct, brake_col = brake_pred.update(
                closest_obstacle["distance"],
                closest_obstacle["ttc"],
                closest_obstacle["risk"],
                speed_kmh=closest_obstacle["speed"],
                risk_score=closest_obstacle.get("risk_score"))
        else:
            brake_pct, brake_col = brake_pred.update(
                999.0, float("inf"), "SAFE", speed_kmh=0.0, risk_score=0.0)

        # Log brake history for the web report chart
        web_reporter.log_brake(session_t, brake_pct)

        # ============================================================
        # ②  TRAJECTORY PLANNING (replaces simple direction arrow)
        # ============================================================
        direction, trajectory_pts, is_safe = traj_planner.compute_safe_trajectory(
            vehicles, animals, potholes, humps, frame.shape, drivable_mask)
    
        # Draw the real trajectory on frame
        frame = traj_planner.draw_trajectory(frame, trajectory_pts, direction, drivable_mask, alpha=0.45)
    
        # Also draw legacy direction arrow (smaller, secondary indicator)
        frame = dir_pred.draw_arrow(frame, direction, drivable_mask, risk=risk_now)

        # ============================================================
        # ③  EVENT CAPTURE (for web gallery)
        # ============================================================
        if risk_now == "DANGER" and closest_obstacle:
            ttc_str = (f"{closest_obstacle['ttc']:.1f}s"
                       if closest_obstacle["ttc"] != float("inf") else "--")
            event_type = "ANIMAL" if closest_obstacle.get("type") == "animal" else "DANGER"
            web_reporter.capture_event(frame, event_type, {
                "dist":    f"{closest_obstacle['distance']:.1f} m",
                "ttc":     ttc_str,
                "vehicle": closest_obstacle["name"],
                "speed":   f"{closest_obstacle['speed']:.1f} km/h",
            })
        elif risk_now == "WARNING" and closest_obstacle:
            event_type = "ANIMAL" if closest_obstacle.get("type") == "animal" else "WARNING"
            web_reporter.capture_event(frame, event_type, {
                "dist":    f"{closest_obstacle['distance']:.1f} m",
                "vehicle": closest_obstacle["name"],
            }, cooldown=3.0)

        if brake_pct > 65 and closest_obstacle:
            web_reporter.capture_event(frame, "BRAKE", {
                "brake":  f"{brake_pct:.0f}%",
                "dist":   f"{closest_obstacle['distance']:.1f} m",
                "speed":  f"{closest_obstacle['speed']:.1f} km/h",
            })
    
        # Capture animal events separately
        for animal in animals:
            if animal["risk"] == "DANGER":
                web_reporter.capture_event(frame, "ANIMAL", {
                    "dist": f"{animal['distance']:.1f} m",
                    "type": animal["name"],
                }, cooldown=2.0)

        if potholes:
            (px1, py1, px2, py2) = potholes[0]
            pd, pd_est = _ground_distance(py2, h, box_height_px=(py2 - py1))
            if pd is not None:
                w_cm, h_cm = _ground_size_cm(px1, py1, px2, py2, pd)
                dist_str = f"{'~' if pd_est else ''}{pd:.1f} m"
                size_str = f"{w_cm}x{h_cm} cm"
            else:
                dist_str, size_str = "--", "--"
            web_reporter.capture_event(frame, "POTHOLE", {
                "dist": dist_str,
                "size": size_str,
            }, cooldown=5.0)

        if humps:
            (hx1, hy1, hx2, hy2) = humps[0]
            hd, hd_est = _ground_distance(hy2, h, box_height_px=(hy2 - hy1))
            if hd is not None:
                w_cm, h_cm = _ground_size_cm(hx1, hy1, hx2, hy2, hd)
                dist_str = f"{'~' if hd_est else ''}{hd:.1f} m"
                size_str = f"{w_cm}x{h_cm} cm"
            else:
                dist_str, size_str = "--", "--"
            web_reporter.capture_event(frame, "HUMP", {
                "dist": dist_str,
                "size": size_str,
            }, cooldown=5.0)

        # ============================================================
        # TESLA MODE
        # ============================================================
        if TESLA_MODE:
            known_mask = build_known_mask(
                frame.shape,
                vehicle_boxes=[v["box"] for v in vehicles],
                pothole_boxes=potholes,
                hump_boxes=humps,
                lane_mask=lane_mask,
                drivable_mask=drivable_mask)
            frame = render_painted_background(
                frame, known_mask,
                inpaint_radius=TESLA_INPAINT_RADIUS, downscale=TESLA_DOWNSCALE)

        # ============================================================
        # PASS 2: DRAW — drivable + lanes (background layer)
        # ============================================================
        if drivable_mask is not None:
            frame = blend_mask(frame, drivable_mask, color=DRIVABLE_COLOR, alpha=0.28)
        if lane_mask is not None:
            frame = blend_mask(frame, lane_mask, color=LANE_COLOR, alpha=0.55)

        # ---- semi-transparent card backgrounds ----
        overlay = frame.copy()
        card_h = 85
        for v in vehicles:
            x1, y1, x2, y2 = v["box"]
            card_y1 = max(0, y1 - card_h)
            cv2.rectangle(overlay, (x1, card_y1), (x1 + 170, card_y1 + card_h), (15, 15, 15), -1)
        # top-left HUD panel
        cv2.rectangle(overlay, (10, 10), (245, 165), (15, 15, 15), -1)

        # ---- SMOOTH BRAKE BAR (drawn on overlay) ----
        bar_w, bar_h = 420, 22
        bar_x = w // 2 - bar_w // 2
        bar_y = h - 55
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 30, 30), -1)
        fill_w = int((brake_pct / 100.0) * bar_w)
        if fill_w > 0:
            cv2.rectangle(overlay, (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h), brake_col, -1)

        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

        # ============================================================
        # PASS 3: OPAQUE TEXT / BORDERS
        # ============================================================

        # ---- Potholes with distance + size (always shows a value now) ----
        for (x1, y1, x2, y2) in potholes:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            p_color = CLASS_COLORS["pothole"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), p_color, 2, cv2.LINE_AA)

            dist_m, is_est = _ground_distance(y2, h, box_height_px=(y2 - y1))
            if dist_m is not None:
                w_cm, h_cm = _ground_size_cm(x1, y1, x2, y2, dist_m)
                tag = "~" if is_est else ""
                line1 = f"POTHOLE  {tag}{dist_m:.1f} m"
                line2 = f"Size: {tag}{w_cm}x{h_cm} cm"
                for line, yoff in ((line1, -24), (line2, -8)):
                    (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(frame,
                                  (x1 - 2, y1 + yoff - th - 2),
                                  (x1 + tw + 4, y1 + yoff + 4),
                                  (15, 15, 15), -1)
                    cv2.putText(frame, line, (x1, y1 + yoff),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, p_color, 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "POTHOLE", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, p_color, 2, cv2.LINE_AA)

        # ---- Humps with distance + size (always shows a value now) ----
        for (x1, y1, x2, y2) in humps:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            hcolor = CLASS_COLORS["hump"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), hcolor, 2, cv2.LINE_AA)

            dist_m, is_est = _ground_distance(y2, h, box_height_px=(y2 - y1))
            if dist_m is not None:
                w_cm, h_cm = _ground_size_cm(x1, y1, x2, y2, dist_m)
                tag = "~" if is_est else ""
                line1 = f"HUMP  {tag}{dist_m:.1f} m"
                line2 = f"Size: {tag}{w_cm}x{h_cm} cm"
                for line, yoff in ((line1, -24), (line2, -8)):
                    (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(frame,
                                  (x1 - 2, y1 + yoff - th - 2),
                                  (x1 + tw + 4, y1 + yoff + 4),
                                  (15, 15, 15), -1)
                    cv2.putText(frame, line, (x1, y1 + yoff),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, hcolor, 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "HUMP", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, hcolor, 2, cv2.LINE_AA)

        # ---- Traffic signs ----
        for (name, x1, y1, x2, y2) in signs:
            s_color = CLASS_COLORS.get(name, (102, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), s_color, 2, cv2.LINE_AA)
            cv2.putText(frame, name.upper(), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, s_color, 2, cv2.LINE_AA)

        # ---- Vehicle cards ----
        for v in vehicles:
            x1, y1, x2, y2 = v["box"]
            card_y1 = max(0, y1 - card_h)
            speed_str = f"{v['speed']:.1f} km/h" if v["speed"] > 0 else "-- km/h"
            ttc_str   = f"{v['ttc']:.1f} s"       if v["ttc"] != float("inf") else "-- s"

            cv2.rectangle(frame, (x1, y1), (x2, y2), v["class_color"], 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, card_y1), (x1 + 170, card_y1 + card_h),
                          v["class_color"], 1, cv2.LINE_AA)
            cv2.putText(frame, v["name"].upper(),
                        (x1 + 8, card_y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, v["class_color"], 2, cv2.LINE_AA)
            cv2.putText(frame, f"Risk : {v['risk']}",
                        (x1 + 8, card_y1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, v["risk_color"], 1, cv2.LINE_AA)
            cv2.putText(frame, f"Dist : {v['distance']:.1f} m",
                        (x1 + 8, card_y1 + 49), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Speed: {speed_str}",
                        (x1 + 8, card_y1 + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"TTC  : {ttc_str}",
                        (x1 + 8, card_y1 + 79), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # ---- Animal cards ----
        for a in animals:
            x1, y1, x2, y2 = a["box"]
            card_y1 = max(0, y1 - card_h)
            speed_str = f"{a['speed']:.1f} km/h" if a["speed"] > 0 else "-- km/h"
            ttc_str   = f"{a['ttc']:.1f} s"       if a["ttc"] != float("inf") else "-- s"

            cv2.rectangle(frame, (x1, y1), (x2, y2), a["class_color"], 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, card_y1), (x1 + 170, card_y1 + card_h),
                          a["class_color"], 1, cv2.LINE_AA)
            cv2.putText(frame, a["name"].upper() + " (ANIMAL)",
                        (x1 + 8, card_y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, a["class_color"], 2, cv2.LINE_AA)
            cv2.putText(frame, f"Risk : {a['risk']}",
                        (x1 + 8, card_y1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, a["risk_color"], 1, cv2.LINE_AA)
            cv2.putText(frame, f"Dist : {a['distance']:.1f} m",
                        (x1 + 8, card_y1 + 49), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Speed: {speed_str}",
                        (x1 + 8, card_y1 + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"TTC  : {ttc_str}",
                        (x1 + 8, card_y1 + 79), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # ---- Top-left HUD status panel ----
        cv2.rectangle(frame, (10, 10), (245, 165), (255, 255, 255), 1, cv2.LINE_AA)
        total_objects = len(vehicles) + len(animals) + len(potholes) + len(humps) + len(signs)
        cv2.putText(frame, "ADAS STATUS", (25, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Objects : {total_objects}", (25, 57),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
        lane_status = "OK" if (lane_mask is not None and lane_mask.any()) else "--"
        drv_status  = "OK" if (drivable_mask is not None and drivable_mask.any()) else "--"
        cv2.putText(frame, f"Lane    : {lane_status}", (25, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, LANE_COLOR, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Drivable: {drv_status}", (25, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, DRIVABLE_COLOR, 1, cv2.LINE_AA)

        if closest_obstacle:
            obs_type = "ANIMAL" if closest_obstacle.get("type") == "animal" else closest_obstacle["name"].upper()
            cv2.putText(frame, f"Closest : {obs_type}", (25, 116),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, closest_obstacle["class_color"], 1, cv2.LINE_AA)
            cv2.putText(frame, f"Distance: {closest_obstacle['distance']:.1f} m", (25, 134),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Risk    : {closest_obstacle['risk']}", (25, 152),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, closest_obstacle["risk_color"], 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Closest : None", (25, 116),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, "Risk    : SAFE", (25, 152),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2, cv2.LINE_AA)

        # ---- Brake bar labels (smooth value + speed) ----
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1, cv2.LINE_AA)
        brake_label = f"BRAKE: {int(brake_pct)}%"
        cv2.putText(frame, brake_label, (bar_x + bar_w + 12, bar_y + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, brake_col, 2, cv2.LINE_AA)

        # ---- Speed display next to brake bar ----
        if closest_vehicle and closest_vehicle["speed"] > 0:
            spd_label = f"{closest_vehicle['speed']:.0f} km/h"
            cv2.putText(frame, spd_label, (bar_x - 90, bar_y + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 220, 80), 2, cv2.LINE_AA)

        # ---- Direction badge (top-right corner) ----
        dir_colors = {"STRAIGHT": (0, 245, 80), "LEFT": (0, 210, 255), "RIGHT": (30, 160, 255)}
        dir_icons  = {"STRAIGHT": "▲ STRAIGHT", "LEFT": "◀ LEFT", "RIGHT": "▶ RIGHT"}
        dir_col    = dir_colors[direction]
        dir_text   = dir_icons[direction]
        (dtw, _), _ = cv2.getTextSize(dir_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        dx = w - dtw - 20
        cv2.rectangle(frame, (dx - 8, 12), (w - 10, 38), (15, 15, 15), -1)
        cv2.putText(frame, dir_text, (dx, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, dir_col, 2, cv2.LINE_AA)

        # ---- DANGER flash overlay ----
        frame = alert.draw_flash(frame, risk_now)

        window_title = ("ADAS - Full Pipeline [TESLA MODE]"
                        if TESLA_MODE else "ADAS - Full Pipeline [NORMAL COLOR]")
        cv2.imshow(window_title, frame)

        elapsed_ms    = int((time.time() - loop_start) * 1000)
        remaining_ms  = max(1, int(frame_interval * 1000) - elapsed_ms)
        if cv2.waitKey(remaining_ms) & 0xFF == ord("q"):
            break

# ============================================================
# 8.  CLEANUP + WEB REPORT
# ============================================================
cap.release()
if writer_3d is not None:
    writer_3d.release()
    print(f"[main] 3D view saved → {THREED_OUTPUT_PATH}")
cv2.destroyAllWindows()
dashboard.close()
logger.close()

# Generate the interactive HTML web report
session_label = time.strftime("Run %Y-%m-%d %H:%M")
web_reporter.generate(logger.filepath, session_name=session_label)

print("[main] Done.")


# ============================================================
# ENTRY POINT - Only run desktop mode when executed directly
# ============================================================
if __name__ == "__main__":
    run_desktop_mode()
