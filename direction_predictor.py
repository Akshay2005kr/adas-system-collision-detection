"""
direction_predictor.py
------------------
Predicts recommended steering direction when a vehicle is approaching.

Logic:
  - If no active threat  → STRAIGHT
  - Threat left of frame  → steer RIGHT
  - Threat right of frame → steer LEFT
  - Threat dead-center    → pick the side with more drivable area
                            (or default RIGHT, matching most AEB systems)

v2: a threat is now "active" based on distance OR TTC (not distance
alone). A car closing fast from far away used to not trigger a direction
hint at all until it crossed 12m — now a low TTC (< 4s) triggers it too,
so the arrow reacts to closing speed the way Tesla's visualizer does,
not just raw distance.

Soft-vote EMA prevents per-frame flickering between directions.

Arrow is drawn INSIDE the drivable-area mask so it always appears on
the road surface, not floating in the sky. Circle background keeps it
readable over any scene colour.

Usage:
    from direction_predictor import DirectionPredictor
    dp = DirectionPredictor()
    ...
    direction = dp.predict(closest_vehicle, frame.shape, drivable_mask)
    frame = dp.draw_arrow(frame, direction, drivable_mask)
"""

import cv2
import numpy as np


_DIR_COLORS = {
    "STRAIGHT": (0, 245, 80),
    "LEFT":     (0, 210, 255),
    "RIGHT":    (30, 160, 255),
}

_DIR_LABEL = {
    "STRAIGHT": "STRAIGHT",
    "LEFT":     "STEER LEFT",
    "RIGHT":    "STEER RIGHT",
}


class DirectionPredictor:
    def __init__(self, alpha: float = 0.22):
        """alpha: EMA weight for soft-vote smoothing (0.15-0.30 works well)."""
        self._votes: dict[str, float] = {"STRAIGHT": 1.0, "LEFT": 0.0, "RIGHT": 0.0}
        self._alpha = alpha
        self.current = "STRAIGHT"

    # ------------------------------------------------------------------
    def _lateral_zone(self, box, frame_w: int) -> str:
        """Horizontal third of frame where the detected box's centre lies."""
        cx_norm = (box[0] + box[2]) / 2.0 / max(frame_w, 1)
        if cx_norm < 0.37:
            return "left"
        if cx_norm > 0.63:
            return "right"
        return "center"

    def _free_side(self, drivable_mask, frame_w: int) -> str:
        """Which half of the drivable area has more pixels (or 'equal')."""
        if drivable_mask is None or not drivable_mask.any():
            return "equal"
        left  = int(drivable_mask[:, :frame_w // 2].sum())
        right = int(drivable_mask[:, frame_w // 2:].sum())
        total = max(left + right, 1)
        diff  = (right - left) / total
        if diff > 0.10:
            return "right"
        if diff < -0.10:
            return "left"
        return "equal"

    # ------------------------------------------------------------------
    def predict(self, closest_vehicle, frame_shape, drivable_mask=None) -> str:
        """
        Returns 'STRAIGHT', 'LEFT', or 'RIGHT'.
        Uses soft-vote EMA so the direction is stable across frames.
        """
        h, w = frame_shape[:2]

        if closest_vehicle is None:
            return self._vote("STRAIGHT")

        risk = closest_vehicle.get("risk", "SAFE")
        dist = closest_vehicle.get("distance", 999.0)
        ttc  = closest_vehicle.get("ttc", float("inf"))

        # A threat is "active" if risk is elevated, it's already close,
        # OR it's closing fast enough that TTC is short even from farther
        # away (this is the part that was missing before).
        closing_fast = (ttc != float("inf")) and (ttc < 4.0)
        threat_active = (risk != "SAFE") or (dist <= 12.0) or closing_fast

        if not threat_active:
            return self._vote("STRAIGHT")

        lat  = self._lateral_zone(closest_vehicle["box"], w)
        free = self._free_side(drivable_mask, w)

        if lat == "center":
            # Dead ahead — pick freer lane
            raw = "RIGHT" if free in ("right", "equal") else "LEFT"
        elif lat == "left":
            raw = "RIGHT"   # threat on left → go right
        else:
            raw = "LEFT"    # threat on right → go left

        return self._vote(raw)

    def _vote(self, direction: str) -> str:
        """Decay all votes, boost winner, return leading direction."""
        for k in self._votes:
            self._votes[k] *= (1.0 - self._alpha)
        self._votes[direction] += self._alpha
        self.current = max(self._votes, key=self._votes.get)
        return self.current

    # ------------------------------------------------------------------
    def draw_arrow(self, frame, direction: str,
                   drivable_mask=None, risk: str = "SAFE") -> "np.ndarray":
        """
        Draws a circle-arrow indicator inside the drivable area.
        Semi-transparent dark background keeps it readable on any scene colour.
        """
        h, w = frame.shape[:2]
        color = _DIR_COLORS[direction]

        # ---- find anchor: bottom-centre of drivable zone ----
        if drivable_mask is not None and drivable_mask.any():
            rows = np.where(drivable_mask.any(axis=1))[0]
            bottom_row = int(min(rows.max(), h - 80))
            zone = drivable_mask[max(0, bottom_row - 50):bottom_row + 1, :]
            col_idx = np.where(zone)[1]
            cx = int(np.median(col_idx)) if len(col_idx) > 0 else w // 2
            cy = bottom_row - 40
        else:
            cx, cy = w // 2, h - 100

        # Clamp to frame
        cx = int(np.clip(cx, 50, w - 50))
        cy = int(np.clip(cy, 50, h - 50))

        # ---- dark semi-transparent circle background ----
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), 40, (15, 15, 15), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)
        cv2.circle(frame, (cx, cy), 40, color, 2, cv2.LINE_AA)

        # ---- arrow geometry ----
        L  = 26   # shaft half-length
        TW = 11   # arrowhead half-width
        TL = 13   # arrowhead length

        if direction == "STRAIGHT":
            tail = (cx, cy + L)
            tip  = (cx, cy - L)
            cv2.line(frame, tail, tip, color, 4, cv2.LINE_AA)
            pts = np.int32([[cx - TW, cy - L + TL],
                             [cx + TW, cy - L + TL],
                             [cx,      cy - L - TL // 2]])
            cv2.fillConvexPoly(frame, pts, color, cv2.LINE_AA)

        elif direction == "LEFT":
            tail = (cx + L, cy)
            tip  = (cx - L, cy)
            cv2.line(frame, tail, tip, color, 4, cv2.LINE_AA)
            pts = np.int32([[cx - L + TL, cy - TW],
                             [cx - L + TL, cy + TW],
                             [cx - L - TL // 2, cy]])
            cv2.fillConvexPoly(frame, pts, color, cv2.LINE_AA)

        elif direction == "RIGHT":
            tail = (cx - L, cy)
            tip  = (cx + L, cy)
            cv2.line(frame, tail, tip, color, 4, cv2.LINE_AA)
            pts = np.int32([[cx + L - TL, cy - TW],
                             [cx + L - TL, cy + TW],
                             [cx + L + TL // 2, cy]])
            cv2.fillConvexPoly(frame, pts, color, cv2.LINE_AA)

        # ---- text label under circle ----
        label = _DIR_LABEL[direction]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        lx = cx - tw // 2
        ly = cy + 56
        # small dark pill behind text
        cv2.rectangle(frame, (lx - 4, ly - th - 2), (lx + tw + 4, ly + 4),
                      (15, 15, 15), -1, cv2.LINE_AA)
        cv2.putText(frame, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

        return frame
