"""
brake_predictor.py
------------------
Smooth, physics-informed brake % for CollisionAI.

v2 — fixes the "jumps to 40 then snaps to 0" behaviour:

  1. The old code used the ML risk label as a HARD FLOOR
     (raw = max(raw, 48) the instant risk == "DANGER"). Since the
     classifier re-predicts every frame with nothing smoothing the
     label itself, one noisy frame flipping DANGER -> SAFE -> DANGER
     yanked the *target* around before the EMA even got a chance to
     smooth it. Now it accepts an optional continuous `risk_score`
     (0.0-1.0, from CollisionModel.predict_proba(), ideally already
     smoothed per-track upstream) and blends it in proportionally
     instead of snapping to a fixed floor value.
  2. `speed_kmh` now actually feeds the physics: a rough
     required-stopping-distance check (v^2 / 2*mu*g) adds brake demand
     when you're closing faster than you could realistically stop in
     time, even if raw distance still looks "safe".
  3. alpha_fall lowered (0.40 -> 0.25 by default) so releases read as
     an ease-off instead of a snap back down.

Usage:
    from brake_predictor import BrakePredictor
    brake = BrakePredictor()
    ...
    pct, color = brake.update(distance, ttc, risk,
                               speed_kmh=speed, risk_score=score)
    # pct  : float 0-100
    # color: BGR tuple (green / orange / red)
"""


class BrakePredictor:
    """
    Asymmetric EMA brake predictor.

    alpha_rise = 0.18  → ~5-frame smooth ramp-up
    alpha_fall = 0.25  → smooth, gradual release (not a snap to 0)
    """

    def __init__(self, alpha_rise: float = 0.18, alpha_fall: float = 0.25):
        self._ema: float = 0.0
        self.alpha_rise = alpha_rise
        self.alpha_fall = alpha_fall
        self.history: list[float] = []   # for live graph
        self._MAX_HIST = 150

    # ------------------------------------------------------------------
    # Internal: raw physics target (no smoothing yet)
    # ------------------------------------------------------------------
    @staticmethod
    def _raw_target(distance: float, ttc: float, risk: str,
                     speed_kmh: float = 0.0,
                     risk_score: float | None = None) -> float:
        """Continuous 0-100 brake demand from TTC + distance + speed + ML
        risk, before smoothing."""

        # ---- TTC-based curve ----
        if ttc == float("inf") or ttc > 6.0:
            ttc_pct = 0.0
        elif ttc > 4.0:
            # 6 → 4 s  :  0 → 12 %
            ttc_pct = (6.0 - ttc) / 2.0 * 12.0
        elif ttc > 2.5:
            # 4 → 2.5 s : 12 → 45 %
            ttc_pct = 12.0 + (4.0 - ttc) / 1.5 * 33.0
        elif ttc > 1.5:
            # 2.5 → 1.5 s : 45 → 78 %
            ttc_pct = 45.0 + (2.5 - ttc) / 1.0 * 33.0
        elif ttc > 0.8:
            # 1.5 → 0.8 s : 78 → 94 %
            ttc_pct = 78.0 + (1.5 - ttc) / 0.7 * 16.0
        else:
            # < 0.8 s : emergency 94 → 100 %
            ttc_pct = min(100.0, 94.0 + (0.8 - max(ttc, 0.05)) * 8.0)

        # ---- Distance cap (override when too far) ----
        if distance > 20:
            dist_cap = 8.0
        elif distance > 14:
            dist_cap = 28.0
        elif distance > 8:
            dist_cap = 58.0
        elif distance > 4:
            dist_cap = 85.0
        else:
            dist_cap = 100.0

        raw = min(ttc_pct, dist_cap)

        # ---- Speed-informed stopping-distance check ----
        # Rough dry-road required stopping distance: v^2 / (2 * mu * g).
        # If the closest vehicle's distance is less than that, add extra
        # brake demand scaled by how far short of it we are. This is what
        # actually uses "speed" as a physics input rather than just a HUD
        # number.
        if speed_kmh and speed_kmh > 5.0:
            v_ms = speed_kmh / 3.6
            stop_dist = (v_ms ** 2) / (2 * 0.7 * 9.8)
            if distance < stop_dist:
                deficit = min(1.0, (stop_dist - distance) / max(stop_dist, 1e-3))
                raw = max(raw, deficit * 70.0)

        # ---- ML risk: continuous score blends in smoothly if available ----
        if risk_score is not None:
            score = min(1.0, max(0.0, risk_score))
            raw = max(raw, score * 62.0)
        elif risk == "DANGER":
            raw = max(raw, 48.0)
        elif risk == "WARNING":
            raw = max(raw, 14.0)

        return min(100.0, raw)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, distance: float, ttc: float, risk: str,
               speed_kmh: float = 0.0, risk_score: float | None = None):
        """
        Call once per frame.
        Returns (smoothed_pct: float, bgr_color: tuple[int,int,int]).

        risk_score : optional continuous 0.0-1.0 ML confidence (from
                     CollisionModel.predict_proba(), ideally already
                     EMA-smoothed per-track upstream). If omitted, falls
                     back to the old discrete-label floor behaviour.
        """
        target = self._raw_target(distance, ttc, risk, speed_kmh, risk_score)

        # Asymmetric smoothing: ramp up slowly, release gradually
        alpha = self.alpha_rise if target > self._ema else self.alpha_fall
        self._ema = alpha * target + (1.0 - alpha) * self._ema
        self._ema = max(0.0, min(100.0, self._ema))

        self.history.append(self._ema)
        if len(self.history) > self._MAX_HIST:
            self.history.pop(0)

        return self._ema, self._color(self._ema)

    def reset(self):
        self._ema = 0.0
        self.history.clear()

    @property
    def value(self) -> float:
        return self._ema

    @staticmethod
    def _color(pct: float) -> tuple:
        """BGR: green → orange → red as brake %  increases."""
        if pct < 20:
            return (0, 220, 0)
        elif pct < 55:
            return (0, 155, 255)
        else:
            return (0, 0, 235)
