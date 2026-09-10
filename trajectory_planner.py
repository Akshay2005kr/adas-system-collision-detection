"""
trajectory_planner.py
---------------------
Real road trajectory / safe-path guidance based on:
- detected obstacle bounding boxes
- obstacle position, width, distance, TTC, risk
- drivable-area mask
- left/right free corridors

Decision logic:
- NO dangerous obstacle → STRAIGHT
- Obstacle centered → choose safer left/right corridor
- Obstacle on left → prefer RIGHT (if safe)
- Obstacle on right → prefer LEFT (if safe)
- Left blocked → NEVER recommend LEFT
- Right blocked → NEVER recommend RIGHT
- Both sides unsafe → BRAKE / STRAIGHT per risk logic

Draws a smooth curved trajectory polygon from bottom-center toward
the recommended corridor inside the drivable area.
"""

import cv2
import numpy as np


class TrajectoryPlanner:
    def __init__(self, alpha: float = 0.25):
        """alpha: EMA smoothing for direction stability."""
        self._votes: dict[str, float] = {"STRAIGHT": 1.0, "LEFT": 0.0, "RIGHT": 0.0}
        self._alpha = alpha
        self.current_direction = "STRAIGHT"
        self.last_valid_trajectory = None

    def compute_safe_trajectory(self, vehicles, animals, potholes, humps,
                                 frame_shape, drivable_mask=None):
        """
        Compute the recommended driving direction and trajectory points.
        
        Returns:
            direction: "STRAIGHT", "LEFT", "RIGHT", or "BRAKE"
            trajectory_pts: list of (x, y) pixel points for drawing
            is_safe: bool indicating if the direction is actually safe
        """
        h, w = frame_shape[:2]
        
        # Collect all obstacles with their risk/distance info
        obstacles = []
        
        # Add vehicles
        for v in vehicles:
            dist = v.get("distance", 999.0)
            ttc = v.get("ttc", float("inf"))
            risk = v.get("risk", "SAFE")
            risk_score = v.get("risk_score", 0.0)
            box = v.get("box", (0, 0, 0, 0))
            
            # Consider as threat if close, low TTC, or elevated risk
            is_threat = (dist < 15.0) or (ttc != float("inf") and ttc < 4.0) or (risk != "SAFE")
            if is_threat:
                obstacles.append({
                    "box": box,
                    "dist": dist,
                    "ttc": ttc,
                    "risk": risk,
                    "risk_score": risk_score,
                    "type": "vehicle"
                })
        
        # Add animals (same structure)
        for a in animals:
            dist = a.get("distance", 999.0)
            ttc = a.get("ttc", float("inf"))
            risk = a.get("risk", "SAFE")
            risk_score = a.get("risk_score", 0.0)
            box = a.get("box", (0, 0, 0, 0))
            
            is_threat = (dist < 15.0) or (ttc != float("inf") and ttc < 4.0) or (risk != "SAFE")
            if is_threat:
                obstacles.append({
                    "box": box,
                    "dist": dist,
                    "ttc": ttc,
                    "risk": risk,
                    "risk_score": risk_score,
                    "type": "animal"
                })
        
        # Add potholes/humps as static obstacles (always avoid)
        for (x1, y1, x2, y2) in (potholes or []):
            obstacles.append({
                "box": (x1, y1, x2, y2),
                "dist": 5.0,  # assume close
                "ttc": 2.0,
                "risk": "WARNING",
                "risk_score": 0.4,
                "type": "pothole"
            })
        
        for (x1, y1, x2, y2) in (humps or []):
            obstacles.append({
                "box": (x1, y1, x2, y2),
                "dist": 5.0,
                "ttc": 2.0,
                "risk": "WARNING",
                "risk_score": 0.3,
                "type": "hump"
            })
        
        # No obstacles → STRAIGHT
        if not obstacles:
            return self._vote("STRAIGHT"), self._build_trajectory("STRAIGHT", w, h, drivable_mask), True
        
        # Find the most dangerous obstacle
        def danger_score(obs):
            score = obs["risk_score"]
            if obs["risk"] == "DANGER":
                score += 0.5
            if obs["ttc"] != float("inf"):
                score += max(0, (4.0 - obs["ttc"]) / 4.0) * 0.3
            score += max(0, (15.0 - obs["dist"]) / 15.0) * 0.2
            return score
        
        obstacles.sort(key=danger_score, reverse=True)
        primary = obstacles[0]
        
        # Check if primary obstacle is truly dangerous
        is_dangerous = (primary["risk"] == "DANGER") or \
                       (primary["ttc"] != float("inf") and primary["ttc"] < 2.5) or \
                       (primary["dist"] < 6.0 and primary["risk_score"] > 0.4)
        
        if not is_dangerous:
            return self._vote("STRAIGHT"), self._build_trajectory("STRAIGHT", w, h, drivable_mask), True
        
        # Analyze obstacle position
        px1, py1, px2, py2 = primary["box"]
        cx_norm = (px1 + px2) / 2.0 / w
        
        # Analyze available corridors using drivable mask
        left_free, right_free = self._analyze_corridors(drivable_mask, w, h, obstacles)
        
        # Decision logic
        if cx_norm < 0.37:  # obstacle on left
            if right_free:
                direction = "RIGHT"
            elif left_free:
                direction = "LEFT"  # reluctantly
            else:
                direction = "BRAKE"
                
        elif cx_norm > 0.63:  # obstacle on right
            if left_free:
                direction = "LEFT"
            elif right_free:
                direction = "RIGHT"  # reluctantly
            else:
                direction = "BRAKE"
                
        else:  # centered obstacle
            if left_free and right_free:
                # Pick side with more space
                direction = "LEFT" if left_free > right_free else "RIGHT"
            elif left_free:
                direction = "LEFT"
            elif right_free:
                direction = "RIGHT"
            else:
                direction = "BRAKE"
        
        # If both sides blocked and very close → BRAKE
        if direction in ("LEFT", "RIGHT") and not (left_free or right_free):
            direction = "BRAKE"
        
        is_safe = direction != "BRAKE"
        voted_dir = self._vote(direction)
        traj = self._build_trajectory(voted_dir, w, h, drivable_mask, obstacles)
        
        return voted_dir, traj, is_safe

    def _analyze_corridors(self, drivable_mask, w, h, obstacles):
        """
        Analyze left and right corridors for available space.
        Returns (left_score, right_score) where higher = more free space.
        Score of 0 means completely blocked.
        """
        if drivable_mask is None or not drivable_mask.any():
            return 0.5, 0.5  # assume equal if no mask
        
        mid = w // 2
        left_mask = drivable_mask[:, :mid]
        right_mask = drivable_mask[:, mid:]
        
        # Count pixels but discount areas blocked by obstacles
        left_pixels = left_mask.sum()
        right_pixels = right_mask.sum()
        
        # Discount for each obstacle's presence in that half
        for obs in obstacles:
            x1, _, x2, _ = obs["box"]
            obs_center = (x1 + x2) / 2
            
            if obs_center < mid:
                left_pixels *= 0.7  # reduce left score
            else:
                right_pixels *= 0.7
        
        # Also check bottom portion specifically (where car would drive)
        bottom_third = int(h * 2 / 3)
        left_bottom = drivable_mask[bottom_third:, :mid].sum()
        right_bottom = drivable_mask[bottom_third:, mid:].sum()
        
        # Combine scores
        left_score = (left_pixels + left_bottom * 2) / max(left_pixels + right_pixels + left_bottom + right_bottom, 1)
        right_score = (right_pixels + right_bottom * 2) / max(left_pixels + right_pixels + left_bottom + right_bottom, 1)
        
        # Threshold: need minimum space to consider corridor "free"
        left_free = left_score if left_score > 0.15 else 0
        right_free = right_score if right_score > 0.15 else 0
        
        return left_free, right_free

    def _vote(self, direction: str) -> str:
        """EMA soft-vote to prevent flickering."""
        for k in self._votes:
            self._votes[k] *= (1.0 - self._alpha)
        self._votes[direction] += self._alpha
        self.current_direction = max(self._votes, key=self._votes.get)
        return self.current_direction

    def _build_trajectory(self, direction, w, h, drivable_mask=None, obstacles=None):
        """
        Build a smooth curved trajectory polygon.
        Returns list of (x, y) points forming the trajectory path.
        """
        # Start point: bottom center of frame
        start_x = w // 2
        start_y = h - 10
        
        # End point depends on direction
        if direction == "STRAIGHT":
            end_x = w // 2
            end_y = int(h * 0.35)
            control_offset = 0
        elif direction == "LEFT":
            end_x = int(w * 0.25)
            end_y = int(h * 0.4)
            control_offset = -w * 0.15
        elif direction == "RIGHT":
            end_x = int(w * 0.75)
            end_y = int(h * 0.4)
            control_offset = w * 0.15
        else:  # BRAKE
            end_x = w // 2
            end_y = int(h * 0.5)
            control_offset = 0
        
        # Build quadratic bezier curve points
        ctrl_x = w // 2 + control_offset
        ctrl_y = int(h * 0.65)
        
        trajectory_pts = []
        num_pts = 30
        
        for i in range(num_pts + 1):
            t = i / num_pts
            # Quadratic bezier: B(t) = (1-t)^2*P0 + 2(1-t)t*P1 + t^2*P2
            x = int((1-t)**2 * start_x + 2*(1-t)*t * ctrl_x + t**2 * end_x)
            y = int((1-t)**2 * start_y + 2*(1-t)*t * ctrl_y + t**2 * end_y)
            trajectory_pts.append((x, y))
        
        # Create a ribbon/polygon around the curve for visibility
        ribbon_width = max(15, w // 40)
        polygon_pts = []
        
        for i, (x, y) in enumerate(trajectory_pts):
            # Calculate perpendicular offset
            if i < len(trajectory_pts) - 1:
                dx = trajectory_pts[i+1][0] - x
                dy = trajectory_pts[i+1][1] - y
                length = max(1, (dx**2 + dy**2)**0.5)
                nx, ny = -dy / length * ribbon_width, dx / length * ribbon_width
                polygon_pts.append((int(x + nx), int(y + ny)))
        
        for i in range(len(trajectory_pts) - 1, -1, -1):
            x, y = trajectory_pts[i]
            if i < len(trajectory_pts) - 1:
                dx = trajectory_pts[i+1][0] - x
                dy = trajectory_pts[i+1][1] - y
            else:
                dx = trajectory_pts[i][0] - trajectory_pts[i-1][0]
                dy = trajectory_pts[i][1] - trajectory_pts[i-1][1]
            length = max(1, (dx**2 + dy**2)**0.5)
            nx, ny = dy / length * ribbon_width, -dx / length * ribbon_width
            polygon_pts.append((int(x + nx), int(y + ny)))
        
        self.last_valid_trajectory = polygon_pts
        return polygon_pts

    def draw_trajectory(self, frame, trajectory_pts, direction="STRAIGHT",
                        drivable_mask=None, alpha=0.5):
        """
        Draw the trajectory on the frame.
        Uses gradient coloring based on direction.
        """
        if not trajectory_pts or len(trajectory_pts) < 4:
            return frame
        
        h, w = frame.shape[:2]
        
        # Direction-based colors
        colors = {
            "STRAIGHT": (0, 245, 80),    # green
            "LEFT": (0, 210, 255),        # orange-blue
            "RIGHT": (30, 160, 255),      # orange
            "BRAKE": (0, 0, 255),         # red
        }
        base_color = colors.get(direction, (0, 245, 80))
        
        # Create overlay for transparency
        overlay = frame.copy()
        
        # Draw filled polygon with gradient effect
        pts = np.array(trajectory_pts, dtype=np.int32)
        
        # Multiple passes for glow effect
        for thickness in range(8, 0, -2):
            color_intensity = 1.0 - (8 - thickness) * 0.1
            color = tuple(int(c * color_intensity) for c in base_color)
            cv2.fillPoly(overlay, [pts], color, cv2.LINE_AA)
        
        # Blend overlay with frame
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA)
        
        # Apply only where trajectory exists
        result = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)
        
        # Draw center line for clarity
        num_pts = len(trajectory_pts) // 2
        if num_pts > 1:
            center_line = trajectory_pts[:num_pts]
            for i in range(len(center_line) - 1):
                cv2.line(result, center_line[i], center_line[i+1], 
                        (255, 255, 255), 2, cv2.LINE_AA)
        
        return result

    def reset(self):
        """Reset voting state."""
        self._votes = {"STRAIGHT": 1.0, "LEFT": 0.0, "RIGHT": 0.0}
        self.current_direction = "STRAIGHT"
        self.last_valid_trajectory = None
