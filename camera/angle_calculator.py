"""
Biomechanical posture and lever movement angle calculation module.
Includes Phase 4 Stability Engine:
- Stable neutral-position calibration with variance validation
- Spike rejection / physiological rate-of-change clamping
- Measurement hold buffer during momentary tracking loss
- Adaptive low-jitter EMA smoothing
- Clear distinction between GOOD, LIMITED, and LOST tracking
"""

import math
from typing import Dict, Any, Optional, List
import numpy as np


class AngleCalculator:
    """
    Calculates biomechanical hand/wrist posture angles with stability filtering,
    outlier spike rejection, measurement hold buffers, and calibration validation.
    """

    def __init__(
        self,
        angle_smoothing_alpha: float = 0.35,
        speed_smoothing_alpha: float = 0.30,
        calibration_frames: int = 60,
        max_angle_jump_deg: float = 40.0,
        tracking_hold_frames: int = 12,
        calibration_max_std_deg: float = 8.0,
    ) -> None:
        self.base_angle_alpha = angle_smoothing_alpha
        self.speed_alpha = speed_smoothing_alpha
        self.calibration_target_frames = calibration_frames
        self.max_angle_jump_deg = max_angle_jump_deg
        self.tracking_hold_frames = tracking_hold_frames
        self.calibration_max_std_deg = calibration_max_std_deg

        # Neutral posture calibration state
        self.neutral_wrist_angle: Optional[float] = None
        self.is_calibrating: bool = False
        self.is_calibrated: bool = False
        self.calibration_samples: List[float] = []
        self.calibration_warning: Optional[str] = None

        # Smoothed state variables
        self.smoothed_wrist_angle: Optional[float] = None
        self.smoothed_lever_angle: Optional[float] = None
        self.smoothed_angular_speed: float = 0.0

        # Last confirmed valid measurements
        self.last_valid_wrist_angle: Optional[float] = None
        self.last_valid_lever_angle: Optional[float] = None
        self.last_valid_wrist_deviation: float = 0.0
        self.hold_counter: int = 0
        self.is_holding: bool = False

        # Movement range tracking
        self.min_movement_angle: Optional[float] = None
        self.max_movement_angle: Optional[float] = None

        # Temporal tracking
        self.last_timestamp: Optional[float] = None
        self.last_lever_angle_for_speed: Optional[float] = None

    def start_neutral_calibration(self, target_frames: Optional[int] = None) -> None:
        """
        Initiates neutral posture baseline collection.
        """
        if target_frames is not None:
            self.calibration_target_frames = target_frames
        self.calibration_samples.clear()
        self.calibration_warning = None
        self.is_calibrating = True
        self.is_calibrated = False

    def reset_movement_range(self) -> None:
        """
        Resets the tracked min/max movement angle range.
        """
        ref = self.smoothed_lever_angle if self.smoothed_lever_angle is not None else 0.0
        self.min_movement_angle = ref
        self.max_movement_angle = ref

    def calculate_wrist_angle(self, landmarks_data: Dict[str, Any]) -> Optional[float]:
        """
        Computes anatomical wrist angle between Forearm (Elbow -> Wrist) and Hand (Wrist -> Middle MCP).
        Falls back to vertical reference if elbow is unavailable.
        """
        if not landmarks_data or "wrist_px" not in landmarks_data or "middle_mcp_px" not in landmarks_data:
            return None

        w_x, w_y = landmarks_data["wrist_px"]
        m_x, m_y = landmarks_data["middle_mcp_px"]
        elbow_px = landmarks_data.get("elbow_px")

        h_dx = m_x - w_x
        h_dy = m_y - w_y
        len_h = math.hypot(h_dx, h_dy)

        # Minimum physical proportion check to reject collapsed landmarks
        if len_h < 15:
            return None

        if elbow_px is not None:
            e_x, e_y = elbow_px
            f_dx = w_x - e_x
            f_dy = w_y - e_y
            len_f = math.hypot(f_dx, f_dy)

            if len_f > 15:
                cross = f_dx * h_dy - f_dy * h_dx
                dot = f_dx * h_dx + f_dy * h_dy
                angle_rad = math.atan2(cross, dot)
                return float(math.degrees(angle_rad))

        # Vertical reference fallback
        angle_rad = math.atan2(h_dx, -h_dy)
        return float(math.degrees(angle_rad))

    def calculate_lever_angle(self, landmarks_data: Dict[str, Any]) -> Optional[float]:
        """
        Computes the lever movement angle based on the palm orientation vector.
        """
        if not landmarks_data or "wrist_px" not in landmarks_data or "index_mcp_px" not in landmarks_data:
            return None

        w_x, w_y = landmarks_data["wrist_px"]
        idx_x, idx_y = landmarks_data["index_mcp_px"]

        dx = idx_x - w_x
        dy = idx_y - w_y

        if math.hypot(dx, dy) < 15:
            return None

        angle_rad = math.atan2(-dy, dx)
        return float(math.degrees(angle_rad))

    def update(
        self,
        landmarks_data: Optional[Dict[str, Any]],
        timestamp: float,
        tracking_quality: str = "GOOD",
    ) -> Dict[str, Any]:
        """
        Updates posture and movement angles with spike filtering, hold buffers,
        and calibration validation.
        """
        raw_wrist = None
        raw_lever = None

        if landmarks_data is not None and tracking_quality in ("GOOD", "LIMITED"):
            raw_wrist = self.calculate_wrist_angle(landmarks_data)
            raw_lever = self.calculate_lever_angle(landmarks_data)

        # 1. Measurement Hold Buffer for temporary tracking loss
        if raw_wrist is None or raw_lever is None:
            self.hold_counter += 1
            if self.hold_counter <= self.tracking_hold_frames and self.last_valid_wrist_angle is not None:
                # Hold last valid measurements
                self.is_holding = True
                self.last_timestamp = timestamp
                return {
                    "raw_wrist_angle": None,
                    "wrist_angle": round(self.last_valid_wrist_angle, 2),
                    "neutral_wrist_angle": round(self.neutral_wrist_angle, 2) if self.neutral_wrist_angle is not None else None,
                    "wrist_deviation": round(self.last_valid_wrist_deviation, 2),
                    "raw_lever_angle": None,
                    "lever_angle": round(self.last_valid_lever_angle, 2) if self.last_valid_lever_angle is not None else 0.0,
                    "movement_range": round(self._compute_range(), 2),
                    "angular_speed": 0.0,
                    "is_calibrated": self.is_calibrated,
                    "is_calibrating": self.is_calibrating,
                    "calibration_progress": len(self.calibration_samples) / max(1, self.calibration_target_frames),
                    "calibration_warning": self.calibration_warning,
                    "valid": True,
                    "is_holding": True,
                }
            else:
                # Expired or no prior valid measurement
                self.is_holding = False
                self.last_timestamp = timestamp
                return {
                    "raw_wrist_angle": None,
                    "wrist_angle": round(self.smoothed_wrist_angle, 2) if self.smoothed_wrist_angle is not None else 0.0,
                    "neutral_wrist_angle": round(self.neutral_wrist_angle, 2) if self.neutral_wrist_angle is not None else None,
                    "wrist_deviation": 0.0,
                    "raw_lever_angle": None,
                    "lever_angle": round(self.smoothed_lever_angle, 2) if self.smoothed_lever_angle is not None else 0.0,
                    "movement_range": round(self._compute_range(), 2),
                    "angular_speed": 0.0,
                    "is_calibrated": self.is_calibrated,
                    "is_calibrating": self.is_calibrating,
                    "calibration_progress": len(self.calibration_samples) / max(1, self.calibration_target_frames),
                    "calibration_warning": self.calibration_warning,
                    "valid": False,
                    "is_holding": False,
                }

        # Valid measurement received
        self.hold_counter = 0
        self.is_holding = False

        # 2. Outlier Spike Rejection / Clamping
        if self.smoothed_wrist_angle is not None:
            wrist_jump = abs(raw_wrist - self.smoothed_wrist_angle)
            if wrist_jump > self.max_angle_jump_deg:
                # Clamp raw angle jump to maximum allowable step
                sign = 1.0 if raw_wrist > self.smoothed_wrist_angle else -1.0
                raw_wrist = self.smoothed_wrist_angle + sign * self.max_angle_jump_deg

        if self.smoothed_lever_angle is not None:
            lever_jump = abs(raw_lever - self.smoothed_lever_angle)
            if lever_jump > self.max_angle_jump_deg:
                sign = 1.0 if raw_lever > self.smoothed_lever_angle else -1.0
                raw_lever = self.smoothed_lever_angle + sign * self.max_angle_jump_deg

        # 3. Stable Neutral Calibration with Variance Verification
        if self.is_calibrating:
            if tracking_quality == "GOOD":
                self.calibration_samples.append(raw_wrist)
                if len(self.calibration_samples) >= self.calibration_target_frames:
                    std_dev = float(np.std(self.calibration_samples))
                    if std_dev <= self.calibration_max_std_deg:
                        # Posture is steady and valid
                        self.neutral_wrist_angle = float(np.median(self.calibration_samples))
                        self.is_calibrating = False
                        self.is_calibrated = True
                        self.calibration_warning = None
                    else:
                        # Too much movement during calibration; reset and notify
                        self.calibration_samples.clear()
                        self.calibration_warning = "Calibration unsteady: Please hold hand steady."
            else:
                self.calibration_warning = "Tracking degraded: Keep hand in view."

        # 4. Adaptive EMA Smoothing (responsive during movement, rock-steady when still)
        if self.smoothed_wrist_angle is None:
            self.smoothed_wrist_angle = raw_wrist
        else:
            diff = abs(raw_wrist - self.smoothed_wrist_angle)
            dynamic_alpha = min(0.50, max(0.20, self.base_angle_alpha * (1.0 + diff / 30.0)))
            self.smoothed_wrist_angle = (
                dynamic_alpha * raw_wrist + (1.0 - dynamic_alpha) * self.smoothed_wrist_angle
            )

        if self.smoothed_lever_angle is None:
            self.smoothed_lever_angle = raw_lever
        else:
            diff_l = abs(raw_lever - self.smoothed_lever_angle)
            dynamic_alpha_l = min(0.50, max(0.20, self.base_angle_alpha * (1.0 + diff_l / 30.0)))
            self.smoothed_lever_angle = (
                dynamic_alpha_l * raw_lever + (1.0 - dynamic_alpha_l) * self.smoothed_lever_angle
            )

        # 5. Movement range tracking
        if self.min_movement_angle is None or self.smoothed_lever_angle < self.min_movement_angle:
            self.min_movement_angle = self.smoothed_lever_angle
        if self.max_movement_angle is None or self.smoothed_lever_angle > self.max_movement_angle:
            self.max_movement_angle = self.smoothed_lever_angle

        # 6. Angular speed calculation
        raw_speed = 0.0
        if self.last_timestamp is not None and self.last_lever_angle_for_speed is not None:
            dt = timestamp - self.last_timestamp
            if dt > 0.001:
                d_angle = abs(self.smoothed_lever_angle - self.last_lever_angle_for_speed)
                raw_speed = d_angle / dt

        self.smoothed_angular_speed = (
            self.speed_alpha * raw_speed + (1.0 - self.speed_alpha) * self.smoothed_angular_speed
        )

        self.last_timestamp = timestamp
        self.last_lever_angle_for_speed = self.smoothed_lever_angle

        # 7. Wrist deviation relative to calibrated neutral
        wrist_deviation = self._compute_deviation(self.smoothed_wrist_angle)
        movement_range = self._compute_range()

        # Update last confirmed valid cache
        self.last_valid_wrist_angle = self.smoothed_wrist_angle
        self.last_valid_lever_angle = self.smoothed_lever_angle
        self.last_valid_wrist_deviation = wrist_deviation

        return {
            "raw_wrist_angle": raw_wrist,
            "wrist_angle": round(self.smoothed_wrist_angle, 2),
            "neutral_wrist_angle": round(self.neutral_wrist_angle, 2) if self.neutral_wrist_angle is not None else None,
            "wrist_deviation": round(wrist_deviation, 2),
            "raw_lever_angle": raw_lever,
            "lever_angle": round(self.smoothed_lever_angle, 2),
            "movement_range": round(movement_range, 2),
            "angular_speed": round(self.smoothed_angular_speed, 2),
            "is_calibrated": self.is_calibrated,
            "is_calibrating": self.is_calibrating,
            "calibration_progress": len(self.calibration_samples) / max(1, self.calibration_target_frames),
            "calibration_warning": self.calibration_warning,
            "valid": True,
            "is_holding": False,
        }

    def _compute_deviation(self, current_angle: float) -> float:
        ref = self.neutral_wrist_angle if self.is_calibrated and self.neutral_wrist_angle is not None else 0.0
        return abs(current_angle - ref)

    def _compute_range(self) -> float:
        if self.min_movement_angle is None or self.max_movement_angle is None:
            return 0.0
        return max(0.0, self.max_movement_angle - self.min_movement_angle)
