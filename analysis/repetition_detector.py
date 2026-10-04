"""
Automatic lever repetition detector module for aircraft lever monitoring.
Implements a robust deterministic state-machine:
REST -> MOVING_FORWARD -> AT_PEAK -> RETURNING -> REST
Calculates comprehensive per-repetition kinematic and force metrics.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class RepetitionDetector:
    """
    Detects discrete lever strokes and calculates kinematics, duration, range,
    force profile, and wrist deviation metrics for each repetition.
    """

    # State Machine States
    STATE_REST = "REST"
    STATE_MOVING_FORWARD = "MOVING_FORWARD"
    STATE_AT_PEAK = "AT_PEAK"
    STATE_RETURNING = "RETURNING"

    def __init__(
        self,
        min_angle_span: float = 15.0,
        start_threshold_deg: float = 8.0,
        return_threshold_deg: float = 6.0,
        min_stroke_duration: float = 0.4,
        max_stroke_duration: float = 5.0,
    ) -> None:
        self.min_angle_span = min_angle_span
        self.start_threshold_deg = start_threshold_deg
        self.return_threshold_deg = return_threshold_deg
        self.min_stroke_duration = min_stroke_duration
        self.max_stroke_duration = max_stroke_duration

        # State Machine Tracking
        self.current_state: str = self.STATE_REST
        self.repetition_count: int = 0
        self.completed_repetitions: List[Dict[str, Any]] = []
        self.last_repetition_metrics: Optional[Dict[str, Any]] = None

        # Active Stroke Cycle Variables
        self.rest_baseline_angle: Optional[float] = None
        self.stroke_start_time: Optional[float] = None
        self.stroke_start_angle: Optional[float] = None
        self.peak_displacement: float = 0.0
        self.cycle_samples: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """
        Resets repetition detector counters and active state.
        """
        self.current_state = self.STATE_REST
        self.repetition_count = 0
        self.completed_repetitions.clear()
        self.last_repetition_metrics = None
        self.rest_baseline_angle = None
        self.stroke_start_time = None
        self.stroke_start_angle = None
        self.peak_displacement = 0.0
        self.cycle_samples.clear()

    def update(self, fused_record: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Processes a synchronized observation record through the repetition state machine.

        Args:
            fused_record: Timestamped observation dictionary from DataFusion.

        Returns:
            Tuple of (completed_repetition_event_dict or None, current_state_name).
        """
        curr_angle = fused_record.get("movement_angle")
        curr_time = fused_record.get("elapsed_time", fused_record.get("timestamp", 0.0))
        tracking_status = fused_record.get("tracking_status", "LOST")

        # Gracefully handle missing/lost tracking
        if curr_angle is None or tracking_status == "LOST":
            if self.current_state != self.STATE_REST and self.stroke_start_time is not None:
                if (curr_time - self.stroke_start_time) > 1.5:
                    self.current_state = self.STATE_REST
                    self.cycle_samples.clear()
            return None, self.current_state

        ang_speed = fused_record.get("angular_speed", 0.0)
        force_pct = fused_record.get("force_percent")
        wrist_ang = fused_record.get("wrist_angle")
        wrist_dev = fused_record.get("wrist_deviation")

        sample = {
            "time": curr_time,
            "angle": curr_angle,
            "speed": ang_speed,
            "force": force_pct,
            "wrist_angle": wrist_ang,
            "wrist_dev": wrist_dev,
        }

        # Initialize resting baseline if empty
        if self.rest_baseline_angle is None:
            self.rest_baseline_angle = curr_angle

        # -------------------------------------------------------------
        # STATE: REST
        # -------------------------------------------------------------
        if self.current_state == self.STATE_REST:
            # Slowly track resting drift while stationary
            if ang_speed < 4.0:
                self.rest_baseline_angle = 0.95 * self.rest_baseline_angle + 0.05 * curr_angle

            displacement = abs(curr_angle - self.rest_baseline_angle)
            
            # Transition to MOVING_FORWARD on deliberate movement
            if displacement >= self.start_threshold_deg and ang_speed > 3.0:
                self.current_state = self.STATE_MOVING_FORWARD
                self.stroke_start_time = curr_time
                self.stroke_start_angle = self.rest_baseline_angle
                self.peak_displacement = displacement
                self.cycle_samples = [sample]

            return None, self.current_state

        # -------------------------------------------------------------
        # STATE: MOVING_FORWARD
        # -------------------------------------------------------------
        elif self.current_state == self.STATE_MOVING_FORWARD:
            self.cycle_samples.append(sample)
            displacement = abs(curr_angle - self.stroke_start_angle)

            if displacement > self.peak_displacement:
                self.peak_displacement = displacement

            # Turnaround detection: angle reverses by at least 3 degrees after crossing min span
            if self.peak_displacement >= self.min_angle_span:
                if displacement < (self.peak_displacement - 3.0):
                    self.current_state = self.STATE_RETURNING

            # Timeout check: discard if stroke stalls
            if (curr_time - self.stroke_start_time) > self.max_stroke_duration:
                self.current_state = self.STATE_REST
                self.cycle_samples.clear()

            return None, self.current_state

        # -------------------------------------------------------------
        # STATE: RETURNING
        # -------------------------------------------------------------
        elif self.current_state == self.STATE_RETURNING:
            self.cycle_samples.append(sample)
            displacement = abs(curr_angle - self.stroke_start_angle)

            # Check if returned within resting threshold or <= 15% of peak displacement
            if displacement <= self.return_threshold_deg or displacement <= (0.15 * self.peak_displacement):
                duration = curr_time - self.stroke_start_time

                # Validate duration and amplitude criteria
                if duration >= self.min_stroke_duration and self.peak_displacement >= self.min_angle_span:
                    self.repetition_count += 1
                    rep_metrics = self._compute_repetition_metrics(
                        rep_num=self.repetition_count,
                        start_t=self.stroke_start_time,
                        end_t=curr_time,
                        duration=duration,
                        disp_range=self.peak_displacement,
                        samples=self.cycle_samples,
                    )
                    self.completed_repetitions.append(rep_metrics)
                    self.last_repetition_metrics = rep_metrics

                    # Reset cycle
                    self.current_state = self.STATE_REST
                    self.rest_baseline_angle = curr_angle
                    self.cycle_samples.clear()
                    return rep_metrics, self.STATE_REST
                else:
                    # Incomplete or jitter stroke
                    self.current_state = self.STATE_REST
                    self.cycle_samples.clear()

            # Timeout check: discard if return stalls
            if (curr_time - self.stroke_start_time) > self.max_stroke_duration:
                self.current_state = self.STATE_REST
                self.cycle_samples.clear()

            return None, self.current_state

        return None, self.current_state

    def _compute_repetition_metrics(
        self,
        rep_num: int,
        start_t: float,
        end_t: float,
        duration: float,
        disp_range: float,
        samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Computes all frozen summary metrics for a completed repetition.
        """
        forces = [s["force"] for s in samples if s["force"] is not None]
        wrist_angles = [s["wrist_angle"] for s in samples if s["wrist_angle"] is not None]
        wrist_devs = [s["wrist_dev"] for s in samples if s["wrist_dev"] is not None]
        speeds = [s["speed"] for s in samples]

        avg_force = float(np.mean(forces)) if forces else 0.0
        peak_force = float(np.max(forces)) if forces else 0.0

        avg_wrist_angle = float(np.mean(wrist_angles)) if wrist_angles else 0.0
        avg_wrist_dev = float(np.mean(wrist_devs)) if wrist_devs else 0.0
        max_wrist_dev = float(np.max(wrist_devs)) if wrist_devs else 0.0
        avg_speed = float(np.mean(speeds)) if speeds else 0.0

        # Wrist angle at instant of peak force
        wrist_at_peak_force = None
        if forces and len(forces) == len(samples):
            peak_force_idx = int(np.argmax([s["force"] if s["force"] is not None else -1 for s in samples]))
            wrist_at_peak_force = samples[peak_force_idx].get("wrist_angle")

        # Force at instant of maximum movement displacement
        force_at_max_mov = None
        if samples:
            displacements = [abs(s["angle"] - self.stroke_start_angle) for s in samples]
            max_disp_idx = int(np.argmax(displacements))
            force_at_max_mov = samples[max_disp_idx].get("force")

        return {
            "repetition_number": rep_num,
            "start_time": round(start_t, 2),
            "end_time": round(end_t, 2),
            "movement_duration": round(duration, 2),
            "movement_range": round(disp_range, 2),
            "average_force": round(avg_force, 2) if forces else None,
            "peak_force": round(peak_force, 2) if forces else None,
            "average_wrist_angle": round(avg_wrist_angle, 2) if wrist_angles else 0.0,
            "average_wrist_deviation": round(avg_wrist_dev, 2) if wrist_devs else 0.0,
            "maximum_wrist_deviation": round(max_wrist_dev, 2) if wrist_devs else 0.0,
            "average_angular_speed": round(avg_speed, 2),
            "wrist_angle_at_peak_force": round(wrist_at_peak_force, 2) if wrist_at_peak_force is not None else None,
            "force_at_maximum_movement": round(force_at_max_mov, 2) if force_at_max_mov is not None else None,
        }
