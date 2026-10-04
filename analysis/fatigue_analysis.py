"""
Fatigue-Associated Behavioural Deviation Analysis module for the Aircraft Lever System.
Compares ongoing operator biomechanics and force consistency against their initial session baseline.
Does NOT claim medical/clinical fatigue diagnosis; reports behavioral deviation from baseline.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class FatigueAnalyzer:
    """
    Evaluates fatigue-associated behavioural deviations by comparing recent
    lever strokes and posture metrics against the operator's calibrated baseline.
    """

    def __init__(
        self,
        baseline_reps_required: int = 3,
        recent_window_size: int = 3,
        low_threshold: float = 0.15,
        moderate_threshold: float = 0.35,
    ) -> None:
        self.baseline_reps_required = baseline_reps_required
        self.recent_window_size = recent_window_size
        self.low_threshold = low_threshold
        self.moderate_threshold = moderate_threshold

        # Baseline State
        self.is_baseline_established: bool = False
        self.baseline_metrics: Optional[Dict[str, float]] = None

    def reset(self) -> None:
        """
        Resets baseline and analysis state.
        """
        self.is_baseline_established = False
        self.baseline_metrics = None

    def establish_baseline(self, initial_reps: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Computes operator baseline reference metrics from initial valid repetitions.
        """
        wrist_devs = [r.get("average_wrist_deviation", 0.0) for r in initial_reps]
        durations = [r.get("movement_duration", 1.0) for r in initial_reps]
        speeds = [r.get("average_angular_speed", 20.0) for r in initial_reps]
        ranges = [r.get("movement_range", 15.0) for r in initial_reps]
        
        forces = [r["average_force"] for r in initial_reps if r.get("average_force") is not None]
        peak_forces = [r["peak_force"] for r in initial_reps if r.get("peak_force") is not None]

        base_wrist_dev = float(np.mean(wrist_devs)) if wrist_devs else 5.0
        base_duration = float(np.mean(durations)) if durations else 1.0
        base_speed = float(np.mean(speeds)) if speeds else 20.0
        base_force = float(np.mean(forces)) if forces else 0.0
        base_force_var = float(np.std(forces)) if len(forces) > 1 else 2.0
        base_mov_var = float(np.std(ranges)) if len(ranges) > 1 else 2.0

        self.baseline_metrics = {
            "base_wrist_deviation": max(0.5, base_wrist_dev),
            "base_duration": max(0.2, base_duration),
            "base_angular_speed": max(1.0, base_speed),
            "base_force": max(0.0, base_force),
            "base_force_variability": max(0.5, base_force_var),
            "base_movement_variability": max(0.5, base_mov_var),
        }
        self.is_baseline_established = True
        return self.baseline_metrics

    def analyze(self, completed_reps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes repetition history against baseline and returns fatigue-associated deviation levels.

        Args:
            completed_reps: List of completed repetition metric dictionaries from RepetitionDetector.

        Returns:
            Dictionary containing deviation level, composite score, percentage deviations,
            and formatted contributing factor text.
        """
        total_reps = len(completed_reps)

        # 1. Handle insufficient baseline data explicitly
        if total_reps < self.baseline_reps_required:
            return {
                "status": "COLLECTING_BASELINE",
                "fatigue_level": "INITIALIZING",
                "composite_deviation_score": 0.0,
                "reps_collected": total_reps,
                "reps_required": self.baseline_reps_required,
                "percent_changes": {},
                "contributing_factors": {},
                "explanation_text": (
                    f"Establishing baseline: {total_reps}/{self.baseline_reps_required} "
                    "initial repetitions completed."
                ),
            }

        # 2. Establish baseline if not already established
        if not self.is_baseline_established:
            self.establish_baseline(completed_reps[:self.baseline_reps_required])

        # 3. Extract recent window of repetitions
        recent_reps = completed_reps[-self.recent_window_size:]

        rec_wrist_dev = float(np.mean([r.get("average_wrist_deviation", 0.0) for r in recent_reps]))
        rec_duration = float(np.mean([r.get("movement_duration", 1.0) for r in recent_reps]))
        rec_speed = float(np.mean([r.get("average_angular_speed", 20.0) for r in recent_reps]))
        rec_ranges = [r.get("movement_range", 15.0) for r in recent_reps]
        rec_forces = [r["average_force"] for r in recent_reps if r.get("average_force") is not None]

        rec_force_var = float(np.std(rec_forces)) if len(rec_forces) > 1 else (self.baseline_metrics["base_force_variability"] if self.baseline_metrics else 2.0)
        rec_mov_var = float(np.std(rec_ranges)) if len(rec_ranges) > 1 else (self.baseline_metrics["base_movement_variability"] if self.baseline_metrics else 2.0)

        bm = self.baseline_metrics

        # 4. Calculate percentage deviations from baseline
        # Posture deviation: increased wrist deviation (+) indicates poorer ergonomics
        delta_wrist_pct = ((rec_wrist_dev - bm["base_wrist_deviation"]) / bm["base_wrist_deviation"]) * 100.0

        # Duration: increased stroke duration (+) indicates slower execution
        delta_dur_pct = ((rec_duration - bm["base_duration"]) / bm["base_duration"]) * 100.0

        # Speed: decreased angular speed (-) indicates sluggish lever stroke
        delta_speed_pct = ((rec_speed - bm["base_angular_speed"]) / bm["base_angular_speed"]) * 100.0

        # Force variability: increased force variation (+) indicates unsteadiness / erratic grip
        delta_force_var_pct = ((rec_force_var - bm["base_force_variability"]) / bm["base_force_variability"]) * 100.0

        # Movement consistency: increased range variation (+) indicates degraded repeatability
        delta_mov_var_pct = ((rec_mov_var - bm["base_movement_variability"]) / bm["base_movement_variability"]) * 100.0

        percent_changes = {
            "wrist_deviation_pct": round(delta_wrist_pct, 1),
            "movement_duration_pct": round(delta_dur_pct, 1),
            "angular_speed_pct": round(delta_speed_pct, 1),
            "force_variability_pct": round(delta_force_var_pct, 1),
            "movement_variability_pct": round(delta_mov_var_pct, 1),
        }

        # 5. Composite Normalized Behavioural Deviation Score (0.0 to 1.0+)
        # Combines ergonomic drift (30%), duration increase (25%), speed reduction (25%), and force variation (20%)
        w_dev = max(0.0, delta_wrist_pct / 100.0) * 0.30
        w_dur = max(0.0, delta_dur_pct / 100.0) * 0.25
        w_spd = max(0.0, -delta_speed_pct / 100.0) * 0.25
        w_fvar = max(0.0, delta_force_var_pct / 100.0) * 0.20

        composite_score = float(w_dev + w_dur + w_spd + w_fvar)

        # 6. Rule-Based Classification: LOW / MODERATE / HIGH
        if composite_score < self.low_threshold:
            fatigue_level = "LOW"
        elif composite_score < self.moderate_threshold:
            fatigue_level = "MODERATE"
        else:
            fatigue_level = "HIGH"

        contributing_factors = {
            "Wrist deviation": f"{delta_wrist_pct:+.1f}%",
            "Movement duration": f"{delta_dur_pct:+.1f}%",
            "Angular speed": f"{delta_speed_pct:+.1f}%",
            "Force variability": f"{delta_force_var_pct:+.1f}%",
        }

        # Formatted Explanation String
        explanation_lines = [
            f"Fatigue-Associated Deviation: {fatigue_level}",
            "",
            "Contributing changes:",
            f"Wrist deviation: {delta_wrist_pct:+.1f}%",
            f"Movement duration: {delta_dur_pct:+.1f}%",
            f"Angular speed: {delta_speed_pct:+.1f}%",
            f"Force variability: {delta_force_var_pct:+.1f}%",
        ]
        explanation_text = "\n".join(explanation_lines)

        return {
            "status": "ACTIVE_MONITORING",
            "fatigue_level": fatigue_level,
            "composite_deviation_score": round(composite_score, 3),
            "reps_collected": total_reps,
            "reps_required": self.baseline_reps_required,
            "percent_changes": percent_changes,
            "contributing_factors": contributing_factors,
            "explanation_text": explanation_text,
            "baseline_metrics": self.baseline_metrics,
        }
