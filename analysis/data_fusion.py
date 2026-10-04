"""
Data fusion and timestamp synchronization module between vision and force streams.
Synchronizes asynchronous webcam posture metrics and ESP32 force sensor readings
without blocking or fabricating missing data.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd


class DataFusion:
    """
    Fuses real-time vision-based biomechanical tracking with ESP32 force sensing
    into a unified, timestamp-aligned observation stream.
    """

    def __init__(
        self,
        max_history: int = 3000,
        max_staleness_sec: float = 0.5,
    ) -> None:
        self.max_history = max_history
        self.max_staleness_sec = max_staleness_sec
        self.start_time: Optional[float] = None
        self.synchronized_records: List[Dict[str, Any]] = []

    def reset_session(self) -> None:
        """
        Clears session history and resets the reference start timestamp.
        """
        self.start_time = None
        self.synchronized_records.clear()

    def fuse(
        self,
        angles_data: Dict[str, Any],
        sensor_data: Dict[str, Any],
        tracking_status: str,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Aligns the latest camera posture measurements with the latest ESP32 force reading
        using precise unified timestamps.

        Args:
            angles_data: Processed posture dictionary from AngleCalculator.
            sensor_data: Force sensor dictionary from ESP32Reader.
            tracking_status: Current tracking quality state ('GOOD', 'LIMITED', 'LOST').
            timestamp: Optional monotonic timestamp; defaults to time.perf_counter().

        Returns:
            Synchronized observation dictionary.
        """
        now = timestamp if timestamp is not None else time.perf_counter()
        if self.start_time is None:
            self.start_time = now

        elapsed_time = max(0.0, now - self.start_time)

        # 1. Camera Measurement Extraction (Safe representation without fabrication)
        camera_valid = angles_data.get("valid", False) or angles_data.get("is_holding", False)
        
        wrist_angle = angles_data.get("wrist_angle") if camera_valid else None
        wrist_deviation = angles_data.get("wrist_deviation") if camera_valid else None
        movement_angle = angles_data.get("lever_angle") if camera_valid else None
        movement_range = angles_data.get("movement_range", 0.0)
        angular_speed = angles_data.get("angular_speed", 0.0)

        # 2. Sensor Measurement Extraction (Safe representation for offline/disconnected ESP32)
        sensor_connected = sensor_data.get("is_connected", False)
        sensor_status = sensor_data.get("connection_state", "DISCONNECTED")
        force_source = sensor_data.get("force_source", "Disconnected")
        sensor_timestamp = sensor_data.get("timestamp", 0.0)

        is_sensor_fresh = sensor_connected and ((now - sensor_timestamp) <= self.max_staleness_sec)
        
        if sensor_connected and is_sensor_fresh:
            raw_force = sensor_data.get("force_percent")
            force_percent = float(raw_force) if raw_force is not None else None
            force_valid = (force_percent is not None)
        else:
            # When sensor is offline/disconnected, force is None (not fabricated)
            force_percent = None
            force_valid = False

        record: Dict[str, Any] = {
            "timestamp": round(now, 4),
            "elapsed_time": round(elapsed_time, 2),
            "force_percent": round(force_percent, 2) if force_percent is not None else None,
            "force_valid": force_valid,
            "force_source": force_source,
            "wrist_angle": round(wrist_angle, 2) if wrist_angle is not None else None,
            "wrist_deviation": round(wrist_deviation, 2) if wrist_deviation is not None else None,
            "movement_angle": round(movement_angle, 2) if movement_angle is not None else None,
            "movement_range": round(movement_range, 2),
            "angular_speed": round(angular_speed, 2),
            "tracking_status": tracking_status,
            "sensor_status": sensor_status,
            "is_holding": angles_data.get("is_holding", False),
            "is_calibrated": angles_data.get("is_calibrated", False),
            "neutral_wrist_angle": angles_data.get("neutral_wrist_angle"),
        }

        self.synchronized_records.append(record)
        if len(self.synchronized_records) > self.max_history:
            self.synchronized_records.pop(0)

        return record

    def format_observation(self, record: Dict[str, Any]) -> str:
        """
        Formats a synchronized observation for terminal or logging display.
        """
        elapsed = f"{record.get('elapsed_time', 0.0):.2f} s"
        
        force_val = record.get("force_percent")
        source = record.get("force_source", "Disconnected")
        force_str = f"{force_val:.1f} % ({source})" if force_val is not None else f"N/A ({source})"

        wrist_val = record.get("wrist_angle")
        wrist_str = f"{wrist_val:+.1f} deg" if wrist_val is not None else "N/A (Lost)"

        dev_val = record.get("wrist_deviation")
        dev_str = f"{dev_val:.1f} deg" if dev_val is not None else "N/A"

        mov_val = record.get("movement_angle")
        mov_str = f"{mov_val:+.1f} deg" if mov_val is not None else "N/A"

        speed = record.get("angular_speed", 0.0)
        tracking = record.get("tracking_status", "LOST")

        return (
            f"Time: {elapsed}\n"
            f"Force: {force_str}\n"
            f"Wrist Angle: {wrist_str}\n"
            f"Wrist Deviation: {dev_str}\n"
            f"Movement Angle: {mov_str}\n"
            f"Angular Speed: {speed:.1f} deg/s\n"
            f"Tracking: {tracking}"
        )

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """
        Returns the most recent fused record.
        """
        if self.synchronized_records:
            return self.synchronized_records[-1]
        return None

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the in-memory synchronized session history to a Pandas DataFrame.
        """
        if not self.synchronized_records:
            return pd.DataFrame()
        return pd.DataFrame(self.synchronized_records)


class SessionRecorder:
    """
    Manages experimental recording sessions for the aircraft lever monitoring system.
    Records synchronized multi-sensor observations during an active trial and saves
    a single structured CSV file containing complete experimental telemetry and metadata.
    """

    def __init__(self, sessions_dir: Optional[Path] = None) -> None:
        from pathlib import Path
        import config

        self.sessions_dir = Path(sessions_dir) if sessions_dir is not None else config.SESSIONS_DIR
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        self.is_recording: bool = False
        self.session_id: Optional[str] = None
        self.start_iso_time: Optional[str] = None
        self.start_perf_time: Optional[float] = None
        self.stop_perf_time: Optional[float] = None
        self.records: List[Dict[str, Any]] = []
        self.last_saved_path: Optional[Path] = None

    def start_session(
        self,
        session_id: Optional[str] = None,
        participant_id: Optional[str] = None,
        trial_number: Optional[int] = None,
    ) -> str:
        """
        Starts a new experimental recording session, auto-generating timestamped session ID.

        Returns:
            The active session ID (e.g. session_20260919_153000).
        """
        now_dt = datetime.now()
        dt_str = now_dt.strftime("%Y%m%d_%H%M%S")
        self.start_iso_time = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        self.start_perf_time = time.perf_counter()
        self.stop_perf_time = None

        if session_id is None or not str(session_id).strip():
            self.session_id = f"session_{dt_str}"
        else:
            self.session_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(session_id).strip())

        self.records.clear()
        self.is_recording = True
        self.last_saved_path = None
        return self.session_id

    def record_observation(
        self,
        fused_record: Dict[str, Any],
        repetition: int = 0,
        fatigue_deviation: str = "INITIALIZING",
    ) -> None:
        """
        Appends a synchronized multi-sensor observation to the active session buffer.

        Args:
            fused_record: Synchronized data point from DataFusion.
            repetition: Current repetition counter from RepetitionDetector.
            fatigue_deviation: Current fatigue-associated deviation state.
        """
        if not self.is_recording or self.start_perf_time is None:
            return

        now = fused_record.get("timestamp", time.perf_counter())
        elapsed = max(0.0, time.perf_counter() - self.start_perf_time)

        row = {
            "timestamp": round(now, 4),
            "elapsed_time": round(elapsed, 2),
            "force_percent": fused_record.get("force_percent"),
            "force_source": fused_record.get("force_source", "Disconnected"),
            "wrist_angle": fused_record.get("wrist_angle"),
            "wrist_deviation": fused_record.get("wrist_deviation"),
            "movement_angle": fused_record.get("movement_angle"),
            "movement_range": round(float(fused_record.get("movement_range", 0.0)), 2),
            "angular_speed": round(float(fused_record.get("angular_speed", 0.0)), 2),
            "repetition": int(repetition),
            "tracking_status": fused_record.get("tracking_status", "LOST"),
            "fatigue_deviation": str(fatigue_deviation),
        }
        self.records.append(row)

    def stop_session(self) -> Optional[Path]:
        """
        Stops active recording session and writes a single primary CSV file.

        Returns:
            Path to the saved CSV file, or None if no active session or records.
        """
        if not self.is_recording:
            return self.last_saved_path

        self.is_recording = False
        self.stop_perf_time = time.perf_counter()
        total_duration = max(0.0, self.stop_perf_time - (self.start_perf_time or self.stop_perf_time))

        if not self.records:
            return None

        # Format primary CSV filename
        filename = f"{self.session_id}.csv"
        out_path = self.sessions_dir / filename

        # Prepare header metadata comments
        metadata_lines = [
            "# ===================================================================",
            "# Aircraft Lever Operator Ergonomic and Fatigue Monitoring System",
            "# Experimental Session Recording Telemetry",
            "# ===================================================================",
            f"# session_id: {self.session_id}",
            f"# start_time: {self.start_iso_time}",
            f"# duration_seconds: {total_duration:.2f}",
            f"# total_observations: {len(self.records)}",
            "# ===================================================================",
        ]

        df = pd.DataFrame(self.records)

        with open(out_path, "w", encoding="utf-8") as f:
            for line in metadata_lines:
                f.write(line + "\n")
            df.to_csv(f, index=False)

        self.last_saved_path = out_path
        return out_path

    @property
    def record_count(self) -> int:
        """Returns the number of recorded observations in the active session."""
        return len(self.records)

    @property
    def session_duration(self) -> float:
        """Returns active or completed session duration in seconds."""
        if self.start_perf_time is None:
            return 0.0
        if self.is_recording:
            return max(0.0, time.perf_counter() - self.start_perf_time)
        if self.stop_perf_time is not None:
            return max(0.0, self.stop_perf_time - self.start_perf_time)
        return 0.0


def load_session_csv(filepath: Any) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Loads an experimental session CSV file, parsing metadata headers and telemetry observations.

    Args:
        filepath: Path to the session CSV file.

    Returns:
        Tuple of (metadata_dict, dataframe).
    """
    from pathlib import Path
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Session file not found: {p}")

    metadata: Dict[str, Any] = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str.startswith("#"):
                # Extract key-value pairs if present: '# key: value'
                content = line_str.lstrip("#").strip()
                if ":" in content:
                    parts = content.split(":", 1)
                    k = parts[0].strip()
                    v = parts[1].strip()
                    metadata[k] = v
            else:
                break

    df = pd.read_csv(p, comment="#")
    return metadata, df
