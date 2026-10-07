"""
Dual-mode Force sensor acquisition client for the Aircraft Lever Monitoring System.
Supports:
1. ESP32 Wi-Fi HTTP client interacting with the frozen ESP32 Hele_Lever_FSR firmware (/data endpoint).
2. USB / Serial force sensor auto-detection & manual port configuration via PySerial on Windows COM ports.
Provides non-blocking background polling, automatic source arbitration, connection state monitoring, and safe parsing.
"""

import time
import json
import re
import threading
from typing import Dict, Any, Optional, List
import requests

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


class ESP32Reader:
    """
    Acquires real-time normalized grip force percentage data from either ESP32 Wi-Fi
    or USB / Serial connection without blocking camera or vision processing loops.
    """

    # Connection States
    STATE_CONNECTED = "CONNECTED"
    STATE_DISCONNECTED = "DISCONNECTED"
    STATE_ERROR = "ERROR"

    # Modes
    MODE_AUTO = "AUTO"
    MODE_WIFI = "WIFI"
    MODE_SERIAL = "SERIAL"

    # Force Sources
    SOURCE_ESP32_WIFI = "ESP32 Network"
    SOURCE_USB_SERIAL = "USB Serial"
    SOURCE_DISCONNECTED = "Disconnected"

    def __init__(
        self,
        base_url: str = "http://192.168.4.1",
        timeout: float = 0.3,
        poll_interval: float = 0.05,
        serial_port: Optional[str] = None,
        serial_baudrates: Optional[List[int]] = None,
        serial_timeout: float = 0.2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.data_endpoint = f"{self.base_url}/data"
        self.timeout = timeout
        self.poll_interval = poll_interval

        self.mode = self.MODE_AUTO
        self.serial_port = serial_port
        self.serial_baudrates = serial_baudrates or [115200, 9600]
        self.serial_timeout = serial_timeout

        # Connection & Telemetry State
        self.connection_state: str = self.STATE_DISCONNECTED
        self.force_source: str = self.SOURCE_DISCONNECTED
        self.device_info: str = "None"
        self.connection_type: str = "None"
        self.latest_force_percent: Optional[float] = None
        self.latest_timestamp: float = time.perf_counter()
        self.last_success_time: Optional[float] = None
        self.latest_raw_response: Optional[Dict[str, Any]] = None
        self.consecutive_failures: int = 0

        # Statistical Metrics for the active session
        self.force_history: List[float] = []
        self.average_force: Optional[float] = None
        self.peak_force: Optional[float] = None

        # Serial Monitor & Stream Logging Buffers
        self.serial_logs: List[Dict[str, Any]] = []
        self.serial_packet_counter: int = 0
        self.serial_rx_timestamps: List[float] = []
        self.serial_rate_hz: float = 0.0
        self.active_baud_rate: int = 115200
        self._log_id_counter: int = 0
        self.session_serial_records: List[Dict[str, Any]] = []

        # Active serial connection handle
        self._serial_handle: Optional[Any] = None
        self._active_serial_port: Optional[str] = None
        self._last_port_scan_time: float = 0.0

        # Threading & Concurrency Controls
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._session = requests.Session()

    def set_mode(self, mode: str) -> None:
        """Sets active force acquisition mode: AUTO, WIFI, or SERIAL."""
        mode_upper = mode.upper()
        if mode_upper in (self.MODE_AUTO, self.MODE_WIFI, self.MODE_SERIAL):
            with self._lock:
                self.mode = mode_upper
                if self.mode == self.MODE_WIFI and self._serial_handle is not None:
                    self._close_serial()

    def set_wifi_address(self, ip_or_url: str) -> None:
        """Sets the ESP32 IP address or base URL."""
        with self._lock:
            url = ip_or_url.strip()
            if not url.startswith("http://") and not url.startswith("https://"):
                url = f"http://{url}"
            self.base_url = url.rstrip("/")
            self.data_endpoint = f"{self.base_url}/data"

    def set_serial_port(self, port: Optional[str]) -> None:
        """Sets or changes the targeted serial port."""
        with self._lock:
            if port != self.serial_port:
                self._close_serial()
                self.serial_port = port.strip() if port else None

    @staticmethod
    def scan_available_ports() -> List[Dict[str, Any]]:
        """Scans and returns available COM ports with comprehensive hardware metadata."""
        if not SERIAL_AVAILABLE:
            return []
        results = []
        try:
            ports = serial.tools.list_ports.comports()
            for p in ports:
                vid_hex = f"{p.vid:04X}" if p.vid is not None else None
                pid_hex = f"{p.pid:04X}" if p.pid is not None else None
                vid_pid_str = f"{vid_hex}:{pid_hex}" if (vid_hex and pid_hex) else None

                results.append({
                    "port": p.device,
                    "description": p.description or p.device,
                    "manufacturer": p.manufacturer or "Generic / Unknown",
                    "vid": vid_hex,
                    "pid": pid_hex,
                    "vid_pid": vid_pid_str,
                    "hwid": p.hwid or "",
                })
        except Exception:
            pass
        return results

    def check_esp32_network(self, timeout: float = 0.25) -> Dict[str, Any]:
        """Probes the configured ESP32 network address to verify connectivity."""
        clean_ip = self.base_url.replace("http://", "").replace("https://", "")
        try:
            resp = self._session.get(self.data_endpoint, timeout=timeout)
            if resp.status_code == 200:
                return {
                    "name": "ESP32 Force Sensor",
                    "address": clean_ip,
                    "endpoint": self.data_endpoint,
                    "status": "Connected",
                    "connected": True,
                }
        except Exception:
            pass
        return {
            "name": "ESP32 Force Sensor",
            "address": clean_ip,
            "endpoint": self.data_endpoint,
            "status": "Not Found",
            "connected": False,
        }

    def start(self) -> None:
        """Starts the asynchronous background acquisition thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stops background acquisition and closes open serial / network handles."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._close_serial()
        try:
            self._session.close()
        except Exception:
            pass

    def reset_session_metrics(self) -> None:
        """Resets average and peak force metrics."""
        with self._lock:
            self.force_history.clear()
            self.average_force = None
            self.peak_force = None

    def _close_serial(self) -> None:
        """Closes any active serial port safely."""
        if self._serial_handle is not None:
            try:
                self._serial_handle.close()
            except Exception:
                pass
            self._serial_handle = None
            self._active_serial_port = None

    def _poll_loop(self) -> None:
        """
        Continuous non-blocking background worker.
        Handles AUTO, WIFI, or SERIAL mode cleanly.
        """
        while self._running:
            now = time.perf_counter()
            mode = self.mode

            success = False
            if mode == self.MODE_AUTO:
                # Priority 1: Wi-Fi HTTP, Priority 2: USB Serial
                success = self._poll_wifi(now)
                if not success:
                    success = self._poll_serial(now)
            elif mode == self.MODE_WIFI:
                success = self._poll_wifi(now)
            elif mode == self.MODE_SERIAL:
                success = self._poll_serial(now)

            if not success:
                with self._lock:
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= 3:
                        self.connection_state = self.STATE_DISCONNECTED
                        self.force_source = self.SOURCE_DISCONNECTED
                        self.latest_force_percent = None
                        self.connection_type = "None"
                        self.device_info = "None"
                    else:
                        self.connection_state = self.STATE_ERROR

            time.sleep(self.poll_interval)

    def _poll_wifi(self, now: float) -> bool:
        """Attempts to read from ESP32 HTTP /data endpoint."""
        try:
            resp = self._session.get(self.data_endpoint, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                force_val = 0.0
                if "total_force" in data:
                    force_val = float(data["total_force"])
                elif "sensors" in data and isinstance(data["sensors"], list) and len(data["sensors"]) > 0:
                    force_val = float(data["sensors"][0])
                elif "force" in data:
                    force_val = float(data["force"])

                force_clamped = round(max(0.0, min(100.0, force_val)), 1)
                clock_str = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"

                with self._lock:
                    self.latest_force_percent = force_clamped
                    self.latest_timestamp = now
                    self.last_success_time = now
                    self.latest_raw_response = data
                    self.connection_state = self.STATE_CONNECTED
                    self.force_source = self.SOURCE_ESP32_WIFI
                    self.connection_type = "ESP32 Network"
                    self.device_info = self.base_url.replace("http://", "").replace("https://", "")
                    self.consecutive_failures = 0

                    self.force_history.append(force_clamped)
                    if len(self.force_history) > 1000:
                        self.force_history.pop(0)
                    self.average_force = round(sum(self.force_history) / len(self.force_history), 1)
                    self.peak_force = max(self.peak_force or 0.0, force_clamped)

                    # Track packet rate & stream log
                    self.serial_packet_counter += 1
                    self.serial_rx_timestamps.append(now)
                    if len(self.serial_rx_timestamps) > 30:
                        self.serial_rx_timestamps.pop(0)
                    if len(self.serial_rx_timestamps) >= 2:
                        dt = self.serial_rx_timestamps[-1] - self.serial_rx_timestamps[0]
                        if dt > 0.05:
                            self.serial_rate_hz = round((len(self.serial_rx_timestamps) - 1) / dt, 1)

                    self._log_id_counter += 1
                    log_entry = {
                        "id": self._log_id_counter,
                        "time": clock_str,
                        "timestamp": round(now, 3),
                        "raw": json.dumps(data) if isinstance(data, dict) else str(data),
                        "value": force_clamped,
                        "source": "WIFI (ESP32)",
                    }
                    self.serial_logs.append(log_entry)
                    if len(self.serial_logs) > 300:
                        self.serial_logs.pop(0)
                    self.session_serial_records.append(log_entry)
                    if len(self.session_serial_records) > 5000:
                        self.session_serial_records.pop(0)

                # Release serial if Wi-Fi succeeded in auto mode
                if self.mode == self.MODE_AUTO and self._serial_handle is not None:
                    self._close_serial()
                return True
        except Exception:
            pass
        return False

    def _poll_serial(self, now: float) -> bool:
        """Attempts to read from USB Serial port."""
        if not SERIAL_AVAILABLE:
            return False

        if self._serial_handle is None or not self._serial_handle.is_open:
            if (now - self._last_port_scan_time) < 1.5:
                return False
            self._last_port_scan_time = now
            self._try_open_serial()

        if self._serial_handle is None or not self._serial_handle.is_open:
            return False

        try:
            if self._serial_handle.in_waiting > 0:
                line = self._serial_handle.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    parsed_force = self._parse_serial_line(line)
                    if parsed_force is not None:
                        parsed_force = round(parsed_force, 1)
                        port_label = f"USB Serial ({self._active_serial_port})"
                        clock_str = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"

                        with self._lock:
                            self.latest_force_percent = parsed_force
                            self.latest_timestamp = now
                            self.last_success_time = now
                            self.latest_raw_response = {"raw_line": line, "parsed_force": parsed_force}
                            self.connection_state = self.STATE_CONNECTED
                            self.force_source = port_label
                            self.connection_type = "USB Serial"
                            self.device_info = self._active_serial_port or "USB Port"
                            self.consecutive_failures = 0

                            self.force_history.append(parsed_force)
                            if len(self.force_history) > 1000:
                                self.force_history.pop(0)
                            self.average_force = round(sum(self.force_history) / len(self.force_history), 1)
                            self.peak_force = max(self.peak_force or 0.0, parsed_force)

                            # Track packet rate & stream log
                            self.serial_packet_counter += 1
                            self.serial_rx_timestamps.append(now)
                            if len(self.serial_rx_timestamps) > 30:
                                self.serial_rx_timestamps.pop(0)
                            if len(self.serial_rx_timestamps) >= 2:
                                dt = self.serial_rx_timestamps[-1] - self.serial_rx_timestamps[0]
                                if dt > 0.05:
                                    self.serial_rate_hz = round((len(self.serial_rx_timestamps) - 1) / dt, 1)

                            self._log_id_counter += 1
                            log_entry = {
                                "id": self._log_id_counter,
                                "time": clock_str,
                                "timestamp": round(now, 3),
                                "raw": line,
                                "value": parsed_force,
                                "source": f"SERIAL ({self._active_serial_port})",
                            }
                            self.serial_logs.append(log_entry)
                            if len(self.serial_logs) > 300:
                                self.serial_logs.pop(0)
                            self.session_serial_records.append(log_entry)
                            if len(self.session_serial_records) > 5000:
                                self.session_serial_records.pop(0)

                        return True
        except Exception:
            self._close_serial()
            return False

        if self.last_success_time is not None and (now - self.last_success_time) < 1.0:
            return True

        return False

    def _try_open_serial(self) -> None:
        """Scans available COM ports or opens user-configured serial port."""
        if not SERIAL_AVAILABLE:
            return

        candidate_ports = []
        if self.serial_port:
            candidate_ports.append(self.serial_port)
        else:
            ports = serial.tools.list_ports.comports()
            for p in ports:
                candidate_ports.append(p.device)

        for port in candidate_ports:
            for baud in self.serial_baudrates:
                try:
                    ser = serial.Serial(port, baud, timeout=self.serial_timeout)
                    self._serial_handle = ser
                    self._active_serial_port = port
                    self.active_baud_rate = baud
                    return
                except Exception:
                    continue

    def _parse_serial_line(self, line: str) -> Optional[float]:
        """Safely parses force value from JSON or raw text line."""
        if not line:
            return None

        # 1. Try JSON
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                if "total_force" in data:
                    return max(0.0, min(100.0, float(data["total_force"])))
                if "force" in data:
                    return max(0.0, min(100.0, float(data["force"])))
                if "sensors" in data and isinstance(data["sensors"], list) and len(data["sensors"]) > 0:
                    return max(0.0, min(100.0, float(data["sensors"][0])))
        except Exception:
            pass

        # 2. Try regex for numeric value
        match = re.search(r"[-+]?\d*\.?\d+", line)
        if match:
            try:
                val = float(match.group())
                return max(0.0, min(100.0, val))
            except ValueError:
                pass

        return None

    def get_serial_stream(self, since_id: int = 0) -> Dict[str, Any]:
        """Returns new serial log items since since_id, telemetry stats, and stream rate."""
        with self._lock:
            new_logs = [log for log in self.serial_logs if log["id"] > since_id]
            last_packet_time = self.serial_logs[-1]["time"] if self.serial_logs else "--"
            return {
                "logs": new_logs,
                "packet_rate_hz": self.serial_rate_hz if (self.connection_state == self.STATE_CONNECTED) else 0.0,
                "total_packets": self.serial_packet_counter,
                "active_port": self._active_serial_port or "Auto-Detecting",
                "active_baud": self.active_baud_rate,
                "connection_state": self.connection_state,
                "latest_value": self.latest_force_percent,
                "last_id": self._log_id_counter,
                "last_packet_time": last_packet_time,
            }

    def clear_serial_logs(self) -> None:
        """Clears buffered real-time serial terminal entries."""
        with self._lock:
            self.serial_logs.clear()

    def generate_serial_csv(self) -> str:
        """Generates a downloadable CSV string of all session serial stream packets."""
        lines = ["Timestamp,Clock_Time,Source,Parsed_Force_Percent,Raw_Data"]
        with self._lock:
            records = list(self.session_serial_records)
        for r in records:
            raw_escaped = '"' + r.get("raw", "").replace('"', '""') + '"'
            lines.append(f"{r.get('timestamp')},{r.get('time')},{r.get('source')},{r.get('value')},{raw_escaped}")
        return "\n".join(lines)

    def read_force(self) -> Dict[str, Any]:
        """
        Thread-safe method returning the latest timestamped force measurement.
        """
        with self._lock:
            return {
                "timestamp": self.latest_timestamp,
                "force_percent": self.latest_force_percent,
                "average_force": self.average_force,
                "peak_force": self.peak_force,
                "connection_state": self.connection_state,
                "force_source": self.force_source,
                "connection_type": self.connection_type,
                "device_info": self.device_info,
                "mode": self.mode,
                "is_connected": (self.connection_state == self.STATE_CONNECTED),
                "last_success_time": self.last_success_time,
                "raw_response": self.latest_raw_response,
            }
