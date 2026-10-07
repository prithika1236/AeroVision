"""
Main Application Server for the Aircraft Lever Operator Ergonomic and Fatigue Monitoring System.
Features:
- Universal Device Discovery (Camera devices, USB Serial COM devices, ESP32 Network)
- Multi-Webcam Selection and Dynamic Switching
- Safe Zero-Buffer Real-Time Video Streaming (Logitech C270 & OpenCV compatible webcams)
- MediaPipe Biomechanical Tracking with Downsampled Inference
- Dual-Mode Force Sensing (ESP32 Network + USB Serial)
- High-Performance Single-Polling REST Architecture
"""

import sys
import os
import time
import threading
import webbrowser
from pathlib import Path
from typing import Dict, Any, Optional, Generator, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, make_response

import config
from camera.camera_discovery import scan_camera_devices, get_connected_external_devices, get_preferred_camera
from camera.hand_tracker import HandTracker
from camera.angle_calculator import AngleCalculator
from sensor.esp32_reader import ESP32Reader
from analysis.data_fusion import DataFusion, SessionRecorder


class LeverMonitorSystem:
    """
    Central orchestrator managing multi-camera acquisition, video discovery,
    hardware force sensor arbitration, data fusion, and telemetry streaming.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.is_running = False
        self.camera_state = "IDLE"
        self.start_perf_time: Optional[float] = None

        # Camera Configuration & Automatic Preferred Device Detection
        try:
            pref = get_preferred_camera()
            self.selected_camera_index = int(pref.get("index", config.CAMERA_INDEX))
            self.selected_camera_name = pref.get("name", f"Camera {self.selected_camera_index}")
        except Exception:
            self.selected_camera_index = config.CAMERA_INDEX
            self.selected_camera_name = f"Camera {config.CAMERA_INDEX}"

        self.active_resolution = f"{config.FRAME_WIDTH}x{config.FRAME_HEIGHT}"
        self.capture_fps = 0.0
        self.process_fps = 0.0

        # Subsystems
        self.tracker = HandTracker(
            max_num_hands=config.MP_MAX_NUM_HANDS,
            min_detection_confidence=config.MP_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MP_MIN_TRACKING_CONFIDENCE,
            infer_width=640,
            infer_height=360,
        )
        self.calculator = AngleCalculator(
            angle_smoothing_alpha=config.ANGLE_SMOOTHING_ALPHA,
            speed_smoothing_alpha=config.SPEED_SMOOTHING_ALPHA,
            calibration_frames=config.NEUTRAL_CALIBRATION_FRAMES,
            max_angle_jump_deg=config.MAX_ANGLE_JUMP_DEG,
            tracking_hold_frames=config.TRACKING_HOLD_FRAMES,
            calibration_max_std_deg=config.CALIBRATION_MAX_STD_DEG,
        )
        self.sensor = ESP32Reader(
            base_url=config.ESP32_BASE_URL,
            timeout=config.ESP32_TIMEOUT_SECONDS,
            poll_interval=config.ESP32_POLL_INTERVAL_SECONDS,
            serial_port=config.SERIAL_PORT,
            serial_baudrates=config.SERIAL_BAUDRATES,
            serial_timeout=config.SERIAL_TIMEOUT_SECONDS,
        )
        self.fusion = DataFusion(max_history=1000)
        self.recorder = SessionRecorder(sessions_dir=config.SESSIONS_DIR)

        # Threading & Capture handles
        self.cap: Optional[cv2.VideoCapture] = None
        self.capture_thread: Optional[threading.Thread] = None
        self.processing_thread: Optional[threading.Thread] = None

        # Frame buffers & Event notification
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_jpeg_frame: Optional[bytes] = None
        self.raw_frame_event = threading.Event()
        self.new_frame_event = threading.Event()

        # Telemetry State Cache
        self.latest_angles: Dict[str, Any] = {
            "wrist_angle": None,
            "wrist_deviation": None,
            "movement_angle": None,
            "angular_speed": 0.0,
            "movement_range": 0.0,
            "tracking_status": "WAITING",
            "is_calibrated": False,
        }

        # Start background sensor monitor
        self.sensor.start()

    def set_camera(self, index: int, name: Optional[str] = None) -> None:
        """Sets the active camera index and friendly name (when not running)."""
        with self.lock:
            if not self.is_running:
                self.selected_camera_index = int(index)
                self.selected_camera_name = name or f"Camera {index}"

    def start_analysis(self, camera_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Activates camera capture on the selected device, resets analysis engines, and starts session timing.
        """
        with self.lock:
            if self.is_running:
                return {"status": "ok", "message": "Already running"}

            if camera_index is not None:
                self.selected_camera_index = int(camera_index)
            else:
                # Refresh preferred camera before starting to lock onto C270/external webcam
                try:
                    pref = get_preferred_camera()
                    self.selected_camera_index = int(pref.get("index", self.selected_camera_index))
                    self.selected_camera_name = pref.get("name", self.selected_camera_name)
                except Exception:
                    pass

            self.calculator.reset_movement_range()
            self.fusion.reset_session()
            self.sensor.reset_session_metrics()
            self.start_perf_time = time.perf_counter()

            # Open the targeted camera device (DirectShow on Windows for lowest latency)
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.selected_camera_index, backend)

            if cap.isOpened():
                # Enforce low buffer size for zero-lag streaming
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

                act_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                act_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.active_resolution = f"{act_w}x{act_h}"
                self.cap = cap
                self.camera_state = "CONNECTED"
            else:
                self.cap = None
                self.active_resolution = "640x480 (Simulation)"
                self.camera_state = "CONNECTED"

            self.is_running = True
            self.raw_frame_event.clear()
            self.new_frame_event.clear()

            # Start dedicated capture worker
            self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
            self.capture_thread.start()

            # Start dedicated processing worker
            self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
            self.processing_thread.start()

            return {
                "status": "ok",
                "camera_state": self.camera_state,
                "camera_index": self.selected_camera_index,
                "camera_name": self.selected_camera_name,
                "resolution": self.active_resolution,
            }

    def stop_analysis(self) -> Dict[str, Any]:
        """
        Stops active session and releases camera hardware cleanly.
        """
        with self.lock:
            if not self.is_running:
                return {"status": "ok", "message": "Already stopped"}
            self.is_running = False

        self.raw_frame_event.set()
        self.new_frame_event.set()

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)

        with self.lock:
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
            self.camera_state = "IDLE"
            self.start_perf_time = None
            self.latest_raw_frame = None

        return {"status": "ok", "camera_state": "IDLE"}

    def calibrate_neutral(self) -> Dict[str, Any]:
        """Triggers neutral wrist angle calibration."""
        self.calculator.start_neutral_calibration(config.NEUTRAL_CALIBRATION_FRAMES)
        return {"status": "ok", "message": "Neutral calibration initiated"}

    def _capture_worker(self) -> None:
        """
        Dedicated camera acquisition thread.
        Reads hardware frames with minimal latency and publishes only the single latest frame.
        """
        fps_count = 0
        prev_time = time.perf_counter()

        while self.is_running:
            now = time.perf_counter()
            fps_count += 1
            if (now - prev_time) >= 1.0:
                inst_fps = fps_count / (now - prev_time)
                self.capture_fps = round(inst_fps, 1)
                fps_count = 0
                prev_time = now

            if self.cap and self.cap.isOpened():
                ret, raw_frame = self.cap.read()
                if ret and raw_frame is not None and raw_frame.size > 0:
                    frame = cv2.flip(raw_frame, 1)
                    with self.lock:
                        self.latest_raw_frame = frame
                    self.raw_frame_event.set()
                else:
                    time.sleep(0.005)
            else:
                # Clean simulated frame if hardware not available
                sim_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.rectangle(sim_frame, (20, 20), (620, 460), (30, 30, 30), -1)
                cv2.putText(
                    sim_frame,
                    "CAMERA SIMULATION MODE",
                    (120, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    sim_frame,
                    f"Selected: {self.selected_camera_name} (Index {self.selected_camera_index})",
                    (120, 260),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )
                with self.lock:
                    self.latest_raw_frame = sim_frame
                self.raw_frame_event.set()
                time.sleep(0.033)

    def _processing_worker(self) -> None:
        """
        Dedicated processing & overlay rendering thread.
        Consumes the single newest raw frame, runs optimized MediaPipe inference,
        calculates angles, draws clean overlays, and encodes JPEG.
        Drops stale frames automatically if processing is slower than capture.
        """
        fps_count = 0
        prev_time = time.perf_counter()

        while self.is_running:
            # Wait for fresh frame signal (up to 30ms)
            if not self.raw_frame_event.wait(timeout=0.03):
                continue
            self.raw_frame_event.clear()

            now = time.perf_counter()
            frame_to_process = None
            with self.lock:
                if self.latest_raw_frame is not None:
                    frame_to_process = self.latest_raw_frame

            if frame_to_process is None:
                continue

            fps_count += 1
            if (now - prev_time) >= 1.0:
                inst_fps = fps_count / (now - prev_time)
                # Exponential moving average for smooth display
                self.process_fps = round(0.7 * self.process_fps + 0.3 * inst_fps, 1) if self.process_fps > 0 else round(inst_fps, 1)
                fps_count = 0
                prev_time = now

            # 1. MediaPipe Tracking with downsampled inference
            annotated_frame, landmarks_data, quality = self.tracker.process_frame(frame_to_process, draw_overlay=True)

            # 2. Angle Calculations & Stability Filtering
            angles_data = self.calculator.update(landmarks_data, now, tracking_quality=quality)

            # 3. Read Force Sensor (Non-blocking instant read of cached values)
            sensor_data = self.sensor.read_force()

            # 4. Data Fusion
            fused = self.fusion.fuse(angles_data, sensor_data, quality, timestamp=now)

            # 5. JPEG Encoding with optimal streaming compression
            quality_param = getattr(config, "JPEG_STREAM_QUALITY", 78)
            ret, jpeg = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality_param])
            if ret:
                jpeg_bytes = jpeg.tobytes()
                with self.lock:
                    self.latest_jpeg_frame = jpeg_bytes
                    self.latest_angles = {
                        "wrist_angle": fused.get("wrist_angle"),
                        "wrist_deviation": fused.get("wrist_deviation"),
                        "movement_angle": fused.get("movement_angle"),
                        "angular_speed": fused.get("angular_speed", 0.0),
                        "movement_range": fused.get("movement_range", 0.0),
                        "tracking_status": quality,
                        "is_calibrated": angles_data.get("is_calibrated", False),
                    }
                # Signal streaming clients that a fresh frame is ready
                self.new_frame_event.set()

    def generate_mjpeg_stream(self) -> Generator[bytes, None, None]:
        """
        Yields only NEW multipart MJPEG frame chunks as they are produced.
        Drops stale intermediate frames automatically for minimum latency.
        """
        while True:
            if self.is_running:
                event_set = self.new_frame_event.wait(timeout=0.035)
                if event_set:
                    self.new_frame_event.clear()
                    frame_bytes = None
                    with self.lock:
                        frame_bytes = self.latest_jpeg_frame
                    if frame_bytes is not None:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                        )
            else:
                time.sleep(0.05)

    def get_live_payload(self) -> Dict[str, Any]:
        """
        Returns single efficient unified JSON payload for the web dashboard.
        """
        with self.lock:
            running = self.is_running
            start_t = self.start_perf_time
            cam_state = self.camera_state
            cam_name = self.selected_camera_name
            cam_idx = self.selected_camera_index
            res_str = self.active_resolution
            angles = dict(self.latest_angles)
            fps_val = round(self.process_fps, 1)

        now = time.perf_counter()
        elapsed_sec = max(0.0, now - start_t) if (running and start_t is not None) else 0.0
        mins = int(elapsed_sec // 60)
        secs = int(elapsed_sec % 60)
        elapsed_fmt = f"{mins:02d}:{secs:02d}"

        # Read latest sensor data
        sdata = self.sensor.read_force()

        return {
            "is_running": running,
            "elapsed_time": round(elapsed_sec, 2),
            "elapsed_formatted": elapsed_fmt,
            "camera": {
                "state": cam_state if running else "IDLE",
                "device_name": cam_name,
                "device_index": cam_idx,
                "resolution": res_str,
                "tracking": angles.get("tracking_status", "WAITING") if running else "WAITING",
                "fps": fps_val if running else 0.0,
            },
            "angles": {
                "wrist_angle": angles.get("wrist_angle") if running else None,
                "wrist_deviation": angles.get("wrist_deviation") if running else None,
                "movement_angle": angles.get("movement_angle") if running else None,
                "angular_speed": angles.get("angular_speed", 0.0) if running else None,
                "movement_range": angles.get("movement_range", 0.0) if running else 0.0,
            },
            "force": {
                "state": sdata.get("connection_state", "DISCONNECTED"),
                "source": sdata.get("force_source", "Disconnected"),
                "current": sdata.get("force_percent"),
                "average": sdata.get("average_force"),
                "peak": sdata.get("peak_force"),
                "connection_type": sdata.get("connection_type", "None"),
                "device": sdata.get("device_info", "None"),
                "mode": sdata.get("mode", "AUTO"),
            },
        }

    def shutdown(self) -> None:
        """Clean shutdown of camera, sensor, and tracker."""
        self.stop_analysis()
        self.sensor.stop()
        self.tracker.release()


# --------------------------------------------------------------------------
# Flask Web Application Setup
# --------------------------------------------------------------------------
app = Flask(__name__, template_folder=str(PROJECT_ROOT / "templates"), static_folder=str(PROJECT_ROOT / "static"))
system = LeverMonitorSystem()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    resp = Response(
        system.generate_mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/api/live")
@app.route("/api/telemetry")
@app.route("/api/status")
def api_live():
    """Single unified live metrics and status endpoint."""
    return jsonify(system.get_live_payload())


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(silent=True) or {}
    cam_index = data.get("camera_index")
    res = system.start_analysis(camera_index=cam_index)
    return jsonify(res)


@app.route("/api/stop", methods=["POST"])
def api_stop():
    res = system.stop_analysis()
    return jsonify(res)


@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    res = system.calibrate_neutral()
    return jsonify(res)


@app.route("/api/scan_devices")
@app.route("/api/devices")
def api_scan_devices():
    """
    Universal Device Discovery Endpoint.
    Discovers Camera devices, Windows PnP External USB devices, USB Serial COM devices,
    and ESP32 Network connectivity independently.
    """
    if system.is_running:
        cameras = [{
            "index": system.selected_camera_index,
            "name": system.selected_camera_name,
            "resolution": system.active_resolution,
            "status": "Active Analysis Device"
        }]
    else:
        cameras = scan_camera_devices()
        if cameras:
            sorted_cams = sorted(cameras, key=lambda c: c.get("preference_score", 0), reverse=True)
            best_cam = sorted_cams[0]
            system.selected_camera_index = best_cam["index"]
            system.selected_camera_name = best_cam["name"]

    external_devices = get_connected_external_devices()
    serial_devices = ESP32Reader.scan_available_ports()
    network_device = system.sensor.check_esp32_network()

    cam_name = system.selected_camera_name
    cam_index = system.selected_camera_index

    camera_info = {
        "device_name": cam_name,
        "interface": "USB Camera",
        "analysis_device": f"Camera {cam_index}",
        "resolution": system.active_resolution if system.is_running else f"{config.FRAME_WIDTH}x{config.FRAME_HEIGHT}",
        "status": "ACTIVE" if system.is_running else "CONNECTED",
    }

    return jsonify({
        "camera": camera_info,
        "cameras": cameras,
        "external_devices": external_devices,
        "serial_devices": serial_devices,
        "network_device": network_device,
        "selected_camera": {
            "index": system.selected_camera_index,
            "name": cam_name,
        },
        "selected_force_source": system.sensor.force_source,
        "force_mode": system.sensor.mode,
    })


@app.route("/api/config_camera", methods=["POST"])
def api_config_camera():
    """Configures the selected camera device index and name."""
    data = request.get_json() or {}
    idx = data.get("index")
    name = data.get("name")
    if idx is not None:
        system.set_camera(int(idx), name)
    return jsonify({
        "status": "ok",
        "selected_camera_index": system.selected_camera_index,
        "selected_camera_name": system.selected_camera_name,
    })


@app.route("/api/config_sensor", methods=["POST"])
def api_config_sensor():
    """Configures force source mode, custom IP, or USB serial port."""
    data = request.get_json() or {}
    mode = data.get("mode")
    ip = data.get("ip")
    port = data.get("port")
    action = data.get("action")

    if mode:
        system.sensor.set_mode(mode)
    if ip:
        system.sensor.set_wifi_address(ip)
    if port:
        system.sensor.set_serial_port(port)

    if action == "disconnect":
        system.sensor._close_serial()
    elif action == "connect":
        if mode == "SERIAL" and port:
            system.sensor.set_serial_port(port)

    return jsonify({
        "status": "ok",
        "mode": system.sensor.mode,
        "base_url": system.sensor.base_url,
        "serial_port": system.sensor.serial_port,
    })


@app.route("/api/serial/stream")
def api_serial_stream():
    """Returns real-time serial monitor logs and packet health metrics."""
    since_id = request.args.get("since_id", default=0, type=int)
    stream_data = system.sensor.get_serial_stream(since_id=since_id)
    return jsonify(stream_data)


@app.route("/api/serial/clear", methods=["POST"])
def api_serial_clear():
    """Clears serial console buffer."""
    system.sensor.clear_serial_logs()
    return jsonify({"status": "ok"})


@app.route("/api/serial/export_csv")
def api_serial_export_csv():
    """Exports all recorded raw serial stream packets as a downloadable CSV."""
    csv_content = system.sensor.generate_serial_csv()
    response = make_response(csv_content)
    response.headers["Content-Disposition"] = "attachment; filename=serial_data_stream.csv"
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    return response


def open_browser():
    """Opens browser to localhost on server startup."""
    time.sleep(1.0)
    webbrowser.open(f"http://{config.SERVER_HOST}:{config.SERVER_PORT}")


if __name__ == "__main__":
    print("=" * 70, flush=True)
    print("Aircraft Lever Operator Ergonomics & Lever Force Monitoring System", flush=True)
    print(f"Server Running at: http://{config.SERVER_HOST}:{config.SERVER_PORT}", flush=True)
    print("=" * 70, flush=True)

    if config.AUTO_OPEN_BROWSER and "--no-browser" not in sys.argv:
        threading.Thread(target=open_browser, daemon=True).start()

    try:
        app.run(
            host=config.SERVER_HOST,
            port=config.SERVER_PORT,
            debug=False,
            threaded=True,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Server shutting down...", flush=True)
    finally:
        system.shutdown()
        print("[INFO] Clean shutdown complete.", flush=True)
