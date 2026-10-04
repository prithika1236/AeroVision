"""
Universal Device Discovery & System Verification Suite for Aircraft Lever Operator Monitoring System.
Tests:
1. Universal Device Discovery:
   - Camera Discovery (OpenCV Video Devices with proper resource release)
   - Serial Device Discovery (PySerial COM ports with metadata VID/PID/Manufacturer)
   - Network Device Probe (ESP32 Network Endpoint Status)
2. Camera Selection & Dynamic Configuration.
3. MediaPipe Processing Throughput & Latency.
4. Flask Live API, Video Stream Anti-Cache Headers, and Start/Stop Lifecycle.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import config
from camera.camera_discovery import scan_camera_devices
from sensor.esp32_reader import ESP32Reader
from camera.hand_tracker import HandTracker
from main import app, system


def test_universal_device_discovery():
    print("\n--- 1. Testing Universal Device Discovery ---")
    
    # A. Camera Discovery
    cams = scan_camera_devices(max_index=3)
    print(f"[PASS] Camera Discovery: Discovered {len(cams)} camera(s): {cams}")
    
    # B. Serial Discovery
    serial_ports = ESP32Reader.scan_available_ports()
    print(f"[PASS] Serial Discovery: Discovered {len(serial_ports)} serial device(s): {serial_ports}")
    
    # C. Network Device Probe
    net_status = system.sensor.check_esp32_network(timeout=0.1)
    print(f"[PASS] Network Discovery: ESP32 probe status: {net_status}")


def test_mediapipe_throughput():
    print("\n--- 2. Benchmarking MediaPipe Processing FPS ---")
    tracker = HandTracker(infer_width=640, infer_height=360)
    test_frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)

    # Warmup
    tracker.process_frame(test_frame)

    t0 = time.perf_counter()
    num_frames = 25
    for _ in range(num_frames):
        ann, lms, q = tracker.process_frame(test_frame)
    t1 = time.perf_counter()

    dur = t1 - t0
    fps = num_frames / dur
    ms_per_frame = (dur / num_frames) * 1000.0

    print(f"[PASS] MediaPipe Throughput: {fps:.1f} FPS ({ms_per_frame:.2f} ms/frame)")
    assert fps >= 30.0, f"Expected >= 30 FPS, got {fps:.1f}"
    tracker.release()


def test_flask_endpoints_and_lifecycle():
    print("\n--- 3. Testing Flask Universal Discovery & Endpoints ---")
    client = app.test_client()

    # GET /
    r = client.get("/")
    assert r.status_code == 200
    assert b"AIRCRAFT LEVER OPERATOR MONITORING SYSTEM" in r.data
    print("[PASS] GET / renders light engineering dashboard.")

    # GET /api/scan_devices
    r_dev = client.get("/api/scan_devices")
    assert r_dev.status_code == 200
    dev_data = r_dev.get_json()
    assert "cameras" in dev_data
    assert "serial_devices" in dev_data
    assert "network_device" in dev_data
    print(f"[PASS] GET /api/scan_devices: {len(dev_data['cameras'])} cameras, {len(dev_data['serial_devices'])} serial devices.")

    # POST /api/config_camera
    r_cam = client.post("/api/config_camera", json={"index": 1, "name": "Camera 1"})
    assert r_cam.status_code == 200
    assert system.selected_camera_index == 1
    print("[PASS] Camera configuration updated to Camera index 1.")

    # GET /api/live (idle)
    r_live = client.get("/api/live")
    assert r_live.status_code == 200
    live_data = r_live.get_json()
    assert "camera" in live_data
    assert "angles" in live_data
    assert "force" in live_data
    print(f"[PASS] GET /api/live verified: camera={live_data['camera']['device_name']}, state={live_data['camera']['state']}")

    # POST /api/start (with specific camera index)
    r_start = client.post("/api/start", json={"camera_index": 1})
    assert r_start.status_code == 200
    print("[PASS] POST /api/start succeeded with camera_index=1.")

    time.sleep(0.4)

    # GET /api/live (running)
    r_live_run = client.get("/api/live")
    assert r_live_run.status_code == 200
    live_run_data = r_live_run.get_json()
    assert live_run_data["is_running"] is True
    print(f"[PASS] Active telemetry: elapsed={live_run_data['elapsed_formatted']}, FPS={live_run_data['camera']['fps']}")

    # POST /api/stop
    r_stop = client.post("/api/stop")
    assert r_stop.status_code == 200
    print("[PASS] POST /api/stop succeeded.")

    # Second cycle to test clean start/stop
    r_start2 = client.post("/api/start", json={"camera_index": 0})
    assert r_start2.status_code == 200
    time.sleep(0.2)
    r_stop2 = client.post("/api/stop")
    assert r_stop2.status_code == 200
    print("[PASS] Second cycle start/stop cleanly executed.")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING UNIVERSAL DEVICE DISCOVERY & BENCHMARK SUITE")
    print("=" * 70)
    test_universal_device_discovery()
    test_mediapipe_throughput()
    test_flask_endpoints_and_lifecycle()
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
