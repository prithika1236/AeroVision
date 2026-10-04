"""
Configuration parameters for the Aircraft Lever Operator Ergonomic and Fatigue Monitoring System.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SESSIONS_DIR = DATA_DIR / "sessions"

# Ensure directories exist
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Hardware & Camera Settings
# Logitech C270 720p HD webcam defaults
CAMERA_INDEX = 0
MAX_CAMERA_DISCOVERY_INDEX = 4          # Probe camera indices 0..4
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CAMERA_FPS = 30
JPEG_STREAM_QUALITY = 78                # Optimal balance between landmark clarity and stream latency

# MediaPipe Tracking Settings
MP_MAX_NUM_HANDS = 2
MP_MIN_DETECTION_CONFIDENCE = 0.6
MP_MIN_TRACKING_CONFIDENCE = 0.5

# Tracking Quality States
TRACKING_QUALITY_GOOD = "GOOD"
TRACKING_QUALITY_LIMITED = "LIMITED"
TRACKING_QUALITY_LOST = "LOST"

# Signal Smoothing & Stability Settings (Phase 4)
ANGLE_SMOOTHING_ALPHA = 0.35
SPEED_SMOOTHING_ALPHA = 0.30
MAX_ANGLE_JUMP_DEG = 40.0              # Maximum plausible angular change per frame (spike rejection)
TRACKING_HOLD_FRAMES = 12              # Frames to hold last valid measurement during temporary loss (~0.4s)
CALIBRATION_MAX_STD_DEG = 8.0          # Maximum standard deviation tolerated during neutral calibration

# ESP32-S3 Wi-Fi Force Sensor Settings
# ESP32 endpoint providing normalized force / grip data
ESP32_BASE_URL = "http://192.168.4.1"
ESP32_DATA_ENDPOINT = f"{ESP32_BASE_URL}/data"
ESP32_TIMEOUT_SECONDS = 0.5
ESP32_POLL_INTERVAL_SECONDS = 0.05  # 20 Hz polling rate

# USB / Serial Force Sensor Settings
SERIAL_PORT = None                  # None = auto-detect COM ports, or specify e.g. "COM3"
SERIAL_BAUDRATES = [115200, 9600]   # Supported serial baud rates
SERIAL_TIMEOUT_SECONDS = 0.3

# Web Server Settings
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5000
AUTO_OPEN_BROWSER = True

# Experiment & Calibration Settings
NEUTRAL_CALIBRATION_FRAMES = 60      # ~2 seconds at 30 FPS for robust calibration
BASELINE_COLLECTION_SECONDS = 15     # Initial baseline collection duration

# Repetition & Movement Detection Thresholds (Phase 7)
LEVER_MIN_ANGLE_SPAN_DEG = 15.0      # Minimum angular displacement to qualify as a valid stroke
LEVER_START_THRESHOLD_DEG = 8.0      # Angular threshold to trigger stroke initiation from rest
LEVER_RETURN_THRESHOLD_DEG = 6.0     # Angular proximity to rest baseline to complete cycle
LEVER_MIN_STROKE_DURATION_SEC = 0.4  # Minimum cycle time to avoid counting jitter
LEVER_MAX_STROKE_DURATION_SEC = 5.0  # Maximum cycle time for a single repetition stroke

# Baseline & Fatigue-Associated Deviation Thresholds (Phase 8)
BASELINE_REPETITIONS_REQUIRED = 3          # Initial valid repetitions to establish operator baseline
RECENT_REPETITIONS_WINDOW = 3              # Rolling window of recent repetitions evaluated against baseline
FATIGUE_DEVIATION_LOW_THRESHOLD = 0.15     # <15% composite deviation: LOW
FATIGUE_DEVIATION_MODERATE_THRESHOLD = 0.35# 15%-35%: MODERATE; >35%: HIGH

FATIGUE_LEVEL_LOW = "LOW"
FATIGUE_LEVEL_MODERATE = "MODERATE"
FATIGUE_LEVEL_HIGH = "HIGH"
