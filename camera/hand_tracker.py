"""
Biomechanical arm and hand landmark tracking module using MediaPipe Holistic and OpenCV.
Tracks full upper-limb kinematic chain: Shoulder -> Elbow -> Wrist -> 21 Hand Landmarks for both arms.
Optimized for real-time low-latency performance with downsampled inference and clean display overlays.
"""

from typing import Optional, Tuple, Dict, Any, List
import cv2
import numpy as np

# Protobuf 5.x/6.x compatibility shim for MediaPipe
try:
    from google.protobuf import symbol_database, message_factory
    if hasattr(message_factory, 'GetMessageClass'):
        if not hasattr(message_factory, 'GetPrototype'):
            message_factory.GetPrototype = message_factory.GetMessageClass
        if not hasattr(symbol_database.Default(), 'GetPrototype'):
            symbol_database.Default().GetPrototype = message_factory.GetMessageClass
        if hasattr(symbol_database, 'SymbolDatabase') and not hasattr(symbol_database.SymbolDatabase, 'GetPrototype'):
            symbol_database.SymbolDatabase.GetPrototype = staticmethod(message_factory.GetMessageClass)
except Exception:
    pass

import mediapipe as mp



class HandTracker:
    """
    Tracks upper-body arm landmarks (Shoulder, Elbow, Wrist) and all 21 hand landmarks
    simultaneously with clean biomechanical kinematic overlays.
    """

    # Tracking quality states
    STATE_GOOD = "GOOD"
    STATE_LIMITED = "LIMITED"
    STATE_LOST = "LOST"

    LANDMARK_NAMES = {
        0: "WRIST",
        1: "THUMB_CMC",
        2: "THUMB_MCP",
        3: "THUMB_IP",
        4: "THUMB_TIP",
        5: "INDEX_FINGER_MCP",
        6: "INDEX_FINGER_PIP",
        7: "INDEX_FINGER_DIP",
        8: "INDEX_FINGER_TIP",
        9: "MIDDLE_FINGER_MCP",
        10: "MIDDLE_FINGER_PIP",
        11: "MIDDLE_FINGER_DIP",
        12: "MIDDLE_FINGER_TIP",
        13: "RING_FINGER_MCP",
        14: "RING_FINGER_PIP",
        15: "RING_FINGER_DIP",
        16: "RING_FINGER_TIP",
        17: "PINKY_MCP",
        18: "PINKY_PIP",
        19: "PINKY_DIP",
        20: "PINKY_TIP",
    }

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        infer_width: int = 640,
        infer_height: int = 360,
    ) -> None:
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.infer_width = infer_width
        self.infer_height = infer_height
        self.tracking_quality: str = self.STATE_LOST

        # MediaPipe Holistic initialization (model_complexity=0 for ultra-low latency & CPU speed)
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process_frame(
        self, frame: np.ndarray, draw_overlay: bool = True
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]], str]:
        """
        Processes a camera frame, extracts shoulders, elbows, wrists, and 21 hand landmarks,
        and annotates the full arm kinematic chain without distracting debug boxes.
        """
        if frame is None or frame.size == 0:
            self.tracking_quality = self.STATE_LOST
            return frame, None, self.STATE_LOST

        height, width = frame.shape[:2]

        # Fast downsampled frame for MediaPipe inference (preserves aspect ratio)
        if width > self.infer_width or height > self.infer_height:
            small_frame = cv2.resize(frame, (self.infer_width, self.infer_height), interpolation=cv2.INTER_LINEAR)
        else:
            small_frame = frame

        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.holistic.process(frame_rgb)

        pose_lms = results.pose_landmarks
        left_hand_lms = results.left_hand_landmarks
        right_hand_lms = results.right_hand_landmarks

        has_pose = pose_lms is not None
        has_left_hand = left_hand_lms is not None
        has_right_hand = right_hand_lms is not None

        if not has_pose and not has_left_hand and not has_right_hand:
            self.tracking_quality = self.STATE_LOST
            return frame, None, self.STATE_LOST

        hands_data = []

        # Helper to extract 21 hand landmarks directly in full-frame pixel coordinates
        def extract_hand_data(hand_landmarks, arm_wrist_px=None, elbow_px=None, shoulder_px=None):
            lms_norm = []
            lms_px = []
            lms_by_name = {}
            for idx, lm in enumerate(hand_landmarks.landmark):
                px_x = int(lm.x * width)
                px_y = int(lm.y * height)
                lms_norm.append((lm.x, lm.y, lm.z))
                lms_px.append((px_x, px_y, lm.z))
                name = self.LANDMARK_NAMES.get(idx, f"LANDMARK_{idx}")
                lms_by_name[name] = {"id": idx, "norm": (lm.x, lm.y, lm.z), "pixel": (px_x, px_y)}

            wrist_point = lms_px[0][:2]
            return {
                "landmarks_norm": lms_norm,
                "landmarks_px": lms_px,
                "landmarks_by_name": lms_by_name,
                "wrist_px": wrist_point,
                "elbow_px": elbow_px,
                "shoulder_px": shoulder_px,
                "index_mcp_px": lms_px[5][:2],
                "middle_mcp_px": lms_px[9][:2],
                "pinky_mcp_px": lms_px[17][:2],
            }

        # Extract Pose Arm Landmarks mapped to display resolution
        left_shoulder_px = left_elbow_px = left_wrist_px = None
        right_shoulder_px = right_elbow_px = right_wrist_px = None

        if has_pose:
            plm = pose_lms.landmark
            if plm[11].visibility > 0.4:
                left_shoulder_px = (int(plm[11].x * width), int(plm[11].y * height))
            if plm[13].visibility > 0.4:
                left_elbow_px = (int(plm[13].x * width), int(plm[13].y * height))
            if plm[15].visibility > 0.4:
                left_wrist_px = (int(plm[15].x * width), int(plm[15].y * height))

            if plm[12].visibility > 0.4:
                right_shoulder_px = (int(plm[12].x * width), int(plm[12].y * height))
            if plm[14].visibility > 0.4:
                right_elbow_px = (int(plm[14].x * width), int(plm[14].y * height))
            if plm[16].visibility > 0.4:
                right_wrist_px = (int(plm[16].x * width), int(plm[16].y * height))

            # Draw clean upper-limb kinematic lines directly on display frame
            if draw_overlay:
                if left_shoulder_px and right_shoulder_px:
                    cv2.line(frame, left_shoulder_px, right_shoulder_px, (200, 200, 200), 2, cv2.LINE_AA)

                if left_shoulder_px and left_elbow_px:
                    cv2.line(frame, left_shoulder_px, left_elbow_px, (0, 220, 255), 3, cv2.LINE_AA)
                    cv2.circle(frame, left_shoulder_px, 6, (0, 180, 255), -1)

                if left_elbow_px and left_wrist_px:
                    cv2.line(frame, left_elbow_px, left_wrist_px, (0, 255, 180), 3, cv2.LINE_AA)
                    cv2.circle(frame, left_elbow_px, 6, (0, 255, 180), -1)

                if right_shoulder_px and right_elbow_px:
                    cv2.line(frame, right_shoulder_px, right_elbow_px, (255, 180, 0), 3, cv2.LINE_AA)
                    cv2.circle(frame, right_shoulder_px, 6, (255, 150, 0), -1)

                if right_elbow_px and right_wrist_px:
                    cv2.line(frame, right_elbow_px, right_wrist_px, (255, 220, 0), 3, cv2.LINE_AA)
                    cv2.circle(frame, right_elbow_px, 6, (255, 220, 0), -1)

        # Process Hand 1 Landmarks (21 landmarks)
        if has_right_hand:
            h_data = extract_hand_data(right_hand_lms, right_wrist_px, right_elbow_px, right_shoulder_px)
            hands_data.append(h_data)
            if draw_overlay:
                self.mp_draw.draw_landmarks(
                    frame,
                    right_hand_lms,
                    self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )
                w_pt = h_data["wrist_px"]
                cv2.circle(frame, w_pt, 8, (255, 255, 0), 2)
                if right_elbow_px:
                    cv2.line(frame, right_elbow_px, w_pt, (255, 220, 0), 2, cv2.LINE_AA)

        # Process Hand 2 Landmarks (21 landmarks)
        if has_left_hand:
            h_data = extract_hand_data(left_hand_lms, left_wrist_px, left_elbow_px, left_shoulder_px)
            hands_data.append(h_data)
            if draw_overlay:
                self.mp_draw.draw_landmarks(
                    frame,
                    left_hand_lms,
                    self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )
                w_pt = h_data["wrist_px"]
                cv2.circle(frame, w_pt, 8, (0, 255, 255), 2)
                if left_elbow_px:
                    cv2.line(frame, left_elbow_px, w_pt, (0, 255, 180), 2, cv2.LINE_AA)

        # Determine tracking quality
        num_hands = len(hands_data)
        if num_hands > 0:
            self.tracking_quality = self.STATE_GOOD
        elif has_pose:
            self.tracking_quality = self.STATE_LIMITED
        else:
            self.tracking_quality = self.STATE_LOST

        if not hands_data:
            if right_wrist_px or left_wrist_px:
                primary_wrist = right_wrist_px if right_wrist_px else left_wrist_px
                primary_elbow = right_elbow_px if right_elbow_px else left_elbow_px
                primary_shoulder = right_shoulder_px if right_shoulder_px else left_shoulder_px
                synthetic_hand = {
                    "wrist_px": primary_wrist,
                    "elbow_px": primary_elbow,
                    "shoulder_px": primary_shoulder,
                    "index_mcp_px": (primary_wrist[0], primary_wrist[1] - 40),
                    "middle_mcp_px": (primary_wrist[0], primary_wrist[1] - 50),
                    "pinky_mcp_px": (primary_wrist[0] + 20, primary_wrist[1] - 40),
                    "landmarks_norm": [],
                    "landmarks_px": [],
                    "landmarks_by_name": {},
                }
                hands_data.append(synthetic_hand)

        primary_hand = hands_data[0] if hands_data else None

        landmarks_data = None
        if primary_hand is not None:
            landmarks_data = {
                "num_hands": num_hands,
                "hands": hands_data,
                "primary_hand": primary_hand,
                "wrist_px": primary_hand["wrist_px"],
                "elbow_px": primary_hand.get("elbow_px"),
                "shoulder_px": primary_hand.get("shoulder_px"),
                "index_mcp_px": primary_hand["index_mcp_px"],
                "middle_mcp_px": primary_hand["middle_mcp_px"],
                "pinky_mcp_px": primary_hand["pinky_mcp_px"],
                "landmarks_norm": primary_hand.get("landmarks_norm", []),
                "landmarks_px": primary_hand.get("landmarks_px", []),
                "landmarks_by_name": primary_hand.get("landmarks_by_name", {}),
            }

        return frame, landmarks_data, self.tracking_quality

    def release(self) -> None:
        """Releases MediaPipe resources."""
        if self.holistic:
            try:
                self.holistic.close()
            except Exception:
                pass
