"""
Hand tracking module using MediaPipe Hands.
Supports single-hand tracking and dual-hand virtual steering wheel tracking,
automatic model resolution, coordinate smoothing (EMA), and single-hand fallback.
"""

import os
import sys
import math
import time
import urllib.request
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

logger = logging.getLogger("burnout.hand_tracking")

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
DEFAULT_MODEL_PATH = os.path.join(ASSET_DIR, "hand_landmarker.task")

try:
    import cv2
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe or OpenCV not found. Hand tracking running in fallback mode.")


def ensure_model_asset(target_path: str = DEFAULT_MODEL_PATH) -> bool:
    """Ensure the hand landmarker task model is available locally, downloading if missing."""
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000000:
        return True
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        logger.info(f"Downloading MediaPipe hand model to {target_path}...")
        urllib.request.urlretrieve(MODEL_URL, target_path)
        logger.info("Model download complete.")
        return True
    except Exception as e:
        logger.error(f"Failed to download MediaPipe model: {e}")
        return False


@dataclass
class HandData:
    is_detected: bool = False
    is_two_handed: bool = False
    is_fallback: bool = False

    # Primary steering hand
    landmarks_norm: List[Tuple[float, float, float]] = field(default_factory=list)
    landmarks_pixel: List[Tuple[int, int]] = field(default_factory=list)
    hand_center: Tuple[float, float] = (0.5, 0.5)
    hand_size: float = 0.20
    tilt_angle: float = 0.0          # degrees tilt (-90 to +90)
    fingers_extended: List[bool] = field(default_factory=lambda: [False]*5) # Thumb, Index, Middle, Ring, Pinky
    handedness: str = "Right"

    # Virtual steering wheel dual-hand metrics
    left_hand_center: Optional[Tuple[float, float]] = None
    right_hand_center: Optional[Tuple[float, float]] = None
    wheel_angle: float = 0.0         # degrees rotation of the virtual wheel
    second_landmarks_pixel: List[Tuple[int, int]] = field(default_factory=list)
    second_fingers_extended: List[bool] = field(default_factory=lambda: [False]*5)

    confidence: float = 0.0
    tracking_lost_time: float = 0.0


class HandTracker:
    """
    Tracks hand landmarks using MediaPipe with EMA smoothing and 2-hand wheel support.
    """

    def __init__(self,
                 two_handed_mode: bool = False,
                 min_detection_confidence: float = 0.60,
                 min_tracking_confidence: float = 0.55,
                 smoothing_factor: float = 0.45):
        self.two_handed_mode = two_handed_mode
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.smoothing_factor = smoothing_factor

        self.api_mode: str = "none"  # "tasks", "solutions", or "none"
        self.landmarker = None
        self._init_detector()

        # EMA smoothing buffers for primary hand
        self._smoothed_landmarks: Optional[np.ndarray] = None
        self._smoothed_center: Optional[Tuple[float, float]] = None
        self._smoothed_size: Optional[float] = None

        # Smoothing for two-hand wheel
        self._smoothed_second_landmarks: Optional[np.ndarray] = None
        self._smoothed_wheel_angle: Optional[float] = None

        self._last_detected_time = time.time()

    def set_two_handed_mode(self, enabled: bool) -> None:
        """Dynamically reinitialize detector when switching single / two-hand mode."""
        if self.two_handed_mode != enabled:
            self.two_handed_mode = enabled
            self.close()
            self._init_detector()
            logger.info(f"HandTracker two-handed mode updated to: {enabled}")

    def _init_detector(self) -> None:
        if not MEDIAPIPE_AVAILABLE:
            return

        max_hands = 2 if self.two_handed_mode else 1

        # 1. Modern MediaPipe Tasks API
        try:
            if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
                ensure_model_asset(DEFAULT_MODEL_PATH)
                if os.path.exists(DEFAULT_MODEL_PATH):
                    base_opts = mp.tasks.BaseOptions(model_asset_path=DEFAULT_MODEL_PATH)
                    options = mp.tasks.vision.HandLandmarkerOptions(
                        base_options=base_opts,
                        num_hands=max_hands,
                        min_hand_detection_confidence=self.min_detection_confidence,
                        min_hand_presence_confidence=self.min_tracking_confidence
                    )
                    self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
                    self.api_mode = "tasks"
                    logger.info(f"MediaPipe Tasks initialized (max_hands={max_hands}).")
                    return
        except Exception as e:
            logger.warning(f"Could not initialize MediaPipe Tasks: {e}. Falling back to solutions...")

        # 2. Legacy MediaPipe Solutions API
        try:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self.landmarker = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=max_hands,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                    model_complexity=1
                )
                self.api_mode = "solutions"
                logger.info(f"MediaPipe Solutions initialized (max_hands={max_hands}).")
                return
        except Exception as e:
            logger.warning(f"Could not initialize MediaPipe Solutions: {e}")

        logger.error("No compatible MediaPipe Hand tracking backend available.")
        self.api_mode = "none"

    def process_frame(self, frame_bgr: np.ndarray, mirror: bool = True) -> Tuple[np.ndarray, HandData]:
        """
        Process camera frame and detect single or dual hands.
        """
        if frame_bgr is None:
            return frame_bgr, HandData(is_detected=False)

        if mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)

        h, w, c = frame_bgr.shape
        hand_data = HandData()

        if self.api_mode == "none" or self.landmarker is None:
            return frame_bgr, hand_data

        detected_hands_raw: List[np.ndarray] = []
        detected_handedness: List[str] = []

        try:
            if self.api_mode == "tasks":
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                results = self.landmarker.detect(mp_image)

                if results.hand_landmarks:
                    for i, lm_list in enumerate(results.hand_landmarks):
                        arr = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)
                        detected_hands_raw.append(arr)
                        h_name = "Right"
                        if results.handedness and i < len(results.handedness) and results.handedness[i]:
                            h_name = results.handedness[i][0].display_name or results.handedness[i][0].category_name or "Right"
                        detected_handedness.append(h_name)

            elif self.api_mode == "solutions":
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = self.landmarker.process(frame_rgb)
                frame_rgb.flags.writeable = True

                if results.multi_hand_landmarks:
                    for i, lm_list in enumerate(results.multi_hand_landmarks):
                        arr = np.array([[lm.x, lm.y, lm.z] for lm in lm_list.landmark], dtype=np.float32)
                        detected_hands_raw.append(arr)
                        h_name = "Right"
                        if results.multi_handedness and i < len(results.multi_handedness):
                            h_name = results.multi_handedness[i].classification[0].label
                        detected_handedness.append(h_name)

        except Exception as e:
            logger.debug(f"Hand detection error: {e}")
            detected_hands_raw.clear()

        # Check if no hands detected
        if not detected_hands_raw:
            self._smoothed_landmarks = None
            self._smoothed_center = None
            self._smoothed_size = None
            self._smoothed_second_landmarks = None
            self._smoothed_wheel_angle = None
            hand_data.is_detected = False
            hand_data.tracking_lost_time = time.time()
            return frame_bgr, hand_data

        hand_data.is_detected = True
        self._last_detected_time = time.time()
        alpha = self.smoothing_factor

        # -------------------------------------------------------------
        # TWO-HAND VIRTUAL STEERING WHEEL LOGIC
        # -------------------------------------------------------------
        if self.two_handed_mode and len(detected_hands_raw) >= 2:
            hand_data.is_two_handed = True
            hand_data.is_fallback = False

            # Sort hands horizontally: left hand on screen left, right hand on screen right
            # Hand center X of first vs second
            h0_cx = float(np.mean(detected_hands_raw[0][[0, 5, 9, 17], 0]))
            h1_cx = float(np.mean(detected_hands_raw[1][[0, 5, 9, 17], 0]))

            if h0_cx < h1_cx:
                left_raw, right_raw = detected_hands_raw[0], detected_hands_raw[1]
            else:
                left_raw, right_raw = detected_hands_raw[1], detected_hands_raw[0]

            # Smooth primary (left) and secondary (right)
            if self._smoothed_landmarks is None:
                self._smoothed_landmarks = left_raw.copy()
            else:
                self._smoothed_landmarks = alpha * left_raw + (1.0 - alpha) * self._smoothed_landmarks

            if self._smoothed_second_landmarks is None:
                self._smoothed_second_landmarks = right_raw.copy()
            else:
                self._smoothed_second_landmarks = alpha * right_raw + (1.0 - alpha) * self._smoothed_second_landmarks

            # Compute centroids for both hands
            left_cx = float(np.mean(self._smoothed_landmarks[[0, 5, 9, 17], 0]))
            left_cy = float(np.mean(self._smoothed_landmarks[[0, 5, 9, 17], 1]))
            right_cx = float(np.mean(self._smoothed_second_landmarks[[0, 5, 9, 17], 0]))
            right_cy = float(np.mean(self._smoothed_second_landmarks[[0, 5, 9, 17], 1]))

            hand_data.left_hand_center = (left_cx, left_cy)
            hand_data.right_hand_center = (right_cx, right_cy)
            hand_data.hand_center = ((left_cx + right_cx) / 2.0, (left_cy + right_cy) / 2.0)

            # Calculate steering-wheel rotation angle
            # dx is horizontal separation (right - left), dy is vertical separation (right - left)
            # In image coords, Y increases downward.
            # When wheel is turned RIGHT: right hand moves down (dy > 0), left hand moves up.
            # When wheel is turned LEFT: right hand moves up (dy < 0), left hand moves down.
            dx = right_cx - left_cx
            dy = right_cy - left_cy
            raw_wheel_angle = math.degrees(math.atan2(dy, max(0.05, dx)))
            raw_wheel_angle = max(-75.0, min(75.0, raw_wheel_angle))

            if self._smoothed_wheel_angle is None:
                self._smoothed_wheel_angle = raw_wheel_angle
            else:
                self._smoothed_wheel_angle = alpha * raw_wheel_angle + (1.0 - alpha) * self._smoothed_wheel_angle

            hand_data.wheel_angle = self._smoothed_wheel_angle
            hand_data.tilt_angle = self._smoothed_wheel_angle

            # Extract pixel landmarks and finger extensions for both hands
            hand_data.landmarks_pixel = [(int(x * w), int(y * h)) for x, y in self._smoothed_landmarks[:, :2]]
            hand_data.second_landmarks_pixel = [(int(x * w), int(y * h)) for x, y in self._smoothed_second_landmarks[:, :2]]

            hand_data.fingers_extended = self._detect_extended_fingers(self._smoothed_landmarks)
            hand_data.second_fingers_extended = self._detect_extended_fingers(self._smoothed_second_landmarks)

            # Hand size (average size of both hands)
            sz1 = float(np.linalg.norm(self._smoothed_landmarks[0, :2] - self._smoothed_landmarks[9, :2]))
            sz2 = float(np.linalg.norm(self._smoothed_second_landmarks[0, :2] - self._smoothed_second_landmarks[9, :2]))
            hand_data.hand_size = (sz1 + sz2) / 2.0

            return frame_bgr, hand_data

        # -------------------------------------------------------------
        # SINGLE HAND (OR FALLBACK FROM TWO-HAND)
        # -------------------------------------------------------------
        raw_landmarks = detected_hands_raw[0]
        hand_data.handedness = detected_handedness[0] if detected_handedness else "Right"

        if self.two_handed_mode:
            # Fallback engaged because only 1 hand is visible
            hand_data.is_two_handed = False
            hand_data.is_fallback = True

        if self._smoothed_landmarks is None:
            self._smoothed_landmarks = raw_landmarks.copy()
        else:
            self._smoothed_landmarks = alpha * raw_landmarks + (1.0 - alpha) * self._smoothed_landmarks

        hand_data.landmarks_pixel = [(int(x * w), int(y * h)) for x, y in self._smoothed_landmarks[:, :2]]
        hand_data.landmarks_norm = [(float(x), float(y), float(z)) for x, y, z in self._smoothed_landmarks]

        # Centroid
        pts = self._smoothed_landmarks[[0, 5, 9, 17], :2]
        raw_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        if self._smoothed_center is None:
            self._smoothed_center = raw_center
        else:
            self._smoothed_center = (
                alpha * raw_center[0] + (1.0 - alpha) * self._smoothed_center[0],
                alpha * raw_center[1] + (1.0 - alpha) * self._smoothed_center[1]
            )
        hand_data.hand_center = self._smoothed_center

        # Size
        wrist = self._smoothed_landmarks[0, :2]
        middle_mcp = self._smoothed_landmarks[9, :2]
        raw_size = float(np.linalg.norm(wrist - middle_mcp))
        if self._smoothed_size is None:
            self._smoothed_size = raw_size
        else:
            self._smoothed_size = alpha * raw_size + (1.0 - alpha) * self._smoothed_size
        hand_data.hand_size = self._smoothed_size

        # Tilt angle deviation from vertical (0 deg = straight up)
        dx = middle_mcp[0] - wrist[0]
        dy = middle_mcp[1] - wrist[1]
        raw_angle = math.degrees(math.atan2(dx, -dy))
        hand_data.tilt_angle = max(-90.0, min(90.0, raw_angle))
        hand_data.wheel_angle = hand_data.tilt_angle  # fallback wheel angle

        hand_data.fingers_extended = self._detect_extended_fingers(self._smoothed_landmarks)

        return frame_bgr, hand_data

    def _detect_extended_fingers(self, landmarks: np.ndarray) -> List[bool]:
        wrist = landmarks[0, :2]
        extended = [False] * 5

        # Thumb
        thumb_tip = landmarks[4, :2]
        thumb_ip = landmarks[3, :2]
        pinky_mcp = landmarks[17, :2]
        extended[0] = bool(np.linalg.norm(thumb_tip - pinky_mcp) > np.linalg.norm(thumb_ip - pinky_mcp) * 1.1)

        # Other 4 fingers
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        for i, (tip_idx, pip_idx) in enumerate(zip(finger_tips, finger_pips), start=1):
            tip_dist = np.linalg.norm(landmarks[tip_idx, :2] - wrist)
            pip_dist = np.linalg.norm(landmarks[pip_idx, :2] - wrist)
            extended[i] = bool(tip_dist > pip_dist * 1.08)

        return extended

    def close(self) -> None:
        if self.landmarker:
            try:
                self.landmarker.close()
            except Exception:
                pass
            self.landmarker = None
