"""
Hand tracking module using MediaPipe Hands.
Supports both modern MediaPipe Tasks API (0.10.x / 1.x) and legacy solutions API (0.9.x),
with automatic model resolution, coordinate smoothing (EMA), and robust telemetry.
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

# Model asset URL for modern MediaPipe Tasks
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
    landmarks_norm: List[Tuple[float, float, float]] = field(default_factory=list)
    landmarks_pixel: List[Tuple[int, int]] = field(default_factory=list)
    hand_center: Tuple[float, float] = (0.5, 0.5)
    hand_size: float = 0.20
    tilt_angle: float = 0.0          # degrees tilt (-90 to +90)
    fingers_extended: List[bool] = field(default_factory=lambda: [False]*5) # Thumb, Index, Middle, Ring, Pinky
    handedness: str = "Right"
    confidence: float = 0.0
    tracking_lost_time: float = 0.0


class HandTracker:
    """
    Tracks hand landmarks using MediaPipe with EMA coordinate smoothing.
    """

    def __init__(self,
                 max_hands: int = 1,
                 min_detection_confidence: float = 0.60,
                 min_tracking_confidence: float = 0.55,
                 smoothing_factor: float = 0.45):
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.smoothing_factor = smoothing_factor

        self.api_mode: str = "none"  # "tasks", "solutions", or "none"
        self.landmarker = None
        self._init_detector()

        # EMA smoothing buffers
        self._smoothed_landmarks: Optional[np.ndarray] = None
        self._smoothed_center: Optional[Tuple[float, float]] = None
        self._smoothed_size: Optional[float] = None
        self._last_detected_time = time.time()

    def _init_detector(self) -> None:
        if not MEDIAPIPE_AVAILABLE:
            return

        # 1. Attempt Modern MediaPipe Tasks API (MediaPipe 0.10.x / 1.x)
        try:
            if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
                ensure_model_asset(DEFAULT_MODEL_PATH)
                if os.path.exists(DEFAULT_MODEL_PATH):
                    base_opts = mp.tasks.BaseOptions(model_asset_path=DEFAULT_MODEL_PATH)
                    options = mp.tasks.vision.HandLandmarkerOptions(
                        base_options=base_opts,
                        num_hands=self.max_hands,
                        min_hand_detection_confidence=self.min_detection_confidence,
                        min_hand_presence_confidence=self.min_tracking_confidence
                    )
                    self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
                    self.api_mode = "tasks"
                    logger.info("MediaPipe HandLandmarker (Tasks API) initialized successfully.")
                    return
        except Exception as e:
            logger.warning(f"Could not initialize MediaPipe Tasks API: {e}. Falling back to Solutions...")

        # 2. Attempt Legacy MediaPipe Solutions API (MediaPipe <=0.9.x)
        try:
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self.landmarker = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self.max_hands,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                    model_complexity=1
                )
                self.api_mode = "solutions"
                logger.info("MediaPipe Hands (Solutions API) initialized successfully.")
                return
        except Exception as e:
            logger.warning(f"Could not initialize MediaPipe Solutions: {e}")

        logger.error("No compatible MediaPipe Hand tracking backend available.")
        self.api_mode = "none"

    def process_frame(self, frame_bgr: np.ndarray, mirror: bool = True) -> Tuple[np.ndarray, HandData]:
        """
        Process a camera frame.
        Returns:
            processed_frame: BGR image (mirrored if mirror=True)
            hand_data: HandData object containing landmarks and derived metrics
        """
        if frame_bgr is None:
            return frame_bgr, HandData(is_detected=False)

        # Mirror frame for natural self-view
        if mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)

        h, w, c = frame_bgr.shape
        hand_data = HandData()

        if self.api_mode == "none" or self.landmarker is None:
            return frame_bgr, hand_data

        raw_landmarks = None
        handedness_str = "Right"

        # -------------------------------------------------------------
        # INFERENCE
        # -------------------------------------------------------------
        try:
            if self.api_mode == "tasks":
                # Convert BGR to RGB for mp.Image
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                results = self.landmarker.detect(mp_image)

                if results.hand_landmarks and len(results.hand_landmarks) > 0:
                    first_hand = results.hand_landmarks[0]
                    raw_landmarks = np.array([
                        [lm.x, lm.y, lm.z] for lm in first_hand
                    ], dtype=np.float32)

                    if results.handedness and len(results.handedness) > 0:
                        handedness_str = results.handedness[0][0].display_name or results.handedness[0][0].category_name

            elif self.api_mode == "solutions":
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = self.landmarker.process(frame_rgb)
                frame_rgb.flags.writeable = True

                if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                    first_hand = results.multi_hand_landmarks[0]
                    raw_landmarks = np.array([
                        [lm.x, lm.y, lm.z] for lm in first_hand.landmark
                    ], dtype=np.float32)

                    if results.multi_handedness and len(results.multi_handedness) > 0:
                        handedness_str = results.multi_handedness[0].classification[0].label
        except Exception as e:
            logger.debug(f"Hand detection error: {e}")
            raw_landmarks = None

        # -------------------------------------------------------------
        # TRACKING LOST CHECK
        # -------------------------------------------------------------
        if raw_landmarks is None:
            self._smoothed_landmarks = None
            self._smoothed_center = None
            self._smoothed_size = None
            hand_data.is_detected = False
            hand_data.tracking_lost_time = time.time()
            return frame_bgr, hand_data

        # Hand detected!
        hand_data.is_detected = True
        hand_data.handedness = handedness_str
        self._last_detected_time = time.time()

        # Apply Exponential Moving Average (EMA) smoothing
        if self._smoothed_landmarks is None:
            self._smoothed_landmarks = raw_landmarks.copy()
        else:
            alpha = self.smoothing_factor
            self._smoothed_landmarks = alpha * raw_landmarks + (1.0 - alpha) * self._smoothed_landmarks

        # Compute pixel and normalized coordinates
        landmarks_norm: List[Tuple[float, float, float]] = []
        landmarks_pixel: List[Tuple[int, int]] = []
        for i in range(21):
            x = float(self._smoothed_landmarks[i, 0])
            y = float(self._smoothed_landmarks[i, 1])
            z = float(self._smoothed_landmarks[i, 2])
            landmarks_norm.append((x, y, z))
            landmarks_pixel.append((int(x * w), int(y * h)))

        hand_data.landmarks_norm = landmarks_norm
        hand_data.landmarks_pixel = landmarks_pixel

        # Centroid: weighted average of Wrist(0), Index MCP(5), Middle MCP(9), Pinky MCP(17)
        pts = self._smoothed_landmarks[[0, 5, 9, 17], :2]
        raw_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))

        if self._smoothed_center is None:
            self._smoothed_center = raw_center
        else:
            alpha = self.smoothing_factor
            self._smoothed_center = (
                alpha * raw_center[0] + (1.0 - alpha) * self._smoothed_center[0],
                alpha * raw_center[1] + (1.0 - alpha) * self._smoothed_center[1]
            )
        hand_data.hand_center = self._smoothed_center

        # Hand size: distance from wrist (0) to middle MCP (9)
        wrist = self._smoothed_landmarks[0, :2]
        middle_mcp = self._smoothed_landmarks[9, :2]
        raw_size = float(np.linalg.norm(wrist - middle_mcp))

        if self._smoothed_size is None:
            self._smoothed_size = raw_size
        else:
            alpha = self.smoothing_factor
            self._smoothed_size = alpha * raw_size + (1.0 - alpha) * self._smoothed_size
        hand_data.hand_size = self._smoothed_size

        # Tilt angle: angle deviation from vertical (0 deg = straight up, negative = tilt left, positive = tilt right)
        dx = middle_mcp[0] - wrist[0]
        dy = middle_mcp[1] - wrist[1]
        raw_angle = math.degrees(math.atan2(dx, -dy))
        hand_data.tilt_angle = max(-90.0, min(90.0, raw_angle))

        # Finger extensions
        hand_data.fingers_extended = self._detect_extended_fingers(self._smoothed_landmarks)

        return frame_bgr, hand_data

    def _detect_extended_fingers(self, landmarks: np.ndarray) -> List[bool]:
        wrist = landmarks[0, :2]
        extended = [False] * 5

        # Thumb: compare distance from tip to pinky MCP vs IP joint
        thumb_tip = landmarks[4, :2]
        thumb_ip = landmarks[3, :2]
        pinky_mcp = landmarks[17, :2]
        extended[0] = bool(np.linalg.norm(thumb_tip - pinky_mcp) > np.linalg.norm(thumb_ip - pinky_mcp) * 1.1)

        # Fingers: Tip distance to wrist vs PIP distance to wrist
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
