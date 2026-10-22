"""
Hand tracking module using MediaPipe Hands.
Supports single-hand tracking, dual-hand tracking, accurate handedness detection (Left vs Right),
and specific gesture recognition including the right-hand raised-index-finger acceleration gesture.
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
    from controller.telemetry_suppress import silence_clearcut_telemetry
    silence_clearcut_telemetry()
except Exception:
    pass

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

    # Specific Gesture Recognition flags
    right_hand_detected: bool = False
    left_hand_detected: bool = False
    right_index_raised: bool = False
    any_index_raised: bool = False
    any_peace_sign: bool = False
    any_fist: bool = False
    any_open_palm: bool = False
    right_hand_landmarks: List[Tuple[int, int]] = field(default_factory=list)
    left_hand_landmarks: List[Tuple[int, int]] = field(default_factory=list)

    confidence: float = 0.0
    tracking_lost_time: float = 0.0


class HandTracker:
    """
    Tracks hand landmarks using MediaPipe with EMA smoothing, dual-hand detection,
    and right-hand index-finger forward gesture recognition.
    """

    def __init__(self,
                 two_handed_mode: bool = True,  # Default True to allow notebook & dual-hand detection
                 min_detection_confidence: float = 0.55,
                 min_tracking_confidence: float = 0.50,
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
        Process camera frame and detect single or dual hands, accurately identifying
        the physical Right Hand and detecting the raised index finger gesture.
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
        detected_labels: List[str] = []

        try:
            if self.api_mode == "tasks":
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                results = self.landmarker.detect(mp_image)

                if results.hand_landmarks:
                    for i, lm_list in enumerate(results.hand_landmarks):
                        arr = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)
                        detected_hands_raw.append(arr)
                        raw_label = "Right"
                        if results.handedness and i < len(results.handedness) and results.handedness[i]:
                            raw_label = results.handedness[i][0].display_name or results.handedness[i][0].category_name or "Right"
                        detected_labels.append(raw_label)

            elif self.api_mode == "solutions":
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = self.landmarker.process(frame_rgb)
                frame_rgb.flags.writeable = True

                if results.multi_hand_landmarks:
                    for i, lm_list in enumerate(results.multi_hand_landmarks):
                        arr = np.array([[lm.x, lm.y, lm.z] for lm in lm_list.landmark], dtype=np.float32)
                        detected_hands_raw.append(arr)
                        raw_label = "Right"
                        if results.multi_handedness and i < len(results.multi_handedness):
                            raw_label = results.multi_handedness[i].classification[0].label
                        detected_labels.append(raw_label)

        except Exception as e:
            logger.debug(f"Hand detection error: {e}")
            detected_hands_raw.clear()

        # If no hands detected
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
        # ACCURATE PHYSICAL RIGHT VS LEFT HAND IDENTIFICATION
        # -------------------------------------------------------------
        right_hand_raw = None
        left_hand_raw = None

        if len(detected_hands_raw) >= 2:
            h0, lbl0 = detected_hands_raw[0], (detected_labels[0] if len(detected_labels) > 0 else "")
            h1, lbl1 = detected_hands_raw[1], (detected_labels[1] if len(detected_labels) > 1 else "")

            h0_is_right = self._is_anatomical_right_hand(h0, lbl0, mirror)
            h1_is_right = self._is_anatomical_right_hand(h1, lbl1, mirror)

            if h0_is_right and not h1_is_right:
                right_hand_raw = h0
                left_hand_raw = h1
            elif h1_is_right and not h0_is_right:
                right_hand_raw = h1
                left_hand_raw = h0
            else:
                # Fallback to spatial position if labels are ambiguous
                cx0 = float(np.mean(h0[[0, 5, 9, 17], 0]))
                cx1 = float(np.mean(h1[[0, 5, 9, 17], 0]))
                if mirror:
                    left_hand_raw, right_hand_raw = (h0, h1) if cx0 < cx1 else (h1, h0)
                else:
                    right_hand_raw, left_hand_raw = (h0, h1) if cx0 < cx1 else (h1, h0)

        elif len(detected_hands_raw) == 1:
            h0 = detected_hands_raw[0]
            lbl0 = detected_labels[0] if len(detected_labels) > 0 else ""
            if self._is_anatomical_right_hand(h0, lbl0, mirror):
                right_hand_raw = h0
            else:
                left_hand_raw = h0
                # Fallback: single detected hand can drive all gestures
                right_hand_raw = h0

        # Update detected flags and pixel coordinates
        if right_hand_raw is not None:
            hand_data.right_hand_detected = True
            hand_data.confidence = 0.95
            hand_data.right_hand_landmarks = [(int(x * w), int(y * h)) for x, y in right_hand_raw[:, :2]]
            hand_data.right_index_raised = self._is_index_finger_raised(right_hand_raw, h, w)

        if left_hand_raw is not None:
            hand_data.left_hand_detected = True
            hand_data.left_hand_landmarks = [(int(x * w), int(y * h)) for x, y in left_hand_raw[:, :2]]
            if not hand_data.right_hand_detected:
                hand_data.confidence = 0.90

        # Unified gesture detection across all visible hands
        for h_raw in detected_hands_raw:
            if self._is_index_finger_raised(h_raw, h, w):
                hand_data.any_index_raised = True
                hand_data.right_index_raised = True
            if self._is_peace_sign(h_raw):
                hand_data.any_peace_sign = True
            if self._is_fist(h_raw):
                hand_data.any_fist = True
            if self._is_open_palm(h_raw):
                hand_data.any_open_palm = True

        # -------------------------------------------------------------
        # TWO-HAND STEERING WHEEL CALCULATIONS
        # -------------------------------------------------------------
        if self.two_handed_mode and len(detected_hands_raw) >= 2 and left_hand_raw is not None and right_hand_raw is not None:
            hand_data.is_two_handed = True
            hand_data.is_fallback = False

            # Smooth primary (left) and secondary (right)
            if self._smoothed_landmarks is None:
                self._smoothed_landmarks = left_hand_raw.copy()
            else:
                self._smoothed_landmarks = alpha * left_hand_raw + (1.0 - alpha) * self._smoothed_landmarks

            if self._smoothed_second_landmarks is None:
                self._smoothed_second_landmarks = right_hand_raw.copy()
            else:
                self._smoothed_second_landmarks = alpha * right_hand_raw + (1.0 - alpha) * self._smoothed_second_landmarks

            left_cx = float(np.mean(self._smoothed_landmarks[[0, 5, 9, 17], 0]))
            left_cy = float(np.mean(self._smoothed_landmarks[[0, 5, 9, 17], 1]))
            right_cx = float(np.mean(self._smoothed_second_landmarks[[0, 5, 9, 17], 0]))
            right_cy = float(np.mean(self._smoothed_second_landmarks[[0, 5, 9, 17], 1]))

            hand_data.left_hand_center = (left_cx, left_cy)
            hand_data.right_hand_center = (right_cx, right_cy)
            hand_data.hand_center = ((left_cx + right_cx) / 2.0, (left_cy + right_cy) / 2.0)

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

            hand_data.landmarks_pixel = [(int(x * w), int(y * h)) for x, y in self._smoothed_landmarks[:, :2]]
            hand_data.second_landmarks_pixel = [(int(x * w), int(y * h)) for x, y in self._smoothed_second_landmarks[:, :2]]

            hand_data.fingers_extended = self._detect_extended_fingers(self._smoothed_landmarks)
            hand_data.second_fingers_extended = self._detect_extended_fingers(self._smoothed_second_landmarks)

            sz1 = float(np.linalg.norm(self._smoothed_landmarks[0, :2] - self._smoothed_landmarks[9, :2]))
            sz2 = float(np.linalg.norm(self._smoothed_second_landmarks[0, :2] - self._smoothed_second_landmarks[9, :2]))
            hand_data.hand_size = (sz1 + sz2) / 2.0
            return frame_bgr, hand_data

        # -------------------------------------------------------------
        # SINGLE HAND TRACKING (OR FALLBACK)
        # -------------------------------------------------------------
        raw_primary = detected_hands_raw[0]
        if self.two_handed_mode:
            hand_data.is_two_handed = False
            hand_data.is_fallback = True

        if self._smoothed_landmarks is None:
            self._smoothed_landmarks = raw_primary.copy()
        else:
            self._smoothed_landmarks = alpha * raw_primary + (1.0 - alpha) * self._smoothed_landmarks

        hand_data.landmarks_pixel = [(int(x * w), int(y * h)) for x, y in self._smoothed_landmarks[:, :2]]
        hand_data.landmarks_norm = [(float(x), float(y), float(z)) for x, y, z in self._smoothed_landmarks]

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

        wrist = self._smoothed_landmarks[0, :2]
        middle_mcp = self._smoothed_landmarks[9, :2]
        raw_size = float(np.linalg.norm(wrist - middle_mcp))
        if self._smoothed_size is None:
            self._smoothed_size = raw_size
        else:
            self._smoothed_size = alpha * raw_size + (1.0 - alpha) * self._smoothed_size
        hand_data.hand_size = self._smoothed_size

        dx = middle_mcp[0] - wrist[0]
        dy = middle_mcp[1] - wrist[1]
        raw_angle = math.degrees(math.atan2(dx, -dy))
        hand_data.tilt_angle = max(-90.0, min(90.0, raw_angle))
        hand_data.wheel_angle = hand_data.tilt_angle

        hand_data.fingers_extended = self._detect_extended_fingers(self._smoothed_landmarks)
        return frame_bgr, hand_data

    def _is_anatomical_right_hand(self, landmarks: np.ndarray, mp_label: str, mirror: bool) -> bool:
        """
        Determines if a hand is anatomically the user's physical Right hand.
        Uses MediaPipe's anatomical classification, verified with palm normal and thumb vector.
        Works anywhere on screen, even if the right hand crosses over to the left side.
        """
        if mp_label:
            # In mirrored selfie view (flipped horizontally), MediaPipe sees a mirrored reflection,
            # so physical Right hand is classified as 'Left' by MP, and vice versa.
            expected_for_phys_right = "Left" if mirror else "Right"
            if mp_label.strip().lower() == expected_for_phys_right.lower():
                return True
            elif mp_label.strip().lower() in ("left", "right"):
                return False

        # Fallback to anatomical thumb/palm vector check:
        # Landmarks: 0=wrist, 5=index_mcp, 17=pinky_mcp, 4=thumb_tip
        try:
            wrist = landmarks[0, :2]
            index_mcp = landmarks[5, :2]
            pinky_mcp = landmarks[17, :2]
            thumb_tip = landmarks[4, :2]

            up_vec = index_mcp - wrist
            palm_across = pinky_mcp - index_mcp
            cross_z = up_vec[0] * palm_across[1] - up_vec[1] * palm_across[0]
            thumb_vec = thumb_tip - index_mcp
            thumb_left = thumb_vec[0] < 0

            if cross_z > 0:  # Palm facing camera
                return thumb_left if mirror else (not thumb_left)
            else:  # Back of hand facing camera
                return (not thumb_left) if mirror else thumb_left
        except Exception:
            # Fallback to screen position if coordinates invalid
            return float(landmarks[0, 0]) > 0.5 if mirror else float(landmarks[0, 0]) < 0.5

    def _is_index_finger_raised(self, landmarks: np.ndarray, h: int = 480, w: int = 640) -> bool:
        """
        Returns True if the index finger is raised and extended upward,
        while middle, ring, pinky fingers are curled towards palm.
        """
        wrist = landmarks[0, :2]
        i_tip = landmarks[8, :2]
        i_pip = landmarks[6, :2]
        i_mcp = landmarks[5, :2]

        # Index tip higher than PIP or extended away from wrist
        i_dist_tip = float(np.linalg.norm(i_tip - wrist))
        i_dist_pip = float(np.linalg.norm(i_pip - wrist))
        is_index_extended = (i_dist_tip > i_dist_pip * 1.08) or (i_tip[1] < i_pip[1] and i_tip[1] < i_mcp[1])

        if not is_index_extended:
            return False

        ext = self._detect_extended_fingers(landmarks)
        # If middle finger is also extended, it's peace sign or open palm, NOT index raised
        if ext[2]:
            return False
        # Ring or pinky shouldn't be extended
        if ext[3] and ext[4]:
            return False

        return True

    def _is_peace_sign(self, landmarks: np.ndarray) -> bool:
        """Returns True if Index and Middle fingers are extended, while Ring and Pinky are curled."""
        ext = self._detect_extended_fingers(landmarks)
        return bool(ext[1] and ext[2] and not ext[3] and not ext[4])

    def _is_fist(self, landmarks: np.ndarray) -> bool:
        """Returns True if all four fingers (Index, Middle, Ring, Pinky) are curled."""
        ext = self._detect_extended_fingers(landmarks)
        return bool(not ext[1] and not ext[2] and not ext[3] and not ext[4])

    def _is_open_palm(self, landmarks: np.ndarray) -> bool:
        """Returns True if all four fingers are extended."""
        ext = self._detect_extended_fingers(landmarks)
        return bool(ext[1] and ext[2] and ext[3] and ext[4])

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
