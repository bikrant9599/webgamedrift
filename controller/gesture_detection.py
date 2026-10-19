"""
Gesture detection module for Burnout Camera Drive.
Analyzes hand landmarks, finger states, depth, and spatial orientation to
generate discrete driving actions (steer left/right, accelerate, brake, handbrake).

Supports:
- NOTEBOOK-BASED PHYSICAL STEERING:
  - Notebook clockwise rotation -> steer right
  - Notebook counter-clockwise rotation -> steer left
  - Return to neutral -> center steering
- RIGHT-HAND RAISED INDEX FINGER ACCELERATION:
  - Activates forward acceleration when user's physical right index finger is raised
  - Releases immediately when finger lowered or hand lost
- SIMULTANEOUS COMBINED CONTROL:
  - Steer left/right while accelerating, accelerate straight, or coast
- Configurable gestures for throttle, brake, and handbrake
- Configurable confirmation debounce frames
- Strict mutual exclusivity (braking always overrides acceleration)
"""

import math
import time
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List

from controller.hand_tracking import HandData
from controller.notebook_tracking import NotebookData
from controller.calibration import CalibrationManager

logger = logging.getLogger("burnout.gesture")


@dataclass
class DrivingState:
    steer_left: bool = False
    steer_right: bool = False
    accelerate: bool = False
    brake: bool = False
    handbrake: bool = False

    steering_val: float = 0.0    # -1.0 (full left) to +1.0 (full right)
    throttle_val: float = 0.0    # 0.0 to 1.0
    brake_val: float = 0.0       # 0.0 to 1.0

    gesture_name: str = "NO HAND"
    details: str = ""
    is_two_handed: bool = False
    is_fallback: bool = False
    notebook_active: bool = False
    right_index_active: bool = False

    def to_action_dict(self) -> Dict[str, bool]:
        return {
            "steer_left": self.steer_left,
            "steer_right": self.steer_right,
            "accelerate": self.accelerate,
            "brake": self.brake,
            "handbrake": self.handbrake,
        }


class GestureDetector:
    """
    Translates tracked hand landmarks and notebook orientation into high-confidence
    driving commands with anti-jitter debouncing.
    """

    def __init__(self, calibration: CalibrationManager):
        self.calib = calibration

        # Hysteresis state for acceleration
        self._is_accel_latched = False

        # Debounce counters
        self._brake_counter = 0
        self._handbrake_counter = 0
        self._accel_counter = 0
        self._right_index_counter = 0

        # Direction change debounce (prevents rapid left-right flutter)
        self._prev_steering_dir = 0
        self._direction_hold_frames = 0

    def reset(self) -> None:
        self._is_accel_latched = False
        self._brake_counter = 0
        self._handbrake_counter = 0
        self._accel_counter = 0
        self._right_index_counter = 0
        self._prev_steering_dir = 0
        self._direction_hold_frames = 0

    def evaluate(self, hand: HandData, notebook: Optional[NotebookData] = None, notebook_data: Optional[NotebookData] = None) -> DrivingState:
        """
        Evaluate current frame's hand data and optional notebook tracking data.
        Returns unified DrivingState.
        """
        if notebook is None and notebook_data is not None:
            notebook = notebook_data

        state = DrivingState()
        use_notebook = self.calib.config.get("notebook_steering_enabled", True) and (notebook is not None)

        confirm_frames = max(1, int(self.calib.config.get("gesture_confirm_frames", 2)))
        brake_type = self.calib.config.get("brake_gesture_type", "fist")
        handbrake_type = self.calib.config.get("handbrake_type", "peace")

        # -------------------------------------------------------------
        # 1. BRAKE & HANDBRAKE DETECTION
        # -------------------------------------------------------------
        is_brake_raw = False
        is_handbrake_raw = False

        if hand.is_detected:
            is_brake_raw = self._check_brake(hand, brake_type)
            is_handbrake_raw = self._check_handbrake(hand, handbrake_type)

            if hand.is_two_handed and hand.second_fingers_extended:
                ext2 = hand.second_fingers_extended
                if (not ext2[1]) and (not ext2[2]) and (not ext2[3]) and (not ext2[4]):
                    is_brake_raw = True

        if is_brake_raw:
            self._brake_counter += 1
        else:
            self._brake_counter = 0
        is_brake_confirmed = (self._brake_counter >= confirm_frames)

        if is_handbrake_raw:
            self._handbrake_counter += 1
        else:
            self._handbrake_counter = 0
        is_handbrake_confirmed = (self._handbrake_counter >= confirm_frames)

        # -------------------------------------------------------------
        # 2. ACCELERATION DETECTION
        # -------------------------------------------------------------
        # Priority: Right-Hand Raised Index Finger Gesture
        is_accel_candidate = False

        if hand.is_detected and hand.right_hand_detected:
            if hand.right_index_raised:
                self._right_index_counter += 1
            else:
                self._right_index_counter = 0

            if self._right_index_counter >= confirm_frames:
                is_accel_candidate = True
                state.right_index_active = True
        else:
            self._right_index_counter = 0

        # Fallback to gesture accel mode if right-hand index finger is not used
        if not is_accel_candidate and not use_notebook and hand.is_detected:
            accel_mode = self.calib.config.get("accel_mode", "forward_push")
            if self._check_acceleration_fallback(hand, accel_mode):
                self._accel_counter += 1
            else:
                self._accel_counter = 0
            if self._accel_counter >= confirm_frames:
                is_accel_candidate = True

        # -------------------------------------------------------------
        # 3. RESOLVE SPEED ACTIONS (MUTUAL EXCLUSIVITY)
        # -------------------------------------------------------------
        if is_brake_confirmed:
            state.brake = True
            state.accelerate = False
            state.handbrake = False
            state.brake_val = 1.0
            state.gesture_name = "BRAKE (FIST)"
            self._is_accel_latched = False

        elif is_handbrake_confirmed:
            state.handbrake = True
            state.brake = False
            state.accelerate = False
            state.gesture_name = "HANDBRAKE (DRIFT)"
            self._is_accel_latched = False

        elif is_accel_candidate:
            state.accelerate = True
            state.brake = False
            state.handbrake = False
            state.throttle_val = 1.0
            state.gesture_name = "FORWARD (RIGHT INDEX RAISED)" if state.right_index_active else "ACCELERATING"

        else:
            state.accelerate = False
            state.brake = False
            state.handbrake = False
            state.gesture_name = "COASTING / NEUTRAL"

        # -------------------------------------------------------------
        # 4. STEERING: NOTEBOOK OR HANDS
        # -------------------------------------------------------------
        if use_notebook and notebook is not None:
            state.notebook_active = True
            raw_steering_val, steer_left, steer_right = self._compute_notebook_steering(notebook)
        elif hand.is_detected:
            raw_steering_val, steer_left, steer_right = self._compute_hand_steering(hand)
        else:
            raw_steering_val, steer_left, steer_right = 0.0, False, False

        # Anti-flutter direction debounce
        dir_debounce_thresh = self.calib.direction_debounce_frames
        desired_dir = -1 if steer_left else (1 if steer_right else 0)

        if desired_dir != self._prev_steering_dir:
            if self._prev_steering_dir != 0 and desired_dir != 0:
                self._direction_hold_frames += 1
                if self._direction_hold_frames < dir_debounce_thresh:
                    steer_left = False
                    steer_right = False
                    raw_steering_val = 0.0
                else:
                    self._prev_steering_dir = desired_dir
                    self._direction_hold_frames = 0
            else:
                self._prev_steering_dir = desired_dir
                self._direction_hold_frames = 0
        else:
            self._direction_hold_frames = 0

        state.steer_left = steer_left
        state.steer_right = steer_right
        state.steering_val = raw_steering_val
        state.is_two_handed = hand.is_two_handed
        state.is_fallback = hand.is_fallback

        # Telemetry detail string
        if use_notebook and notebook is not None and notebook.is_detected:
            rot_str = f"{notebook.rotation_angle:+.1f}°"
            gas_str = "ACCEL ON" if state.accelerate else "COAST"
            if state.steer_left:
                state.details = f"Notebook Left {rot_str} ({abs(state.steering_val)*100:.0f}%) | {gas_str}"
            elif state.steer_right:
                state.details = f"Notebook Right {rot_str} ({abs(state.steering_val)*100:.0f}%) | {gas_str}"
            else:
                state.details = f"Notebook Center {rot_str} (Deadzone) | {gas_str}"
        elif state.steer_left:
            state.details = f"Steering Left ({abs(state.steering_val)*100:.0f}%)"
        elif state.steer_right:
            state.details = f"Steering Right ({abs(state.steering_val)*100:.0f}%)"
        else:
            state.details = "Steering Center (Deadzone)"

        return state

    def _compute_notebook_steering(self, nb: NotebookData) -> Tuple[float, bool, bool]:
        """
        Calculates steering from the physical notebook's rotation angle.
        Rotates clockwise -> Steer Right (positive angle)
        Rotates counter-clockwise -> Steer Left (negative angle)
        Within deadzone -> Center neutral
        If notebook not detected -> Releases all steering (Safety)
        """
        if not nb.is_detected:
            return 0.0, False, False

        deadzone = float(self.calib.config.get("notebook_deadzone_angle", 5.0))
        max_angle = float(self.calib.config.get("notebook_max_steer_angle", 35.0))
        sens = float(self.calib.steering_sensitivity)

        rot_deg = nb.rotation_angle * sens

        if rot_deg < -deadzone:
            # Counter-clockwise: Steer Left
            span = max(5.0, max_angle - deadzone)
            val = max(-1.0, (rot_deg + deadzone) / span)
            return val, True, False
        elif rot_deg > deadzone:
            # Clockwise: Steer Right
            span = max(5.0, max_angle - deadzone)
            val = min(1.0, (rot_deg - deadzone) / span)
            return val, False, True
        else:
            # Inside deadzone: Center neutral
            return 0.0, False, False

    def _compute_hand_steering(self, hand: HandData) -> Tuple[float, bool, bool]:
        """Fallback hand steering (two-hand wheel or single-hand)."""
        sensitivity = max(0.2, self.calib.steering_sensitivity)
        mode = self.calib.steering_mode

        if hand.is_two_handed or (mode == "two_hand_wheel" and not hand.is_fallback):
            wheel_deg = hand.wheel_angle * sensitivity
            deadzone_angle = float(self.calib.config.get("deadzone_angle", 12.0))
            max_angle = float(self.calib.config.get("max_steer_angle", 45.0))

            if wheel_deg < -deadzone_angle:
                val = max(-1.0, (wheel_deg + deadzone_angle) / (max_angle - deadzone_angle))
                return val, True, False
            elif wheel_deg > deadzone_angle:
                val = min(1.0, (wheel_deg - deadzone_angle) / (max_angle - deadzone_angle))
                return val, False, True
            else:
                return 0.0, False, False

        elif mode == "tilt":
            angle = hand.tilt_angle * sensitivity
            deadzone_angle = float(self.calib.config.get("deadzone_angle", 12.0))
            max_angle = float(self.calib.config.get("max_steer_angle", 45.0))
            if angle < -deadzone_angle:
                val = max(-1.0, (angle + deadzone_angle) / (max_angle - deadzone_angle))
                return val, True, False
            elif angle > deadzone_angle:
                val = min(1.0, (angle - deadzone_angle) / (max_angle - deadzone_angle))
                return val, False, True
            else:
                return 0.0, False, False
        else:
            nx = self.calib.neutral_x
            deadzone = self.calib.deadzone_x
            range_span = max(0.10, self.calib.comfortable_range_x)
            dx = (hand.hand_center[0] - nx) * sensitivity

            if dx < -deadzone:
                val = max(-1.0, (dx + deadzone) / range_span)
                return val, True, False
            elif dx > deadzone:
                val = min(1.0, (dx - deadzone) / range_span)
                return val, False, True
            else:
                return 0.0, False, False

    def _check_brake(self, hand: HandData, brake_type: str) -> bool:
        if hand.right_index_raised:
            return False
        extended = hand.fingers_extended
        if len(extended) < 5:
            return False
        if brake_type == "fist":
            return (not extended[1]) and (not extended[2]) and (not extended[3]) and (not extended[4])
        elif brake_type == "thumbs_down":
            if len(hand.landmarks_pixel) > 4:
                return hand.landmarks_pixel[4][1] > hand.landmarks_pixel[0][1]
        return (not extended[1]) and (not extended[2]) and (not extended[3]) and (not extended[4])

    def _check_handbrake(self, hand: HandData, handbrake_type: str) -> bool:
        extended = hand.fingers_extended
        if len(extended) < 5:
            return False
        if handbrake_type == "peace":
            return extended[1] and extended[2] and (not extended[3]) and (not extended[4])
        elif handbrake_type == "shaka":
            return extended[0] and extended[4] and (not extended[1]) and (not extended[2]) and (not extended[3])
        elif handbrake_type == "pinch":
            if len(hand.landmarks_pixel) > 8:
                t = hand.landmarks_pixel[4]
                i = hand.landmarks_pixel[8]
                return math.hypot(t[0] - i[0], t[1] - i[1]) < 32
        elif handbrake_type == "pinky":
            return extended[4] and (not extended[1]) and (not extended[2]) and (not extended[3])
        return extended[1] and extended[2] and (not extended[3]) and (not extended[4])

    def _check_acceleration_fallback(self, hand: HandData, accel_mode: str) -> bool:
        extended = hand.fingers_extended
        if len(extended) < 5:
            return False
        neutral_size = max(0.05, self.calib.neutral_size)
        current_size = hand.hand_size
        size_ratio = current_size / neutral_size
        forward_thresh = float(self.calib.config.get("accel_forward_threshold", 0.15))
        upper = 1.0 + forward_thresh
        lower = 1.0 + (forward_thresh * 0.40)

        if accel_mode == "forward_push":
            if extended[1] or extended[2]:
                if not self._is_accel_latched:
                    if size_ratio >= upper:
                        self._is_accel_latched = True
                else:
                    if size_ratio >= lower:
                        self._is_accel_latched = True
                    else:
                        self._is_accel_latched = False
                return self._is_accel_latched
        elif accel_mode == "forward_y":
            ny = self.calib.neutral_y
            y_diff = ny - hand.hand_center[1]
            thresh_y = float(self.calib.config.get("accel_y_threshold", 0.10))
            return (y_diff > thresh_y) and (extended[1] or extended[2])
        elif accel_mode == "point":
            return extended[1] and (not extended[2]) and (not extended[3]) and (not extended[4])
        return False
