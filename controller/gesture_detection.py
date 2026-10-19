"""
Gesture detection module for Burnout Camera Drive.
Analyzes hand landmarks, finger states, depth, and spatial orientation to
generate discrete driving actions (steer left/right, accelerate, brake, handbrake).

Supports:
- Configurable gestures for throttle, brake, and handbrake
- Configurable confirmation debounce frames
- Virtual steering wheel mode (two-hand angle) with single-hand fallback
- Direction change smoothing (anti-flutter zero-crossing debounce)
- Strict mutual exclusivity (braking always overrides acceleration)
"""

import math
import time
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List

from controller.hand_tracking import HandData
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
    Translates tracked hand landmarks into high-confidence driving commands with
    configurable gestures, smoothing, and anti-jitter debouncing.
    """

    def __init__(self, calibration: CalibrationManager):
        self.calib = calibration

        # Hysteresis state for acceleration
        self._is_accel_latched = False

        # Debounce counters
        self._brake_counter = 0
        self._handbrake_counter = 0
        self._accel_counter = 0

        # Direction change debounce (prevents rapid left-right flutter)
        self._prev_steering_dir = 0  # -1 = left, 0 = neutral, 1 = right
        self._direction_hold_frames = 0

    def reset(self) -> None:
        self._is_accel_latched = False
        self._brake_counter = 0
        self._handbrake_counter = 0
        self._accel_counter = 0
        self._prev_steering_dir = 0
        self._direction_hold_frames = 0

    def evaluate(self, hand: HandData) -> DrivingState:
        """
        Evaluate current frame's hand data and return DrivingState.
        """
        state = DrivingState()

        # If no hand detected, release all driving controls
        if not hand.is_detected:
            self.reset()
            state.gesture_name = "NO HAND"
            state.details = "Hand not in camera view"
            return state

        state.is_two_handed = hand.is_two_handed
        state.is_fallback = hand.is_fallback

        # Configurable settings
        confirm_frames = max(1, int(self.calib.config.get("gesture_confirm_frames", 2)))
        brake_type = self.calib.config.get("brake_gesture_type", "fist")
        handbrake_type = self.calib.config.get("handbrake_type", "peace")
        accel_mode = self.calib.config.get("accel_mode", "forward_push")

        # -------------------------------------------------------------
        # 1. BRAKE DETECTION (CONFIGURABLE)
        # -------------------------------------------------------------
        is_brake_raw = self._check_brake(hand, brake_type)

        # In two-hand mode, if either hand brakes, trigger brake
        if hand.is_two_handed and hand.second_fingers_extended:
            ext2 = hand.second_fingers_extended
            if (not ext2[1]) and (not ext2[2]) and (not ext2[3]) and (not ext2[4]):
                is_brake_raw = True

        if is_brake_raw:
            self._brake_counter += 1
        else:
            self._brake_counter = 0

        is_brake_confirmed = (self._brake_counter >= confirm_frames)

        # -------------------------------------------------------------
        # 2. HANDBRAKE DETECTION (CONFIGURABLE)
        # -------------------------------------------------------------
        is_handbrake_raw = self._check_handbrake(hand, handbrake_type)

        if is_handbrake_raw:
            self._handbrake_counter += 1
        else:
            self._handbrake_counter = 0

        is_handbrake_confirmed = (self._handbrake_counter >= confirm_frames)

        # -------------------------------------------------------------
        # 3. ACCELERATION DETECTION (DELIBERATE FORWARD MOVEMENT)
        # -------------------------------------------------------------
        is_accel_candidate = self._check_acceleration(hand, accel_mode)

        if is_accel_candidate and not is_brake_raw and not is_handbrake_raw:
            self._accel_counter += 1
        else:
            self._accel_counter = 0

        is_accel_confirmed = (self._accel_counter >= confirm_frames)

        # -------------------------------------------------------------
        # 4. RESOLVE ACTION PRIORITIES (MUTUAL EXCLUSIVITY)
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

        elif is_accel_confirmed:
            state.accelerate = True
            state.brake = False
            state.handbrake = False
            state.throttle_val = 1.0
            state.gesture_name = "ACCELERATING"

        else:
            # Coasting / Open Palm Neutral
            state.accelerate = False
            state.brake = False
            state.handbrake = False
            state.gesture_name = "COASTING (NEUTRAL)"

        # -------------------------------------------------------------
        # 5. STEERING EVALUATION & ANTI-FLUTTER DEBOUNCING
        # -------------------------------------------------------------
        raw_steering_val, steer_left, steer_right = self._compute_steering(hand)

        # Direction change debounce (prevents rapid left-right oscillations)
        dir_debounce_thresh = self.calib.direction_debounce_frames
        desired_dir = -1 if steer_left else (1 if steer_right else 0)

        if desired_dir != self._prev_steering_dir:
            # If changing from left directly to right (or vice-versa), require brief hold
            if self._prev_steering_dir != 0 and desired_dir != 0:
                self._direction_hold_frames += 1
                if self._direction_hold_frames < dir_debounce_thresh:
                    # Enforce neutral transition frame to avoid twitching
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

        # Telemetry detail text
        if state.steer_left:
            mode_tag = "2-Hand Wheel" if hand.is_two_handed else ("Fallback" if hand.is_fallback else "1-Hand")
            state.details = f"Steering Left {abs(state.steering_val)*100:.0f}% [{mode_tag}]"
        elif state.steer_right:
            mode_tag = "2-Hand Wheel" if hand.is_two_handed else ("Fallback" if hand.is_fallback else "1-Hand")
            state.details = f"Steering Right {abs(state.steering_val)*100:.0f}% [{mode_tag}]"
        else:
            state.details = "Steering Center (Deadzone)"

        return state

    def _check_brake(self, hand: HandData, brake_type: str) -> bool:
        """Evaluate if brake gesture is present."""
        extended = hand.fingers_extended
        if len(extended) < 5:
            return False

        if brake_type == "fist":
            # All four fingers curled closed
            return (not extended[1]) and (not extended[2]) and (not extended[3]) and (not extended[4])
        elif brake_type == "thumbs_down":
            # Thumb pointing downward
            if len(hand.landmarks_pixel) > 4:
                return hand.landmarks_pixel[4][1] > hand.landmarks_pixel[0][1]
        return (not extended[1]) and (not extended[2]) and (not extended[3]) and (not extended[4])

    def _check_handbrake(self, hand: HandData, handbrake_type: str) -> bool:
        """Evaluate if handbrake gesture is present."""
        extended = hand.fingers_extended
        if len(extended) < 5:
            return False

        if handbrake_type == "peace":
            # Peace / Victory sign: Index & Middle up, Ring & Pinky down
            return extended[1] and extended[2] and (not extended[3]) and (not extended[4])
        elif handbrake_type == "shaka":
            # Shaka: Thumb & Pinky extended, middle three curled
            return extended[0] and extended[4] and (not extended[1]) and (not extended[2]) and (not extended[3])
        elif handbrake_type == "pinch":
            # Pinch between thumb tip and index tip
            if len(hand.landmarks_pixel) > 8:
                t = hand.landmarks_pixel[4]
                i = hand.landmarks_pixel[8]
                dist = math.hypot(t[0] - i[0], t[1] - i[1])
                return dist < 32
        elif handbrake_type == "pinky":
            # Only pinky extended
            return extended[4] and (not extended[1]) and (not extended[2]) and (not extended[3])
        return extended[1] and extended[2] and (not extended[3]) and (not extended[4])

    def _check_acceleration(self, hand: HandData, accel_mode: str) -> bool:
        """Deliberate forward movement detection with hysteresis."""
        extended = hand.fingers_extended
        if len(extended) < 5:
            return False

        neutral_size = max(0.05, self.calib.neutral_size)
        current_size = hand.hand_size
        size_ratio = current_size / neutral_size
        forward_thresh = float(self.calib.config.get("accel_forward_threshold", 0.15))

        # Hysteresis bounds
        upper = 1.0 + forward_thresh
        lower = 1.0 + (forward_thresh * 0.40)

        if accel_mode == "forward_push":
            # Deliberate push toward camera (size expands)
            # Require at least 2 fingers extended so it's not a fist
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
            # Hand pushed upward/forward on screen past neutral Y
            ny = self.calib.neutral_y
            y_diff = ny - hand.hand_center[1]
            thresh_y = float(self.calib.config.get("accel_y_threshold", 0.10))
            return (y_diff > thresh_y) and (extended[1] or extended[2])

        elif accel_mode == "point":
            # Pointing index finger forward with other fingers curled
            return extended[1] and (not extended[2]) and (not extended[3]) and (not extended[4])

        return False

    def _compute_steering(self, hand: HandData) -> Tuple[float, bool, bool]:
        """
        Compute steering direction and intensity with deadzone snapping.
        Returns: (steering_val [-1.0..+1.0], steer_left, steer_right)
        """
        sensitivity = max(0.2, self.calib.steering_sensitivity)
        mode = self.calib.steering_mode

        # 1. TWO-HAND VIRTUAL STEERING WHEEL MODE
        if hand.is_two_handed or (mode == "two_hand_wheel" and not hand.is_fallback):
            wheel_deg = hand.wheel_angle * sensitivity
            deadzone_angle = float(self.calib.config.get("deadzone_angle", 12.0))
            max_angle = float(self.calib.config.get("max_steer_angle", 45.0))

            if wheel_deg < -deadzone_angle:
                # Turn Left
                val = max(-1.0, (wheel_deg + deadzone_angle) / (max_angle - deadzone_angle))
                return val, True, False
            elif wheel_deg > deadzone_angle:
                # Turn Right
                val = min(1.0, (wheel_deg - deadzone_angle) / (max_angle - deadzone_angle))
                return val, False, True
            else:
                # Deadzone neutral return
                return 0.0, False, False

        # 2. SINGLE-HAND TILT MODE
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

        # 3. POSITION OFFSET MODE (HORIZONTAL DISPLACEMENT)
        else:
            nx = self.calib.neutral_x
            deadzone = self.calib.deadzone_x
            range_span = max(0.10, self.calib.comfortable_range_x)

            # dx: negative when moved left of neutral, positive when moved right
            dx = (hand.hand_center[0] - nx) * sensitivity

            if dx < -deadzone:
                val = max(-1.0, (dx + deadzone) / range_span)
                return val, True, False
            elif dx > deadzone:
                val = min(1.0, (dx - deadzone) / range_span)
                return val, False, True
            else:
                # Return steering cleanly to neutral when hand returns to center
                return 0.0, False, False
