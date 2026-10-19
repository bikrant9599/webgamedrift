"""
Gesture detection module for Burnout Camera Drive.
Analyzes hand landmarks, finger states, depth, and spatial orientation to
generate discrete driving actions (steer left/right, accelerate, brake, handbrake).

Includes:
- Hysteresis and debouncing to prevent fluttering
- Mutual exclusivity for throttle and brakes
- Support for position-based and steering-wheel tilt modes
- Configurable handbrake gestures (Peace sign, Thumb out, Pinch)
"""

import math
import time
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

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
    Translates tracked hand landmarks into high-confidence driving commands.
    """

    def __init__(self, calibration: CalibrationManager):
        self.calib = calibration

        # Hysteresis state for acceleration
        self._is_accel_latched = False

        # Configurable handbrake gesture: "peace", "thumb", "pinch"
        self.handbrake_gesture_type = "peace"

        # Acceleration detection mode: "forward_push" (depth), "forward_y" (pitch/forward), "point"
        self.accel_mode = "forward_push"

        # Debounce counters
        self._consecutive_brake_frames = 0
        self._consecutive_handbrake_frames = 0
        self._min_confirm_frames = 2

    def reset(self) -> None:
        self._is_accel_latched = False
        self._consecutive_brake_frames = 0
        self._consecutive_handbrake_frames = 0

    def evaluate(self, hand: HandData) -> DrivingState:
        """
        Evaluate current frame's hand data and return DrivingState.
        """
        state = DrivingState()

        # If no hand is detected, all driving inputs are safely released
        if not hand.is_detected:
            self.reset()
            state.gesture_name = "NO HAND"
            state.details = "Hand not in camera view"
            return state

        extended = hand.fingers_extended
        thumb_ext = extended[0] if len(extended) > 0 else False
        index_ext = extended[1] if len(extended) > 1 else False
        middle_ext = extended[2] if len(extended) > 2 else False
        ring_ext = extended[3] if len(extended) > 3 else False
        pinky_ext = extended[4] if len(extended) > 4 else False

        # -------------------------------------------------------------
        # 1. BRAKE DETECTION (FIST)
        # -------------------------------------------------------------
        # When all four main fingers are curled into palm
        is_fist = (not index_ext) and (not middle_ext) and (not ring_ext) and (not pinky_ext)

        if is_fist:
            self._consecutive_brake_frames += 1
        else:
            self._consecutive_brake_frames = 0

        is_brake_confirmed = (self._consecutive_brake_frames >= self._min_confirm_frames)

        # -------------------------------------------------------------
        # 2. HANDBRAKE DETECTION (CONFIGURABLE)
        # -------------------------------------------------------------
        is_handbrake_raw = False
        if self.handbrake_gesture_type == "peace":
            # Victory / Peace sign: Index and Middle up, Ring and Pinky curled
            is_handbrake_raw = index_ext and middle_ext and (not ring_ext) and (not pinky_ext)
        elif self.handbrake_gesture_type == "thumb":
            # Thumb out / Shaka: Thumb and Pinky extended, Middle fingers curled
            is_handbrake_raw = thumb_ext and pinky_ext and (not index_ext) and (not middle_ext) and (not ring_ext)
        elif self.handbrake_gesture_type == "pinch":
            # Check Euclidean distance between thumb tip and index tip in pixel space
            if len(hand.landmarks_pixel) > 8:
                t_tip = hand.landmarks_pixel[4]
                i_tip = hand.landmarks_pixel[8]
                dist = math.hypot(t_tip[0] - i_tip[0], t_tip[1] - i_tip[1])
                is_handbrake_raw = (dist < 35)

        if is_handbrake_raw:
            self._consecutive_handbrake_frames += 1
        else:
            self._consecutive_handbrake_frames = 0

        is_handbrake_confirmed = (self._consecutive_handbrake_frames >= self._min_confirm_frames)

        # -------------------------------------------------------------
        # 3. ACCELERATION DETECTION (DELIBERATE FORWARD MOVEMENT)
        # -------------------------------------------------------------
        # Must not be braking or pulling handbrake
        is_accel_candidate = False
        neutral_size = max(0.05, self.calib.neutral_size)
        current_size = hand.hand_size
        size_ratio = current_size / neutral_size

        forward_thresh = float(self.calib.config.get("accel_forward_threshold", 0.15))
        # Hysteresis upper and lower thresholds
        upper_limit = 1.0 + forward_thresh
        lower_limit = 1.0 + (forward_thresh * 0.45)

        if self.accel_mode == "forward_push":
            # Hand pushed closer to camera (apparent size increases)
            # Must be an open or semi-open hand (not a fist)
            if index_ext or middle_ext or ring_ext:
                if not self._is_accel_latched:
                    if size_ratio >= upper_limit:
                        self._is_accel_latched = True
                else:
                    if size_ratio >= lower_limit:
                        self._is_accel_latched = True
                    else:
                        self._is_accel_latched = False
                is_accel_candidate = self._is_accel_latched

        elif self.accel_mode == "forward_y":
            # Hand pushed upward/forward on screen beyond neutral Y
            neutral_y = self.calib.neutral_y
            y_diff = neutral_y - hand.hand_center[1]  # positive when moved up/forward
            thresh_y = float(self.calib.config.get("accel_y_threshold", 0.10))
            is_accel_candidate = (y_diff > thresh_y) and (index_ext or middle_ext)

        elif self.accel_mode == "point":
            # Pointing index finger forward, other fingers curled
            is_accel_candidate = index_ext and (not middle_ext) and (not ring_ext) and (not pinky_ext)

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

        elif is_accel_candidate:
            state.accelerate = True
            state.brake = False
            state.handbrake = False
            ratio_norm = max(0.0, min(1.0, (size_ratio - 1.0) / (forward_thresh * 2.0)))
            state.throttle_val = ratio_norm
            state.gesture_name = "ACCELERATING (PUSH)"

        else:
            # Neutral / Open palm / Coasting
            state.accelerate = False
            state.brake = False
            state.handbrake = False
            state.gesture_name = "OPEN PALM (COAST)"

        # -------------------------------------------------------------
        # 5. STEERING EVALUATION (LEFT / RIGHT)
        # -------------------------------------------------------------
        sensitivity = max(0.2, self.calib.steering_sensitivity)
        mode = self.calib.steering_mode

        if mode == "tilt":
            # Steering wheel tilt angle mode
            angle = hand.tilt_angle * sensitivity
            deadzone_angle = float(self.calib.config.get("deadzone_angle", 12.0))
            max_angle = float(self.calib.config.get("max_steer_angle", 45.0))

            if angle < -deadzone_angle:
                state.steer_left = True
                state.steer_right = False
                intensity = min(1.0, (abs(angle) - deadzone_angle) / (max_angle - deadzone_angle))
                state.steering_val = -intensity
            elif angle > deadzone_angle:
                state.steer_right = True
                state.steer_left = False
                intensity = min(1.0, (angle - deadzone_angle) / (max_angle - deadzone_angle))
                state.steering_val = intensity
            else:
                state.steer_left = False
                state.steer_right = False
                state.steering_val = 0.0

        else:
            # Position offset mode (X coordinate relative to neutral_x)
            neutral_x = self.calib.neutral_x
            deadzone_x = self.calib.deadzone_x
            # Normalized displacement (-0.5 to +0.5)
            # Note: with mirrored camera, moving hand to screen left gives x < neutral_x
            dx = (hand.hand_center[0] - neutral_x) * sensitivity

            if dx < -deadzone_x:
                state.steer_left = True
                state.steer_right = False
                # Scale from deadzone boundary to max range (~0.30)
                span = max(0.05, 0.32 - deadzone_x)
                state.steering_val = max(-1.0, (dx + deadzone_x) / span)
            elif dx > deadzone_x:
                state.steer_right = True
                state.steer_left = False
                span = max(0.05, 0.32 - deadzone_x)
                state.steering_val = min(1.0, (dx - deadzone_x) / span)
            else:
                state.steer_left = False
                state.steer_right = False
                state.steering_val = 0.0

        # Append steering direction to gesture name if steering
        if state.steer_left:
            state.details = f"Steering Left ({abs(state.steering_val)*100:.0f}%)"
        elif state.steer_right:
            state.details = f"Steering Right ({abs(state.steering_val)*100:.0f}%)"
        else:
            state.details = "Steering Center (Deadzone)"

        return state
