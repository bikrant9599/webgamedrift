"""
Smart calibration manager for Burnout Camera Drive.
Handles:
- Notebook steering calibration (neutral orientation, left/right steering range, corner locking)
- Right-hand raised index finger gesture verification
- Hand-based calibration and sensitivity presets
- Settings validation and JSON persistence
"""

import os
import json
import time
import math
import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

logger = logging.getLogger("burnout.calibration")

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration_config.json")

PRESETS: Dict[str, Dict[str, Any]] = {
    "beginner": {
        "steering_sensitivity": 0.8,
        "deadzone_x": 0.12,
        "deadzone_angle": 16.0,
        "notebook_deadzone_angle": 8.0,
        "smoothing_factor": 0.30,
        "direction_debounce_frames": 3,
        "preset_name": "Beginner (Smooth & Stable)"
    },
    "normal": {
        "steering_sensitivity": 1.0,
        "deadzone_x": 0.08,
        "deadzone_angle": 12.0,
        "notebook_deadzone_angle": 5.0,
        "smoothing_factor": 0.45,
        "direction_debounce_frames": 2,
        "preset_name": "Normal (Balanced Drift)"
    },
    "pro": {
        "steering_sensitivity": 1.5,
        "deadzone_x": 0.04,
        "deadzone_angle": 8.0,
        "notebook_deadzone_angle": 3.0,
        "smoothing_factor": 0.65,
        "direction_debounce_frames": 1,
        "preset_name": "Pro (Ultra Responsive)"
    }
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "active_preset": "normal",
    "steering_mode": "notebook",  # 'notebook', 'two_hand_wheel', 'position', 'tilt'
    "notebook_steering_enabled": True,
    "notebook_neutral_angle": 0.0,
    "notebook_neutral_pitch": 0.0,
    "notebook_neutral_height": 0.0,
    "notebook_neutral_width": 0.0,
    "notebook_deadzone_angle": 5.0,     # degrees of deadzone for notebook steering
    "notebook_max_steer_angle": 32.0,   # degrees for 100% steering deflection
    "notebook_brake_pitch_thresh": 9.0, # degrees of forward tilt to trigger brake
    "notebook_brake_height_thresh": 0.10, # foreshortening drop ratio to trigger brake
    "notebook_brake_sensitivity": 1.0,
    "notebook_hsv_lower": [5, 50, 60],  # Orange notebook bounds (tolerant of dim room lighting)
    "notebook_hsv_upper": [30, 255, 255],
    "notebook_color_name": "Orange (Default)",
    "notebook_calibrated": True,

    # Hand parameters
    "two_hand_mode_enabled": True,
    "neutral_x": 0.50,
    "neutral_y": 0.55,
    "neutral_hand_size": 0.20,
    "comfortable_range_x": 0.25,
    "deadzone_x": 0.08,
    "deadzone_angle": 12.0,
    "steering_sensitivity": 1.0,
    "smoothing_factor": 0.45,
    "direction_debounce_frames": 2,

    # Gestures
    "gesture_confirm_frames": 2,
    "accel_mode": "right_index",       # 'right_index', 'forward_push', 'point'
    "brake_gesture_type": "fist",
    "handbrake_type": "peace",
    "require_manual_resume": True,
    "is_calibrated": True
}


class CalibrationManager:
    """
    Manages user calibration, notebook wheel dynamics, sensitivity presets, and configuration validation.
    """

    def __init__(self, filepath: str = CONFIG_FILE):
        self.filepath = filepath
        self.config: Dict[str, Any] = dict(DEFAULT_CONFIG)
        self.load()

        # Calibration state machine
        self.is_calibrating = False
        self.calibration_type = "quick"   # "notebook_wizard", "quick_hand", "guided_hand"
        self.calibration_phase = "idle"
        self.phase_start_time = 0.0
        self.phase_duration = 2.2         # seconds per phase
        self.calibration_warning: Optional[str] = None

        # Sample buffers
        self.samples_x: List[float] = []
        self.samples_y: List[float] = []
        self.samples_size: List[float] = []
        self.samples_angles: List[float] = []
        self.samples_pitches: List[float] = []
        self.samples_heights: List[float] = []
        self.samples_widths: List[float] = []
        self.left_angles: List[float] = []
        self.right_angles: List[float] = []
        self.brake_pitches: List[float] = []
        self.brake_heights: List[float] = []
        self.right_index_confirmed = False
        self.right_index_lowered_confirmed = False

    def load(self) -> None:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    is_valid, _ = self.validate_config(data)
                    if is_valid:
                        self.config.update(data)
                        logger.info(f"Loaded valid calibration from {self.filepath}")
            except Exception as e:
                logger.warning(f"Could not load calibration config: {e}. Using defaults.")
        else:
            self.save()

    def save(self) -> None:
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
            logger.info("Saved calibration config.")
        except Exception as e:
            logger.error(f"Failed to save calibration config: {e}")

    def reset_to_defaults(self) -> None:
        self.cancel_calibration()
        self.config = dict(DEFAULT_CONFIG)
        self.save()
        logger.info("Reset calibration and settings to factory defaults.")

    def validate_config(self, cfg: Dict[str, Any]) -> Tuple[bool, str]:
        if not isinstance(cfg, dict):
            return False, "Config must be a dictionary"
        sens = cfg.get("steering_sensitivity", 1.0)
        if not (0.2 <= sens <= 3.5):
            return False, f"Invalid steering sensitivity: {sens}"
        return True, "Valid configuration"

    def apply_preset(self, preset_key: str) -> bool:
        preset_key = preset_key.lower().strip()
        if preset_key not in PRESETS:
            return False
        preset_data = PRESETS[preset_key]
        self.config["active_preset"] = preset_key
        self.config["steering_sensitivity"] = preset_data["steering_sensitivity"]
        self.config["deadzone_x"] = preset_data["deadzone_x"]
        self.config["deadzone_angle"] = preset_data["deadzone_angle"]
        self.config["notebook_deadzone_angle"] = preset_data.get("notebook_deadzone_angle", 5.0)
        self.config["smoothing_factor"] = preset_data["smoothing_factor"]
        self.config["direction_debounce_frames"] = preset_data["direction_debounce_frames"]
        self.save()
        logger.info(f"Applied steering preset: {preset_data['preset_name']}")
        return True

    def recenter_steering(self, nb_data=None, hand_data=None) -> str:
        """
        Immediately recenters steering to the current physical notebook or hand position.
        Allows instant 1-click recalibration during gameplay without a multi-step wizard.
        """
        if nb_data is not None and nb_data.is_detected:
            self.config["notebook_neutral_angle"] = round(float(nb_data.raw_angle), 1)
            self.config["notebook_neutral_pitch"] = round(float(nb_data.raw_pitch), 1)
            self.config["notebook_neutral_height"] = round(float(nb_data.height), 1)
            self.config["notebook_neutral_width"] = round(float(nb_data.width), 1)
            self.config["notebook_calibrated"] = True
            self.save()
            msg = f"Notebook Centered! (Neutral: {nb_data.raw_angle:+.1f}°, Pitch: {nb_data.raw_pitch:+.1f}°)"
            logger.info(msg)
            return msg
        elif hand_data is not None and hand_data.is_detected:
            if hand_data.is_two_handed:
                self.config["neutral_angle"] = round(float(hand_data.wheel_angle), 1)
            self.config["neutral_x"] = round(float(hand_data.hand_center[0]), 3)
            self.config["neutral_y"] = round(float(hand_data.hand_center[1]), 3)
            self.config["is_calibrated"] = True
            self.save()
            msg = f"Hands Centered! (Center X: {hand_data.hand_center[0]:.2f})"
            logger.info(msg)
            return msg
        else:
            return "No notebook or hand visible to center"

    # -----------------------------------------------------------------
    # NOTEBOOK CALIBRATION WIZARD
    # -----------------------------------------------------------------
    def start_notebook_calibration(self) -> None:
        """
        Starts the 5-Step Guided Notebook Calibration Wizard:
        A. Hold notebook in neutral steering position (locks corners, neutral angle, pitch & size).
        B. Rotate left and right (measures comfortable steering range).
        C. Perform intended braking movement (forward tilt; checks distinguishability).
        D. Return notebook to neutral (verifies clean centering & brake release).
        E. Raise and lower right index finger (verifies gas acceleration).
        """
        self.is_calibrating = True
        self.calibration_type = "notebook_wizard"
        self.calibration_phase = "nb_neutral"
        self.calibration_warning = None
        self._clear_buffers()
        self.phase_start_time = time.time()
        logger.info("Started 5-Step Notebook Guided Calibration Wizard.")

    def update_notebook_calibration(self, nb_data, hand_data) -> Tuple[bool, str, float]:
        """
        Feed live notebook & hand telemetry into the 5-step notebook calibration wizard.
        Returns: (is_done, message, progress_ratio)
        """
        if not self.is_calibrating or self.calibration_type != "notebook_wizard":
            return False, "Not calibrating notebook", 0.0

        elapsed = time.time() - self.phase_start_time
        progress = min(1.0, elapsed / self.phase_duration)

        # -------------------------------------------------------------
        # STEP A: NEUTRAL STEERING POSITION & CORNER LOCKING
        # -------------------------------------------------------------
        if self.calibration_phase == "nb_neutral":
            if nb_data.is_detected:
                self.samples_angles.append(nb_data.raw_angle)
                self.samples_pitches.append(nb_data.raw_pitch)
                self.samples_heights.append(nb_data.height)
                self.samples_widths.append(nb_data.width)

            if elapsed >= self.phase_duration:
                if len(self.samples_angles) < 2:
                    self.is_calibrating = False
                    self.calibration_phase = "idle"
                    return False, "⚠️ Notebook not detected steady. Ensure notebook is visible.", 0.0

                avg_angle = float(np.mean(self.samples_angles))
                avg_pitch = float(np.mean(self.samples_pitches))
                avg_h = float(np.mean(self.samples_heights))
                avg_w = float(np.mean(self.samples_widths))

                self.config["notebook_neutral_angle"] = round(avg_angle, 2)
                self.config["notebook_neutral_pitch"] = round(avg_pitch, 2)
                self.config["notebook_neutral_height"] = round(avg_h, 1)
                self.config["notebook_neutral_width"] = round(avg_w, 1)
                self.config["notebook_calibrated"] = True

                self.calibration_phase = "nb_steer_range"
                self.phase_start_time = time.time()
                self._clear_buffers()
                return False, "Rotate notebook LEFT and RIGHT", 0.0

            remaining = max(0.0, self.phase_duration - elapsed)
            det_status = "✔ Locked" if nb_data.is_detected else "Searching Notebook..."
            return False, f"[Step 1/5] Hold notebook LEVEL & NEUTRAL: {remaining:.1f}s ({det_status})", progress

        # -------------------------------------------------------------
        # STEP B: STEERING RANGE (ROTATE LEFT AND RIGHT)
        # -------------------------------------------------------------
        elif self.calibration_phase == "nb_steer_range":
            if nb_data.is_detected:
                if nb_data.rotation_angle < -2.0:
                    self.left_angles.append(nb_data.rotation_angle)
                elif nb_data.rotation_angle > 2.0:
                    self.right_angles.append(nb_data.rotation_angle)

            if elapsed >= self.phase_duration:
                min_l = abs(min(self.left_angles)) if self.left_angles else 25.0
                max_r = abs(max(self.right_angles)) if self.right_angles else 25.0
                avg_range = max(18.0, min(55.0, (min_l + max_r) / 2.0))
                self.config["notebook_max_steer_angle"] = round(avg_range, 1)

                self.calibration_phase = "nb_brake_pitch"
                self.phase_start_time = time.time()
                self._clear_buffers()
                return False, "Tilt notebook FORWARD (Brake)", 0.0

            remaining = max(0.0, self.phase_duration - elapsed)
            return False, f"[Step 2/5] Rotate notebook LEFT and RIGHT: {remaining:.1f}s", progress

        # -------------------------------------------------------------
        # STEP C: BRAKING MOVEMENT (FORWARD TILT) & DISTINGUISHABILITY
        # -------------------------------------------------------------
        elif self.calibration_phase == "nb_brake_pitch":
            if nb_data.is_detected:
                self.brake_pitches.append(nb_data.raw_pitch)
                self.brake_heights.append(nb_data.height)

            if elapsed >= self.phase_duration:
                neutral_pitch = float(self.config.get("notebook_neutral_pitch", 0.0))
                neutral_h = float(self.config.get("notebook_neutral_height", 100.0))

                mean_pitch = float(np.mean(self.brake_pitches)) if self.brake_pitches else neutral_pitch
                pitch_delta = abs(mean_pitch - neutral_pitch)

                mean_h = float(np.mean(self.brake_heights)) if self.brake_heights else neutral_h
                h_drop = max(0.0, 1.0 - (mean_h / max(1.0, neutral_h)))

                # Check if camera can distinguish braking reliably
                if (pitch_delta < 5.0) and (h_drop < 0.06):
                    self.is_calibrating = False
                    self.calibration_phase = "idle"
                    self.calibration_warning = "⚠️ Braking movement too subtle to distinguish reliably."
                    return False, "⚠️ Rejected: Forward tilt too subtle. Please tilt notebook forward more distinctly.", 0.0

                # Set calibrated thresholds safely
                self.config["notebook_brake_pitch_thresh"] = max(6.0, round(pitch_delta * 0.65, 1))
                self.config["notebook_brake_height_thresh"] = max(0.06, round(h_drop * 0.60, 2))

                self.calibration_phase = "nb_return_neutral"
                self.phase_start_time = time.time()
                self._clear_buffers()
                return False, "Return notebook to NEUTRAL", 0.0

            remaining = max(0.0, self.phase_duration - elapsed)
            return False, f"[Step 3/5] Tilt notebook FORWARD (Brake): {remaining:.1f}s", progress

        # -------------------------------------------------------------
        # STEP D: RETURN TO NEUTRAL (VERIFY RELEASE)
        # -------------------------------------------------------------
        elif self.calibration_phase == "nb_return_neutral":
            if elapsed >= self.phase_duration:
                self.calibration_phase = "nb_right_index"
                self.phase_start_time = time.time()
                return False, "Raise RIGHT HAND index finger!", 0.0

            remaining = max(0.0, self.phase_duration - elapsed)
            return False, f"[Step 4/5] Return notebook to NEUTRAL position: {remaining:.1f}s", progress

        # -------------------------------------------------------------
        # STEP E: RIGHT INDEX FINGER RAISE & LOWER VERIFICATION (GAS)
        # -------------------------------------------------------------
        elif self.calibration_phase == "nb_right_index":
            if hand_data.right_hand_detected and hand_data.right_index_raised:
                self.right_index_confirmed = True

            if elapsed >= self.phase_duration:
                self.is_calibrating = False
                self.calibration_phase = "done"
                self.config["notebook_steering_enabled"] = True
                self.save()
                idx_msg = "Right Index Verified! 🚀" if self.right_index_confirmed else "Ready (Raise right index for gas)"
                return True, f"✔ Notebook Steering & Braking Calibrated! {idx_msg}", 1.0

            remaining = max(0.0, self.phase_duration - elapsed)
            stat = "✔ INDEX RAISED DETECTED" if (hand_data.right_hand_detected and hand_data.right_index_raised) else "Raise right index finger"
            return False, f"[Step 5/5] {stat}: {remaining:.1f}s", progress

        return False, "Idle", 0.0

    # -----------------------------------------------------------------
    # HAND-BASED CALIBRATION
    # -----------------------------------------------------------------
    def start_quick_calibration(self) -> None:
        self.is_calibrating = True
        self.calibration_type = "quick_hand"
        self.calibration_phase = "neutral"
        self._clear_buffers()
        self.phase_start_time = time.time()

    def start_guided_calibration(self) -> None:
        self.is_calibrating = True
        self.calibration_type = "guided_hand"
        self.calibration_phase = "neutral"
        self._clear_buffers()
        self.phase_start_time = time.time()

    def cancel_calibration(self) -> None:
        self.is_calibrating = False
        self.calibration_phase = "idle"
        self._clear_buffers()

    def _clear_buffers(self) -> None:
        self.samples_x.clear()
        self.samples_y.clear()
        self.samples_size.clear()
        self.samples_angles.clear()
        self.samples_pitches.clear()
        self.samples_heights.clear()
        self.samples_widths.clear()
        self.left_angles.clear()
        self.right_angles.clear()
        self.brake_pitches.clear()
        self.brake_heights.clear()

    def update_calibration_sample(self, hand_center: Tuple[float, float], hand_size: float) -> Tuple[bool, str, float]:
        if not self.is_calibrating or self.calibration_type == "notebook_wizard":
            return False, "Not calibrating hand", 0.0

        elapsed = time.time() - self.phase_start_time
        progress = min(1.0, elapsed / self.phase_duration)

        # ----------------- PHASE 1: NEUTRAL CENTER -----------------
        if self.calibration_phase == "neutral":
            self.samples_x.append(hand_center[0])
            self.samples_y.append(hand_center[1])
            self.samples_size.append(hand_size)

            if elapsed >= self.phase_duration:
                if self.samples_x and self.samples_size:
                    self.config["neutral_x"] = round(float(np.mean(self.samples_x)), 3)
                    self.config["neutral_y"] = round(float(np.mean(self.samples_y)), 3)
                    self.config["neutral_hand_size"] = round(float(np.mean(self.samples_size)), 3)
                    self.config["is_calibrated"] = True

                if self.calibration_type == "guided_hand":
                    self.calibration_phase = "sample_left"
                    self.phase_start_time = time.time()
                    self.samples_x.clear()
                    return False, "Reach comfortably LEFT", 0.0
                else:
                    self.is_calibrating = False
                    self.calibration_phase = "done"
                    self.save()
                    return True, "Neutral Established!", 1.0

            remaining = max(0.0, self.phase_duration - elapsed)
            step_tag = "[Step 1/3] " if self.calibration_type == "guided_hand" else ""
            return False, f"{step_tag}Hold hand neutral: {remaining:.1f}s", progress

        # ----------------- PHASE 2: COMFORTABLE LEFT -----------------
        elif self.calibration_phase == "sample_left":
            self.samples_x.append(hand_center[0])
            if elapsed >= self.phase_duration:
                if self.samples_x:
                    self.config["comfortable_min_x"] = round(float(min(self.samples_x)), 3)

                self.calibration_phase = "sample_right"
                self.phase_start_time = time.time()
                self.samples_x.clear()
                return False, "Reach comfortably RIGHT", 0.0

            remaining = max(0.0, self.phase_duration - elapsed)
            return False, f"[Step 2/3] Hold comfortable LEFT: {remaining:.1f}s", progress

        # ----------------- PHASE 3: COMFORTABLE RIGHT -----------------
        elif self.calibration_phase == "sample_right":
            self.samples_x.append(hand_center[0])
            if elapsed >= self.phase_duration:
                if self.samples_x:
                    self.config["comfortable_max_x"] = round(float(max(self.samples_x)), 3)

                nx = self.config["neutral_x"]
                left_span = abs(nx - self.config.get("comfortable_min_x", 0.25))
                right_span = abs(self.config.get("comfortable_max_x", 0.75) - nx)
                avg_span = max(0.12, (left_span + right_span) / 2.0)
                self.config["comfortable_range_x"] = round(float(avg_span), 3)

                self.is_calibrating = False
                self.calibration_phase = "done"
                self.save()
                return True, "Guided Range Calibration Complete!", 1.0

            remaining = max(0.0, self.phase_duration - elapsed)
            return False, f"[Step 3/3] Hold comfortable RIGHT: {remaining:.1f}s", progress

        return False, "Idle", 0.0

    @property
    def steering_sensitivity(self) -> float:
        return float(self.config.get("steering_sensitivity", 1.0))

    @property
    def neutral_x(self) -> float:
        return float(self.config.get("neutral_x", 0.50))

    @property
    def neutral_y(self) -> float:
        return float(self.config.get("neutral_y", 0.55))

    @property
    def neutral_size(self) -> float:
        return float(self.config.get("neutral_hand_size", 0.20))

    @property
    def deadzone_x(self) -> float:
        return float(self.config.get("deadzone_x", 0.08))

    @property
    def comfortable_range_x(self) -> float:
        return float(self.config.get("comfortable_range_x", 0.25))

    @property
    def steering_mode(self) -> str:
        return str(self.config.get("steering_mode", "notebook"))

    @property
    def notebook_steering_enabled(self) -> bool:
        return bool(self.config.get("notebook_steering_enabled", True))

    @property
    def two_hand_mode_enabled(self) -> bool:
        return bool(self.config.get("two_hand_mode_enabled", True))

    @property
    def smoothing_factor(self) -> float:
        return float(self.config.get("smoothing_factor", 0.45))

    @property
    def direction_debounce_frames(self) -> int:
        return int(self.config.get("direction_debounce_frames", 2))

    def set_steering_sensitivity(self, val: float) -> None:
        self.config["steering_sensitivity"] = max(0.2, min(3.5, val))
        self.save()

    def set_deadzone_x(self, val: float) -> None:
        self.config["deadzone_x"] = max(0.01, min(0.30, val))
        self.save()

    @property
    def notebook_deadzone_angle(self) -> float:
        return float(self.config.get("notebook_deadzone_angle", 5.0))

    def set_notebook_deadzone_angle(self, val: float) -> None:
        self.config["notebook_deadzone_angle"] = max(1.0, min(20.0, val))
        self.save()

    @property
    def notebook_max_steer_angle(self) -> float:
        return float(self.config.get("notebook_max_steer_angle", 32.0))

    def set_notebook_max_steer_angle(self, val: float) -> None:
        self.config["notebook_max_steer_angle"] = max(15.0, min(60.0, val))
        self.save()

    def set_steering_mode(self, mode: str) -> None:
        if mode in ("notebook", "position", "tilt", "two_hand_wheel"):
            self.config["steering_mode"] = mode
            self.save()

    def set_two_hand_mode(self, enabled: bool) -> None:
        self.config["two_hand_mode_enabled"] = enabled
        if enabled:
            self.config["steering_mode"] = "two_hand_wheel"
        else:
            if self.config["steering_mode"] == "two_hand_wheel":
                self.config["steering_mode"] = "notebook" if self.notebook_steering_enabled else "position"
        self.save()

    def set_notebook_steering_enabled(self, enabled: bool) -> None:
        self.config["notebook_steering_enabled"] = enabled
        self.config["steering_mode"] = "notebook" if enabled else "two_hand_wheel"
        self.save()

    def set_notebook_color_profile(self, profile_name: str) -> None:
        profiles = {
            "Orange (Default)": ([8, 65, 65], [32, 255, 255]),
            "Blue / Dark Cover": ([95, 60, 50], [130, 255, 255]),
            "Green / Bright Cover": ([35, 50, 50], [85, 255, 255]),
            "Red / Crimson Cover": ([165, 70, 50], [180, 255, 255]),
            "Any Color / Edge Adaptive": ([0, 0, 0], [180, 255, 255])
        }
        lower, upper = profiles.get(profile_name, ([8, 65, 65], [32, 255, 255]))
        self.config["notebook_color_name"] = profile_name
        self.config["notebook_hsv_lower"] = lower
        self.config["notebook_hsv_upper"] = upper
        self.save()

    @property
    def notebook_brake_pitch_thresh(self) -> float:
        return float(self.config.get("notebook_brake_pitch_thresh", 9.0))

    def set_notebook_brake_pitch_thresh(self, val: float) -> None:
        self.config["notebook_brake_pitch_thresh"] = max(3.0, min(30.0, val))
        self.save()

    @property
    def notebook_brake_height_thresh(self) -> float:
        return float(self.config.get("notebook_brake_height_thresh", 0.10))

    def set_notebook_brake_height_thresh(self, val: float) -> None:
        self.config["notebook_brake_height_thresh"] = max(0.03, min(0.35, val))
        self.save()

    @property
    def notebook_brake_sensitivity(self) -> float:
        return float(self.config.get("notebook_brake_sensitivity", 1.0))

    def set_notebook_brake_sensitivity(self, val: float) -> None:
        self.config["notebook_brake_sensitivity"] = max(0.2, min(3.0, val))
        self.save()

