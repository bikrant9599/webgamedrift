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
    "notebook_deadzone_angle": 5.0,     # degrees of deadzone for notebook
    "notebook_max_steer_angle": 32.0,   # degrees for 100% steering deflection
    "notebook_hsv_lower": [8, 65, 65],   # Orange notebook bounds
    "notebook_hsv_upper": [32, 255, 255],
    "notebook_color_name": "Orange",
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

        # Sample buffers
        self.samples_x: List[float] = []
        self.samples_y: List[float] = []
        self.samples_size: List[float] = []
        self.samples_angles: List[float] = []
        self.left_angles: List[float] = []
        self.right_angles: List[float] = []
        self.right_index_confirmed = False

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

    # -----------------------------------------------------------------
    # NOTEBOOK CALIBRATION WIZARD
    # -----------------------------------------------------------------
    def start_notebook_calibration(self) -> None:
        """
        Starts the Notebook Calibration Wizard:
        1. Hold notebook horizontal steady (locks 4 corners & neutral angle, samples color).
        2. Rotate slightly left (measures left range).
        3. Rotate slightly right (measures right range).
        4. Raise right index finger (verifies forward acceleration).
        """
        self.is_calibrating = True
        self.calibration_type = "notebook_wizard"
        self.calibration_phase = "nb_neutral"
        self._clear_buffers()
        self.phase_start_time = time.time()
        logger.info("Started Notebook Calibration Wizard.")

    def update_notebook_calibration(self, nb_data, hand_data) -> Tuple[bool, str, float]:
        """
        Feed live notebook & hand telemetry into the notebook calibration wizard.
        Returns: (is_done, message, progress_ratio)
        """
        if not self.is_calibrating or self.calibration_type != "notebook_wizard":
            return False, "Not calibrating notebook", 0.0

        elapsed = time.time() - self.phase_start_time
        progress = min(1.0, elapsed / self.phase_duration)

        # 1. Neutral Position & Corner Locking
        if self.calibration_phase == "nb_neutral":
            if nb_data.is_detected:
                self.samples_angles.append(nb_data.raw_angle)

            if elapsed >= self.phase_duration:
                if self.samples_angles:
                    avg_angle = float(np.mean(self.samples_angles))
                    self.config["notebook_neutral_angle"] = round(avg_angle, 2)
                    self.config["notebook_calibrated"] = True

                self.calibration_phase = "nb_rot_left"
                self.phase_start_time = time.time()
                self._clear_buffers()
                return False, "Rotate notebook LEFT slightly", 0.0

            remaining = max(0.0, self.phase_duration - elapsed)
            det_status = "Locked" if nb_data.is_detected else "Searching Notebook..."
            return False, f"[1/4] Hold notebook LEVEL: {remaining:.1f}s ({det_status})", progress

        # 2. Comfortable Left Tilt
        elif self.calibration_phase == "nb_rot_left":
            if nb_data.is_detected:
                self.left_angles.append(nb_data.rotation_angle)

            if elapsed >= self.phase_duration:
                self.calibration_phase = "nb_rot_right"
                self.phase_start_time = time.time()
                return False, "Rotate notebook RIGHT slightly", 0.0

            remaining = max(0.0, self.phase_duration - elapsed)
            return False, f"[2/4] Rotate notebook LEFT: {remaining:.1f}s", progress

        # 3. Comfortable Right Tilt
        elif self.calibration_phase == "nb_rot_right":
            if nb_data.is_detected:
                self.right_angles.append(nb_data.rotation_angle)

            if elapsed >= self.phase_duration:
                min_l = abs(min(self.left_angles)) if self.left_angles else 25.0
                max_r = abs(max(self.right_angles)) if self.right_angles else 25.0
                avg_range = max(18.0, min(55.0, (min_l + max_r) / 2.0))
                self.config["notebook_max_steer_angle"] = round(avg_range, 1)

                self.calibration_phase = "nb_right_index"
                self.phase_start_time = time.time()
                return False, "Raise RIGHT HAND index finger!", 0.0

            remaining = max(0.0, self.phase_duration - elapsed)
            return False, f"[3/4] Rotate notebook RIGHT: {remaining:.1f}s", progress

        # 4. Verify Right-Hand Raised Index Finger
        elif self.calibration_phase == "nb_right_index":
            if hand_data.right_hand_detected and hand_data.right_index_raised:
                self.right_index_confirmed = True

            if elapsed >= self.phase_duration:
                self.is_calibrating = False
                self.calibration_phase = "done"
                self.config["notebook_steering_enabled"] = True
                self.save()
                idx_msg = "Right Index Verified! 🚀" if self.right_index_confirmed else "Ready (Ensure right index raised for gas)"
                return True, f"Notebook Calibrated! {idx_msg}", 1.0

            remaining = max(0.0, self.phase_duration - elapsed)
            stat = "✔ INDEX RAISED DETECTED" if (hand_data.right_hand_detected and hand_data.right_index_raised) else "Raise right index finger"
            return False, f"[4/4] {stat}: {remaining:.1f}s", progress

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
        self.left_angles.clear()
        self.right_angles.clear()

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
