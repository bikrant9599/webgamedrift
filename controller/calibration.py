"""
Smart calibration manager for Burnout Camera Drive.
Handles neutral hand center detection, comfortable steering range calibration,
deadzones, smoothing factors, sensitivity presets (Beginner, Normal, Pro),
settings validation, and JSON persistence.
"""

import os
import json
import time
import math
import logging
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger("burnout.calibration")

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration_config.json")

PRESETS: Dict[str, Dict[str, Any]] = {
    "beginner": {
        "steering_sensitivity": 0.8,
        "deadzone_x": 0.12,
        "deadzone_angle": 16.0,
        "smoothing_factor": 0.30,
        "direction_debounce_frames": 3,
        "preset_name": "Beginner (Smooth & Stable)"
    },
    "normal": {
        "steering_sensitivity": 1.0,
        "deadzone_x": 0.08,
        "deadzone_angle": 12.0,
        "smoothing_factor": 0.45,
        "direction_debounce_frames": 2,
        "preset_name": "Normal (Balanced Drift)"
    },
    "pro": {
        "steering_sensitivity": 1.5,
        "deadzone_x": 0.04,
        "deadzone_angle": 8.0,
        "smoothing_factor": 0.65,
        "direction_debounce_frames": 1,
        "preset_name": "Pro (Ultra Responsive)"
    }
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "active_preset": "normal",
    "steering_mode": "position",  # 'position', 'tilt', or 'two_hand_wheel'
    "two_hand_mode_enabled": False,
    "neutral_x": 0.50,
    "neutral_y": 0.55,
    "neutral_hand_size": 0.20,
    "comfortable_range_x": 0.25,  # span from neutral to max left/right
    "comfortable_min_x": 0.25,
    "comfortable_max_x": 0.75,
    "deadzone_x": 0.08,
    "deadzone_angle": 12.0,
    "steering_sensitivity": 1.0,
    "smoothing_factor": 0.45,
    "direction_debounce_frames": 2,
    "accel_forward_threshold": 0.15,
    "accel_y_threshold": 0.10,
    "max_steer_angle": 45.0,
    "handbrake_type": "peace",    # 'peace', 'shaka', 'pinch', 'pinky'
    "accel_mode": "forward_push", # 'forward_push', 'forward_y', 'point'
    "brake_gesture_type": "fist", # 'fist', 'thumbs_down'
    "gesture_confirm_frames": 2,
    "require_manual_resume": True,
    "is_calibrated": True
}


class CalibrationManager:
    """
    Manages user calibration, steering range, sensitivity presets, and configuration validation.
    """

    def __init__(self, filepath: str = CONFIG_FILE):
        self.filepath = filepath
        self.config: Dict[str, Any] = dict(DEFAULT_CONFIG)
        self.load()

        # Calibration state machine
        self.is_calibrating = False
        self.calibration_type = "quick"   # "quick" (neutral only) or "guided" (neutral -> left -> right)
        self.calibration_phase = "idle"   # idle, neutral, sample_left, sample_right, done
        self.phase_start_time = 0.0
        self.phase_duration = 2.2         # seconds per phase

        # Sample buffers
        self.samples_x: List[float] = []
        self.samples_y: List[float] = []
        self.samples_size: List[float] = []
        self.left_samples: List[float] = []
        self.right_samples: List[float] = []

    def load(self) -> None:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    is_valid, msg = self.validate_config(data)
                    if is_valid:
                        self.config.update(data)
                        logger.info(f"Loaded valid calibration from {self.filepath}")
                    else:
                        logger.warning(f"Config validation warning: {msg}. Using safe defaults.")
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
        """Reset all parameters to clean safe defaults."""
        self.cancel_calibration()
        self.config = dict(DEFAULT_CONFIG)
        self.save()
        logger.info("Reset calibration and settings to factory defaults.")

    def validate_config(self, cfg: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate configuration parameters for consistency and safety."""
        if not isinstance(cfg, dict):
            return False, "Config must be a dictionary"

        # Check numeric ranges
        sens = cfg.get("steering_sensitivity", 1.0)
        if not (0.2 <= sens <= 3.5):
            return False, f"Invalid steering sensitivity: {sens} (must be 0.2 - 3.5)"

        dz = cfg.get("deadzone_x", 0.08)
        if not (0.01 <= dz <= 0.35):
            return False, f"Invalid deadzone_x: {dz} (must be 0.01 - 0.35)"

        nx = cfg.get("neutral_x", 0.50)
        if not (0.10 <= nx <= 0.90):
            return False, f"Invalid neutral_x: {nx} (must be 0.10 - 0.90)"

        return True, "Valid configuration"

    def apply_preset(self, preset_key: str) -> bool:
        """Apply one of the built-in sensitivity presets: beginner, normal, pro."""
        preset_key = preset_key.lower().strip()
        if preset_key not in PRESETS:
            return False

        preset_data = PRESETS[preset_key]
        self.config["active_preset"] = preset_key
        self.config["steering_sensitivity"] = preset_data["steering_sensitivity"]
        self.config["deadzone_x"] = preset_data["deadzone_x"]
        self.config["deadzone_angle"] = preset_data["deadzone_angle"]
        self.config["smoothing_factor"] = preset_data["smoothing_factor"]
        self.config["direction_debounce_frames"] = preset_data["direction_debounce_frames"]
        self.save()
        logger.info(f"Applied steering preset: {preset_data['preset_name']}")
        return True

    def start_quick_calibration(self) -> None:
        """Start a single-step 2.5-second neutral center calibration."""
        self.is_calibrating = True
        self.calibration_type = "quick"
        self.calibration_phase = "neutral"
        self._clear_buffers()
        self.phase_start_time = time.time()
        logger.info("Started quick neutral calibration.")

    def start_guided_calibration(self) -> None:
        """Start a 3-step smart calibration: Neutral -> Comfortable Left -> Comfortable Right."""
        self.is_calibrating = True
        self.calibration_type = "guided"
        self.calibration_phase = "neutral"
        self._clear_buffers()
        self.phase_start_time = time.time()
        logger.info("Started full smart guided calibration.")

    def cancel_calibration(self) -> None:
        self.is_calibrating = False
        self.calibration_phase = "idle"
        self._clear_buffers()
        logger.info("Calibration cancelled.")

    def _clear_buffers(self) -> None:
        self.samples_x.clear()
        self.samples_y.clear()
        self.samples_size.clear()
        self.left_samples.clear()
        self.right_samples.clear()

    def update_calibration_sample(self, hand_center: Tuple[float, float], hand_size: float) -> Tuple[bool, str, float]:
        """
        Feed a landmark sample during calibration.
        Returns: (completed: bool, message: str, progress: float [0..1])
        """
        if not self.is_calibrating:
            return False, "Not calibrating", 0.0

        elapsed = time.time() - self.phase_start_time
        progress = min(1.0, elapsed / self.phase_duration)

        # ----------------- PHASE 1: NEUTRAL CENTER -----------------
        if self.calibration_phase == "neutral":
            self.samples_x.append(hand_center[0])
            self.samples_y.append(hand_center[1])
            self.samples_size.append(hand_size)

            if elapsed >= self.phase_duration:
                if self.samples_x and self.samples_size:
                    avg_x = sum(self.samples_x) / len(self.samples_x)
                    avg_y = sum(self.samples_y) / len(self.samples_y)
                    avg_sz = sum(self.samples_size) / len(self.samples_size)

                    self.config["neutral_x"] = round(avg_x, 3)
                    self.config["neutral_y"] = round(avg_y, 3)
                    self.config["neutral_hand_size"] = round(avg_sz, 3)
                    self.config["is_calibrated"] = True

                if self.calibration_type == "guided":
                    # Transition to Left Range
                    self.calibration_phase = "sample_left"
                    self.phase_start_time = time.time()
                    self.samples_x.clear()
                    return False, "Reach comfortably LEFT", 0.0
                else:
                    # Quick calibration complete
                    self.is_calibrating = False
                    self.calibration_phase = "done"
                    self.save()
                    return True, "Neutral Established!", 1.0

            remaining = max(0.0, self.phase_duration - elapsed)
            step_tag = "[Step 1/3] " if self.calibration_type == "guided" else ""
            return False, f"{step_tag}Hold hand neutral: {remaining:.1f}s", progress

        # ----------------- PHASE 2: COMFORTABLE LEFT -----------------
        elif self.calibration_phase == "sample_left":
            self.left_samples.append(hand_center[0])
            if elapsed >= self.phase_duration:
                if self.left_samples:
                    min_x = min(self.left_samples)
                    self.config["comfortable_min_x"] = round(min_x, 3)

                # Transition to Right Range
                self.calibration_phase = "sample_right"
                self.phase_start_time = time.time()
                self.samples_x.clear()
                return False, "Reach comfortably RIGHT", 0.0

            remaining = max(0.0, self.phase_duration - elapsed)
            return False, f"[Step 2/3] Hold comfortable LEFT: {remaining:.1f}s", progress

        # ----------------- PHASE 3: COMFORTABLE RIGHT -----------------
        elif self.calibration_phase == "sample_right":
            self.right_samples.append(hand_center[0])
            if elapsed >= self.phase_duration:
                if self.right_samples:
                    max_x = max(self.right_samples)
                    self.config["comfortable_max_x"] = round(max_x, 3)

                # Compute comfortable span
                nx = self.config["neutral_x"]
                left_span = abs(nx - self.config.get("comfortable_min_x", 0.25))
                right_span = abs(self.config.get("comfortable_max_x", 0.75) - nx)
                avg_span = max(0.12, (left_span + right_span) / 2.0)
                self.config["comfortable_range_x"] = round(avg_span, 3)

                self.is_calibrating = False
                self.calibration_phase = "done"
                self.save()
                return True, "Guided Range Calibration Complete!", 1.0

            remaining = max(0.0, self.phase_duration - elapsed)
            return False, f"[Step 3/3] Hold comfortable RIGHT: {remaining:.1f}s", progress

        return False, "Idle", 0.0

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
    def comfortable_range_x(self) -> float:
        return float(self.config.get("comfortable_range_x", 0.25))

    @property
    def deadzone_x(self) -> float:
        return float(self.config.get("deadzone_x", 0.08))

    @property
    def steering_sensitivity(self) -> float:
        return float(self.config.get("steering_sensitivity", 1.0))

    @property
    def steering_mode(self) -> str:
        return str(self.config.get("steering_mode", "position"))

    @property
    def two_hand_mode_enabled(self) -> bool:
        return bool(self.config.get("two_hand_mode_enabled", False))

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

    def set_steering_mode(self, mode: str) -> None:
        if mode in ("position", "tilt", "two_hand_wheel"):
            self.config["steering_mode"] = mode
            self.save()

    def set_two_hand_mode(self, enabled: bool) -> None:
        self.config["two_hand_mode_enabled"] = enabled
        if enabled:
            self.config["steering_mode"] = "two_hand_wheel"
        else:
            if self.config["steering_mode"] == "two_hand_wheel":
                self.config["steering_mode"] = "position"
        self.save()
