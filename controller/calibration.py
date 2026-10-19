"""
Calibration manager for steering and gesture sensitivity.
Persists settings to JSON and provides an interactive calibration routine.
"""

import os
import json
import time
import math
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("burnout.calibration")

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration_config.json")

DEFAULT_CONFIG: Dict[str, Any] = {
    "steering_mode": "position",  # 'position' (horizontal offset) or 'tilt' (steering wheel angle)
    "neutral_x": 0.50,
    "neutral_y": 0.55,
    "neutral_hand_size": 0.20,
    "deadzone_x": 0.08,
    "deadzone_angle": 12.0,       # degrees for tilt mode
    "steering_sensitivity": 1.0,
    "accel_forward_threshold": 0.04,  # hand size increase to trigger forward acceleration
    "accel_y_threshold": 0.12,        # or pushing hand forward/up
    "max_steer_angle": 45.0,
    "is_calibrated": True
}


class CalibrationManager:
    """
    Handles user calibration: neutral hand center, range of motion, dead zones, and persistence.
    """

    def __init__(self, filepath: str = CONFIG_FILE):
        self.filepath = filepath
        self.config: Dict[str, Any] = dict(DEFAULT_CONFIG)
        self.load()

        # Calibration state machine
        self.is_calibrating = False
        self.calibration_phase = "idle"  # idle, neutral, left, right, done
        self.samples_x = []
        self.samples_y = []
        self.samples_size = []
        self.phase_start_time = 0.0
        self.phase_duration = 2.5  # seconds per phase

    def load(self) -> None:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.config.update(data)
                logger.info(f"Loaded calibration from {self.filepath}")
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

    def start_quick_calibration(self) -> None:
        """Start a simple 3-second neutral pose capture."""
        self.is_calibrating = True
        self.calibration_phase = "neutral"
        self.samples_x.clear()
        self.samples_y.clear()
        self.samples_size.clear()
        self.phase_start_time = time.time()
        logger.info("Started quick calibration: hold hand steady in neutral steering position.")

    def cancel_calibration(self) -> None:
        self.is_calibrating = False
        self.calibration_phase = "idle"
        self.samples_x.clear()
        self.samples_y.clear()
        self.samples_size.clear()

    def update_calibration_sample(self, hand_center: Tuple[float, float], hand_size: float) -> Tuple[bool, str, float]:
        """
        Feed a landmark sample during calibration.
        Returns: (completed: bool, message: str, progress: float [0..1])
        """
        if not self.is_calibrating:
            return False, "Not calibrating", 0.0

        elapsed = time.time() - self.phase_start_time
        progress = min(1.0, elapsed / self.phase_duration)

        if self.calibration_phase == "neutral":
            self.samples_x.append(hand_center[0])
            self.samples_y.append(hand_center[1])
            self.samples_size.append(hand_size)

            if elapsed >= self.phase_duration:
                # Finished sampling neutral
                if self.samples_x and self.samples_size:
                    avg_x = sum(self.samples_x) / len(self.samples_x)
                    avg_y = sum(self.samples_y) / len(self.samples_y)
                    avg_sz = sum(self.samples_size) / len(self.samples_size)

                    self.config["neutral_x"] = round(avg_x, 3)
                    self.config["neutral_y"] = round(avg_y, 3)
                    self.config["neutral_hand_size"] = round(avg_sz, 3)
                    self.config["is_calibrated"] = True
                    self.save()

                self.is_calibrating = False
                self.calibration_phase = "done"
                return True, "Calibration Complete! Neutral established.", 1.0

            remaining = max(0.0, self.phase_duration - elapsed)
            return False, f"Hold hand neutral: {remaining:.1f}s", progress

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
    def deadzone_x(self) -> float:
        return float(self.config.get("deadzone_x", 0.08))

    @property
    def steering_sensitivity(self) -> float:
        return float(self.config.get("steering_sensitivity", 1.0))

    @property
    def steering_mode(self) -> str:
        return str(self.config.get("steering_mode", "position"))

    def set_steering_sensitivity(self, val: float) -> None:
        self.config["steering_sensitivity"] = max(0.2, min(3.0, val))
        self.save()

    def set_deadzone_x(self, val: float) -> None:
        self.config["deadzone_x"] = max(0.01, min(0.30, val))
        self.save()

    def set_steering_mode(self, mode: str) -> None:
        if mode in ("position", "tilt"):
            self.config["steering_mode"] = mode
            self.save()
