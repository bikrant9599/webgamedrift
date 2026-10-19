"""
Session statistics and local telemetry logger for Burnout Camera Drive.
Tracks session duration, tracking interruptions, driving time, and practice vs live modes.
All data is strictly stored locally in JSON format; no camera frames or personal data are ever saved or uploaded.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("burnout.session_stats")

HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "session_history.json")


class SessionTracker:
    """
    Tracks real-time telemetry metrics for the active driving session and saves summaries locally.
    """

    def __init__(self, history_file: str = HISTORY_FILE):
        self.history_file = history_file

        # Active session state
        self.is_session_active = False
        self.start_time: float = 0.0
        self.total_elapsed: float = 0.0
        self.active_driving_time: float = 0.0
        self._driving_start_mark: Optional[float] = None

        # Interruptions
        self.interruption_count: int = 0
        self.last_interruption_reason: str = "None"
        self._was_tracking = False

        # Action counters
        self.throttle_time: float = 0.0
        self.brake_count: int = 0
        self.handbrake_count: int = 0

        # Current mode: "live" or "practice"
        self.current_mode: str = "live"

    def start_session(self, mode: str = "live") -> None:
        """Start or restart a driving session."""
        self.is_session_active = True
        self.start_time = time.time()
        self.total_elapsed = 0.0
        self.active_driving_time = 0.0
        self._driving_start_mark = None
        self.interruption_count = 0
        self.last_interruption_reason = "None"
        self._was_tracking = False
        self.throttle_time = 0.0
        self.brake_count = 0
        self.handbrake_count = 0
        self.current_mode = mode
        logger.info(f"Started new session in mode: {mode}")

    def end_session(self) -> Optional[Dict[str, Any]]:
        """End the current session and append summary to local history."""
        if not self.is_session_active:
            return None

        self.total_elapsed = time.time() - self.start_time
        self.is_session_active = False

        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": self.current_mode,
            "duration_seconds": round(self.total_elapsed, 1),
            "driving_time_seconds": round(self.active_driving_time, 1),
            "interruptions": self.interruption_count,
            "brakes_applied": self.brake_count,
            "drifts_initiated": self.handbrake_count
        }

        self._save_to_history(summary)
        return summary

    def record_tracking_status(self, is_tracking: bool, is_driving: bool) -> None:
        """Record tracking state transitions and log interruptions during active driving."""
        if not self.is_session_active:
            return

        if self._was_tracking and not is_tracking and is_driving:
            self.interruption_count += 1
            self.last_interruption_reason = "Hand left camera view"
            logger.info(f"Tracking interruption recorded. Total: {self.interruption_count}")

        self._was_tracking = is_tracking

    def record_driving_state(self, is_driving: bool) -> None:
        """Track active driving time."""
        now = time.time()
        if is_driving:
            if self._driving_start_mark is None:
                self._driving_start_mark = now
        else:
            if self._driving_start_mark is not None:
                self.active_driving_time += now - self._driving_start_mark
                self._driving_start_mark = None

    def record_actions(self, throttle: bool, brake: bool, handbrake: bool) -> None:
        """Record discrete events."""
        if brake:
            self.brake_count += 1
        if handbrake:
            self.handbrake_count += 1

    def get_live_stats(self) -> Dict[str, Any]:
        """Return metrics for UI telemetry."""
        if not self.is_session_active:
            return {
                "elapsed_str": "00:00:00",
                "interruptions": self.interruption_count,
                "driving_time_str": "00:00:00",
                "mode": self.current_mode
            }

        elapsed = time.time() - self.start_time
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        elapsed_str = f"{h:02d}:{m:02d}:{s:02d}"

        d_time = self.active_driving_time
        if self._driving_start_mark:
            d_time += (time.time() - self._driving_start_mark)
        dm, ds = divmod(int(d_time), 60)
        dh, dm = divmod(dm, 60)
        driving_str = f"{dh:02d}:{dm:02d}:{ds:02d}"

        return {
            "elapsed_str": elapsed_str,
            "elapsed_seconds": elapsed,
            "interruptions": self.interruption_count,
            "driving_time_str": driving_str,
            "mode": self.current_mode,
            "last_reason": self.last_interruption_reason
        }

    def _save_to_history(self, summary: Dict[str, Any]) -> None:
        """Append to local history file, retaining last 25 sessions."""
        try:
            history = self.load_history()
            history.insert(0, summary)
            history = history[:25]  # keep max 25

            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=4)
            logger.info("Session saved to local history.")
        except Exception as e:
            logger.error(f"Failed to save session history: {e}")

    def load_history(self) -> List[Dict[str, Any]]:
        """Load session history from local JSON file."""
        if not os.path.exists(self.history_file):
            return []
        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read session history: {e}")
            return []

    def clear_history(self) -> None:
        """Clear session history."""
        try:
            if os.path.exists(self.history_file):
                os.remove(self.history_file)
        except Exception:
            pass
