"""
Keyboard simulation module for Windows.
Uses DirectInput scan codes via Windows SendInput for 100% compatibility with
hardware-accelerated browser games (like Unity WebGL on Poki), with fallbacks
to pydirectinput and pyautogui.

Includes strict safety mechanisms:
- Active key tracking (no stuck keys)
- Idempotent press/release (no repeated down events)
- Immediate release_all on stop, hand loss, emergency stop, or window blur
- Thread-safe operations
"""

import sys
import time
import threading
import logging
from typing import Set, Dict, Optional, List

logger = logging.getLogger("burnout.keyboard")

# DirectInput scan codes for common keys
SCAN_CODES = {
    # Arrow keys (require extended key flag 0x0001)
    "up": (0x48, True),
    "down": (0x50, True),
    "left": (0x4B, True),
    "right": (0x4D, True),
    # WASD controls
    "w": (0x11, False),
    "s": (0x1F, False),
    "a": (0x1E, False),
    "d": (0x20, False),
    # Common actions
    "space": (0x39, False),
    "escape": (0x01, False),
    "enter": (0x1C, False),
    "shift": (0x2A, False),
    "ctrl": (0x1D, False),
}

# Windows API constants and ctypes structures
IS_WINDOWS = sys.platform.startswith("win")
if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    PUL = ctypes.POINTER(ctypes.c_ulong)

    class KeyBdInput(ctypes.Structure):
        _fields_ = [
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", PUL),
        ]

    class HardwareInput(ctypes.Structure):
        _fields_ = [
            ("uMsg", wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        ]

    class MouseInput(ctypes.Structure):
        _fields_ = [
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", PUL),
        ]

    class Input_I(ctypes.Union):
        _fields_ = [
            ("ki", KeyBdInput),
            ("mi", MouseInput),
            ("hi", HardwareInput),
        ]

    class Input(ctypes.Structure):
        _fields_ = [
            ("type", wintypes.DWORD),
            ("ii", Input_I),
        ]

    INPUT_KEYBOARD = 1
    KEYEVENTF_EXTENDEDKEY = 0x0001
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_SCANCODE = 0x0008

    user32 = ctypes.windll.user32
else:
    user32 = None


# Fallback to pyautogui or pydirectinput if needed
pyautogui_available = False
try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.0
    pyautogui_available = True
except ImportError:
    pass

pydirectinput_available = False
try:
    import pydirectinput
    pydirectinput.FAILSAFE = False
    pydirectinput.PAUSE = 0.0
    pydirectinput_available = True
except ImportError:
    pass


def get_foreground_window_title() -> str:
    """Returns the title of the current active Windows window."""
    if not IS_WINDOWS or not user32:
        return ""
    try:
        hwnd = user32.GetForegroundWindow()
        length = user32.GetWindowTextLengthW(hwnd)
        if length > 0:
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            return buff.value
    except Exception as e:
        logger.debug(f"Error fetching active window title: {e}")
    return ""


class KeyboardController:
    """
    Manages safe keyboard inputs with tracking and emergency controls.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._active_keys: Set[str] = set()
        self._is_enabled = False
        self._emergency_stopped = False
        self._key_mapping: Dict[str, str] = {
            "steer_left": "left",
            "steer_right": "right",
            "accelerate": "up",
            "brake": "down",
            "handbrake": "space",
        }
        self.check_focus = False
        self.allowed_focus_keywords = ["Burnout", "Poki", "Chrome", "Edge", "Firefox", "Opera", "Brave"]

    def set_mapping(self, mapping: Dict[str, str]) -> None:
        """Update action-to-key mapping (e.g. {'steer_left': 'a', ...})."""
        with self._lock:
            self.release_all_internal()
            self._key_mapping.update(mapping)

    def get_mapping(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._key_mapping)

    def is_enabled(self) -> bool:
        return self._is_enabled and not self._emergency_stopped

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable driving input sending."""
        with self._lock:
            if not enabled and self._is_enabled:
                self.release_all_internal()
            if enabled:
                self._emergency_stopped = False
            self._is_enabled = enabled
        logger.info(f"KeyboardController enabled state: {self._is_enabled}")

    def emergency_stop(self) -> None:
        """Immediately release all keys and disable inputs until restarted."""
        with self._lock:
            self._emergency_stopped = True
            self._is_enabled = False
            self.release_all_internal()
        logger.warning("EMERGENCY STOP TRIGGERED: All keys released and inputs locked.")

    def reset_emergency_stop(self) -> None:
        with self._lock:
            self._emergency_stopped = False
            self.release_all_internal()

    def get_active_keys(self) -> Set[str]:
        """Return a copy of currently pressed keys."""
        with self._lock:
            return set(self._active_keys)

    def _send_win_key(self, scan_code: int, extended: bool, is_down: bool) -> bool:
        """Send DirectInput scancode via ctypes SendInput."""
        if not IS_WINDOWS or not user32:
            return False
        try:
            flags = KEYEVENTF_SCANCODE
            if extended:
                flags |= KEYEVENTF_EXTENDEDKEY
            if not is_down:
                flags |= KEYEVENTF_KEYUP

            extra = ctypes.c_ulong(0)
            ii_ = Input_I()
            ii_.ki = KeyBdInput(0, scan_code, flags, 0, ctypes.pointer(extra))
            x = Input(ctypes.c_ulong(INPUT_KEYBOARD), ii_)
            user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
            return True
        except Exception as e:
            logger.error(f"SendInput error: {e}")
            return False

    def _raw_press(self, key_name: str) -> None:
        key_name = key_name.lower().strip()
        if key_name in SCAN_CODES:
            code, ext = SCAN_CODES[key_name]
            if self._send_win_key(code, ext, is_down=True):
                return

        # Fallback to pydirectinput
        if pydirectinput_available:
            try:
                pydirectinput.keyDown(key_name)
                return
            except Exception:
                pass

        # Fallback to pyautogui
        if pyautogui_available:
            try:
                pyautogui.keyDown(key_name)
            except Exception as e:
                logger.error(f"pyautogui keyDown failed for {key_name}: {e}")

    def _raw_release(self, key_name: str) -> None:
        key_name = key_name.lower().strip()
        if key_name in SCAN_CODES:
            code, ext = SCAN_CODES[key_name]
            if self._send_win_key(code, ext, is_down=False):
                return

        # Fallback to pydirectinput
        if pydirectinput_available:
            try:
                pydirectinput.keyUp(key_name)
                return
            except Exception:
                pass

        # Fallback to pyautogui
        if pyautogui_available:
            try:
                pyautogui.keyUp(key_name)
            except Exception as e:
                logger.error(f"pyautogui keyUp failed for {key_name}: {e}")

    def release_all_internal(self) -> None:
        """Internal helper to release all active keys without extra locking."""
        if not self._active_keys:
            return
        keys_to_release = list(self._active_keys)
        for key in keys_to_release:
            try:
                self._raw_release(key)
            except Exception as e:
                logger.error(f"Error releasing key '{key}': {e}")
        self._active_keys.clear()

    def release_all(self) -> None:
        """Public thread-safe release of all active keys."""
        with self._lock:
            self.release_all_internal()

    def check_focus_safety(self) -> bool:
        """Returns True if the target window is focused or if focus checking is disabled."""
        if not self.check_focus:
            return True
        title = get_foreground_window_title().lower()
        if not title:
            return True
        return any(keyword.lower() in title for keyword in self.allowed_focus_keywords)

    def apply_actions(self, desired_actions: Dict[str, bool]) -> None:
        """
        Takes a dict of desired actions, e.g.:
        {
            'steer_left': False,
            'steer_right': True,
            'accelerate': True,
            'brake': False,
            'handbrake': False
        }
        Applies changes smoothly and safely.
        """
        with self._lock:
            if not self._is_enabled or self._emergency_stopped:
                self.release_all_internal()
                return

            if self.check_focus and not self.check_focus_safety():
                self.release_all_internal()
                return

            # Determine which keys should currently be pressed
            desired_keys: Set[str] = set()
            for action_name, is_active in desired_actions.items():
                if is_active and action_name in self._key_mapping:
                    key = self._key_mapping[action_name].lower().strip()
                    desired_keys.add(key)

            # Mutual exclusivity safety check:
            # Cannot have both accelerate and brake active simultaneously
            accel_key = self._key_mapping.get("accelerate", "up").lower()
            brake_key = self._key_mapping.get("brake", "down").lower()
            if accel_key in desired_keys and brake_key in desired_keys:
                # Braking takes priority over acceleration for safety
                desired_keys.remove(accel_key)

            # Steer left and steer right cannot be both active
            left_key = self._key_mapping.get("steer_left", "left").lower()
            right_key = self._key_mapping.get("steer_right", "right").lower()
            if left_key in desired_keys and right_key in desired_keys:
                desired_keys.remove(left_key)
                desired_keys.remove(right_key)

            # Keys to release: currently active but not in desired
            keys_to_release = self._active_keys - desired_keys
            for key in keys_to_release:
                self._raw_release(key)
                self._active_keys.remove(key)

            # Keys to press: in desired but not currently active
            keys_to_press = desired_keys - self._active_keys
            for key in keys_to_press:
                self._raw_press(key)
                self._active_keys.add(key)

    def shutdown(self) -> None:
        """Cleanup on application exit."""
        self.emergency_stop()
