"""
Keyboard simulation module for Windows.
Combines Windows keybd_event and DirectInput SendInput for 100% compatibility with
hardware-accelerated browser games (like Unity WebGL on Poki via Microsoft Edge or Chrome).

Includes strict safety mechanisms:
- Active key tracking (no stuck keys)
- Idempotent press/release (no repeated down events)
- Immediate release_all on stop, hand loss, emergency stop, or window blur
- Practice Mode: visual gesture telemetry without sending keys to Windows
- Test Input Mode: safe testing in dedicated UI text field
- Manual Re-Arm Safety: never automatically resumes throttle after tracking loss
- Window Focus Detection: strictly excludes controller window and ensures browser game is active
- Thread-safe operations
"""

import sys
import time
import threading
import logging
from typing import Set, Dict, Optional, List, Tuple

logger = logging.getLogger("burnout.keyboard")

# DirectInput scan codes for common keys
SCAN_CODES = {
    # Arrow keys (extended keys)
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

VK_CODES = {
    "up": 0x26,
    "down": 0x28,
    "left": 0x25,
    "right": 0x27,
    "w": 0x57,
    "s": 0x53,
    "a": 0x41,
    "d": 0x44,
    "space": 0x20,
    "escape": 0x1B,
    "enter": 0x0D,
    "shift": 0x10,
    "ctrl": 0x11,
}

SCAN_TO_VK = {
    (0x48, True): 0x26,   # up
    (0x50, True): 0x28,   # down
    (0x4B, True): 0x25,   # left
    (0x4D, True): 0x27,   # right
    (0x11, False): 0x57,  # w
    (0x1F, False): 0x53,  # s
    (0x1E, False): 0x41,  # a
    (0x20, False): 0x44,  # d
    (0x39, False): 0x20,  # space
    (0x01, False): 0x1B,  # escape
}

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
    Manages safe keyboard inputs with active tracking, emergency controls, practice mode,
    test mode, and strict browser game window focus filtering.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._active_keys: Set[str] = set()
        self._is_enabled = False
        self._emergency_stopped = False

        # Safety: Interruption & Re-Arm State
        self.require_manual_resume = True
        self.is_interrupted = False
        self.interruption_reason = ""

        # Practice Mode: Simulates inputs for UI badges without sending real keys to OS
        self.practice_mode = False

        # Test Mode: Safely test inputs into application test box before game start
        self.test_mode = False

        # Last simulated keyboard event for diagnostics
        self.last_event: str = "NONE"
        self.last_event_time: float = 0.0

        self._key_mapping: Dict[str, str] = {
            "steer_left": "left",
            "steer_right": "right",
            "accelerate": "up",
            "brake": "down",
            "handbrake": "space",
        }
        # Dual key dispatch for maximum WebGL browser compatibility (both Arrows and WASD)
        self.dual_wasd_enabled = True
        self.dual_pairs = {
            "up": "w",
            "down": "s",
            "left": "a",
            "right": "d",
            "w": "up",
            "s": "down",
            "a": "left",
            "d": "right",
        }
        self.check_focus = False
        self.allowed_focus_keywords = [
            "Burnout", "Poki", "Hilltop", "Drift", "Chrome", "Edge", "Firefox", "Opera", "Brave", "Unity", "WebGL"
        ]

    def set_mapping(self, mapping: Dict[str, str]) -> None:
        """Update action-to-key mapping."""
        with self._lock:
            self.release_all_internal()
            self._key_mapping.update(mapping)

    def get_mapping(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._key_mapping)

    def is_enabled(self) -> bool:
        """Controller is active if enabled, not emergency stopped, and not waiting for rearm."""
        return self._is_enabled and not self._emergency_stopped and not self.is_interrupted

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable driving input sending."""
        with self._lock:
            if not enabled and self._is_enabled:
                self.release_all_internal()
            if enabled:
                self._emergency_stopped = False
                self.is_interrupted = False
                self.interruption_reason = ""
            self._is_enabled = enabled
        logger.info(f"KeyboardController enabled state: {self._is_enabled}")

    def set_practice_mode(self, practice: bool) -> None:
        """Toggle practice mode (inputs tracked visually, but no real keys sent)."""
        with self._lock:
            self.release_all_internal()
            self.practice_mode = practice
        logger.info(f"KeyboardController practice mode: {self.practice_mode}")

    def set_test_mode(self, test: bool) -> None:
        """Toggle test mode for safe in-app input verification."""
        with self._lock:
            self.release_all_internal()
            self.test_mode = test
        logger.info(f"KeyboardController test mode: {self.test_mode}")

    def notify_tracking_interrupted(self, reason: str = "Tracking lost") -> None:
        """
        Called when hand tracking drops or camera fails during active driving.
        Immediately releases all keys and, if require_manual_resume is set,
        locks inputs until user manually resumes.
        """
        with self._lock:
            self.release_all_internal()
            if self._is_enabled and self.require_manual_resume:
                self.is_interrupted = True
                self.interruption_reason = reason
                self.last_event = f"STOPPED: {reason}"
                logger.warning(f"Driving interrupted: {reason}. Manual resume required.")

    def rearm_driving(self) -> None:
        """Manually re-arm driving after an interruption."""
        with self._lock:
            self.is_interrupted = False
            self.interruption_reason = ""
            self.release_all_internal()
            self.last_event = "RE-ARMED: READY"
        logger.info("Driving re-armed by user.")

    def emergency_stop(self) -> None:
        """Immediately release all keys and lock inputs until explicitly restarted."""
        with self._lock:
            self._emergency_stopped = True
            self._is_enabled = False
            self.is_interrupted = False
            self.release_all_internal()
            self.last_event = "EMERGENCY STOP (ALL KEYS RELEASED)"
        logger.warning("EMERGENCY STOP TRIGGERED: All keys released and inputs locked.")

    def reset_emergency_stop(self) -> None:
        with self._lock:
            self._emergency_stopped = False
            self.release_all_internal()

    def get_active_keys(self) -> Set[str]:
        """Return a copy of currently pressed keys."""
        with self._lock:
            return set(self._active_keys)

    def get_last_event(self) -> str:
        """Return description of last key event for diagnostics."""
        with self._lock:
            return self.last_event

    def is_controller_window(self, title: str) -> bool:
        """Returns True if the window title belongs to this desktop controller."""
        if not title:
            return False
        tl = title.lower()
        return (
            "burnout camera drive" in tl or
            "burnout mini hud" in tl or
            "camera drive" in tl
        )

    def is_game_window_focused(self) -> bool:
        """Returns True if the active foreground window is the browser game, NOT the controller."""
        if self.test_mode:
            return True  # In test mode, allow inputs to reach test box
        title = get_foreground_window_title()
        if not title:
            return False
        if self.is_controller_window(title):
            return False
        tl = title.lower()
        return any(keyword.lower() in tl for keyword in self.allowed_focus_keywords)

    def get_focus_status(self) -> Tuple[bool, str]:
        """Returns (is_focused, display_message) regarding the game window focus."""
        if self.test_mode:
            return True, "TEST MODE (Input Field Active)"
        title = get_foreground_window_title()
        if not title:
            return False, "NO ACTIVE WINDOW"
        if self.is_controller_window(title):
            return False, "CLICK EDGE GAME WINDOW TO FOCUS"
        is_focused = any(keyword.lower() in title.lower() for keyword in self.allowed_focus_keywords)
        if is_focused:
            short_title = (title[:22] + "...") if len(title) > 25 else title
            return True, f"GAME FOCUSED ({short_title})"
        else:
            short_title = (title[:20] + "...") if len(title) > 23 else title
            return False, f"CLICK GAME TO FOCUS ({short_title})"

    def check_focus_safety(self) -> bool:
        """Returns True if the target window is focused or if focus checking is disabled."""
        if not self.check_focus or self.practice_mode or self.test_mode:
            return True
        return self.is_game_window_focused()

    def find_and_focus_game_window(self) -> Tuple[bool, str]:
        """Locates the open browser window running Poki / Burnout / Edge and brings it to the foreground once."""
        if not IS_WINDOWS or not user32:
            return False, "Focus switching only supported on Windows"

        found_hwnd = None
        found_title = ""

        def enum_windows_callback(hwnd, extra):
            nonlocal found_hwnd, found_title
            if not user32.IsWindowVisible(hwnd):
                return 1
            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return 1
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            t = buff.value
            if self.is_controller_window(t):
                return 1
            tl = t.lower()
            if any(k.lower() in tl for k in self.allowed_focus_keywords):
                found_hwnd = hwnd
                found_title = t
                return 0  # Stop enumeration
            return 1

        try:
            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HWND, wintypes.LPARAM)
            user32.EnumWindows(WNDENUMPROC(enum_windows_callback), 0)
        except Exception as e:
            logger.debug(f"EnumWindows error: {e}")

        if found_hwnd:
            try:
                user32.ShowWindow(found_hwnd, 9)  # SW_RESTORE
                user32.SetForegroundWindow(found_hwnd)
                return True, f"Focused: {found_title[:25]}"
            except Exception as e:
                return False, f"Could not focus window: {e}"
        return False, "Could not find open Edge / Poki window"

    def _raw_press(self, key_name: str) -> None:
        """Dispatches key down event via Win32 keybd_event and SendInput."""
        self._dispatch_key(key_name, is_down=True)

    def _raw_release(self, key_name: str) -> None:
        """Dispatches key up event via Win32 keybd_event and SendInput."""
        self._dispatch_key(key_name, is_down=False)

    def _send_win_key(self, scan_code: int, extended: bool, is_down: bool, key_name: str = "") -> bool:
        """Executes key event via keybd_event and SendInput."""
        if not IS_WINDOWS or not user32:
            return False
        vk = VK_CODES.get(key_name, 0)
        if not vk and scan_code:
            vk = SCAN_TO_VK.get((scan_code, extended), 0)
        if not scan_code and vk and user32:
            scan_code = user32.MapVirtualKeyW(vk, 0)

        # 1. Primary Win32 method: keybd_event
        # Synthesizes WM_KEYDOWN/WM_KEYUP with VK code and scan code into foreground message queue.
        try:
            flags = 0
            if extended:
                flags |= KEYEVENTF_EXTENDEDKEY
            if not is_down:
                flags |= KEYEVENTF_KEYUP
            user32.keybd_event(vk, scan_code, flags, 0)
        except Exception as e:
            logger.error(f"keybd_event error for {key_name}: {e}")

        # 2. Secondary method: SendInput with DirectInput hardware scancode
        try:
            si_flags = KEYEVENTF_SCANCODE
            if extended:
                si_flags |= KEYEVENTF_EXTENDEDKEY
            if not is_down:
                si_flags |= KEYEVENTF_KEYUP

            extra = ctypes.c_ulong(0)
            ii = Input_I()
            ii.ki = KeyBdInput(0, scan_code, si_flags, 0, ctypes.pointer(extra))
            inp = Input(ctypes.c_ulong(INPUT_KEYBOARD), ii)
            user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))
        except Exception as e:
            logger.debug(f"SendInput secondary hook error for {key_name}: {e}")

        return True

    def _dispatch_key(self, key_name: str, is_down: bool) -> bool:
        """
        Executes hardware-level key event on the active foreground window.
        """
        key_name = key_name.lower().strip()
        action_type = "KEY_DOWN" if is_down else "KEY_UP"

        if self.practice_mode:
            self.last_event = f"PRACTICE: {action_type} {key_name.upper()}"
            self.last_event_time = time.time()
            return True

        vk = VK_CODES.get(key_name, 0)
        code, extended = SCAN_CODES.get(key_name, (0, False))

        if not vk and code:
            vk = SCAN_TO_VK.get((code, extended), 0)
        if not code and vk and IS_WINDOWS and user32:
            code = user32.MapVirtualKeyW(vk, 0)

        self.last_event = f"{action_type}: {key_name.upper()} (VK: 0x{vk:02X}, Scan: 0x{code:02X})"
        self.last_event_time = time.time()

        if IS_WINDOWS and user32:
            return self._send_win_key(code, extended, is_down, key_name=key_name)

        # Fallback for non-Windows platforms
        if pydirectinput_available:
            try:
                if is_down:
                    pydirectinput.keyDown(key_name)
                else:
                    pydirectinput.keyUp(key_name)
                return True
            except Exception:
                pass
        if pyautogui_available:
            try:
                if is_down:
                    pyautogui.keyDown(key_name)
                else:
                    pyautogui.keyUp(key_name)
                return True
            except Exception as e:
                logger.error(f"pyautogui error: {e}")
        return False

    def release_all_internal(self) -> None:
        """Internal helper to release all active keys."""
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

    def apply_actions(self, desired_actions: Dict[str, bool]) -> None:
        """
        Takes desired actions, applies safety filters, and handles press/release.
        Releases all keys immediately if game is unfocused or driving is paused.
        """
        with self._lock:
            if not self._is_enabled or self._emergency_stopped or self.is_interrupted:
                self.release_all_internal()
                return

            # Check target window focus: DO NOT send inputs if browser game is not focused!
            if self.check_focus and not self.check_focus_safety():
                self.release_all_internal()
                return

            # Determine desired keys
            desired_keys: Set[str] = set()
            for action_name, is_active in desired_actions.items():
                if is_active and action_name in self._key_mapping:
                    key = self._key_mapping[action_name].lower().strip()
                    desired_keys.add(key)

            # Mutual exclusivity: Braking strictly overrides acceleration
            accel_key = self._key_mapping.get("accelerate", "up").lower()
            brake_key = self._key_mapping.get("brake", "down").lower()
            if accel_key in desired_keys and brake_key in desired_keys:
                desired_keys.remove(accel_key)

            # Left and Right cannot both be pressed simultaneously
            left_key = self._key_mapping.get("steer_left", "left").lower()
            right_key = self._key_mapping.get("steer_right", "right").lower()
            if left_key in desired_keys and right_key in desired_keys:
                desired_keys.remove(left_key)
                desired_keys.remove(right_key)

            # Dual-dispatch paired keys (Arrows + WASD) for 100% browser game compatibility
            if getattr(self, 'dual_wasd_enabled', False):
                extra_keys = set()
                for k in list(desired_keys):
                    if k in self.dual_pairs:
                        extra_keys.add(self.dual_pairs[k])
                desired_keys.update(extra_keys)

            # Release keys no longer desired
            keys_to_release = self._active_keys - desired_keys
            for key in keys_to_release:
                self._raw_release(key)
                self._active_keys.remove(key)

            # Press newly desired keys
            keys_to_press = desired_keys - self._active_keys
            for key in keys_to_press:
                self._raw_press(key)
                self._active_keys.add(key)

    def shutdown(self) -> None:
        """Cleanup on application exit."""
        self.emergency_stop()
