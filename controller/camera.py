"""
Camera capture and processing thread.
Runs real-time OpenCV capture, MediaPipe tracking, gesture evaluation, and HUD rendering
in a high-performance background thread without blocking the CustomTkinter UI.
"""

import sys
import time
import queue
import threading
import logging
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from PIL import Image

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

from controller.hand_tracking import HandTracker, HandData
from controller.gesture_detection import GestureDetector, DrivingState
from controller.calibration import CalibrationManager
from controller.keyboard_control import KeyboardController

logger = logging.getLogger("burnout.camera")


def list_available_cameras(max_test: int = 4) -> List[int]:
    """Discover available camera indices using DirectShow on Windows."""
    available = []
    if not OPENCV_AVAILABLE:
        return [0]

    for index in range(max_test):
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(index)
            cap.release()
    return available if available else [0]


class CameraThread(threading.Thread):
    """
    Background worker thread running camera capture, inference, and HUD rendering.
    """

    def __init__(self,
                 camera_index: int,
                 hand_tracker: HandTracker,
                 gesture_detector: GestureDetector,
                 calibration: CalibrationManager,
                 keyboard_ctrl: KeyboardController,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30):
        super().__init__(name="BurnoutCameraThread", daemon=True)
        self.camera_index = camera_index
        self.hand_tracker = hand_tracker
        self.gesture_detector = gesture_detector
        self.calibration = calibration
        self.keyboard_ctrl = keyboard_ctrl
        self.target_width = width
        self.target_height = height
        self.target_fps = fps

        self._running = False
        self._lock = threading.Lock()

        # Telemetry & latest state
        self.is_connected = False
        self.latest_frame_pil: Optional[Image.Image] = None
        self.current_state = DrivingState()
        self.current_hand = HandData()
        self.fps_actual = 0.0
        self.calibration_feedback = ("Idle", 0.0)

        # Performance counters
        self._frame_count = 0
        self._fps_timer = time.time()

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        self._running = True
        logger.info(f"Opening camera index {self.camera_index}...")

        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.camera_index, backend) if OPENCV_AVAILABLE else None

        if cap and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.is_connected = True
            logger.info("Camera successfully connected.")
        else:
            self.is_connected = False
            logger.warning(f"Failed to open camera index {self.camera_index}.")

        frame_duration = 1.0 / self.target_fps

        while self._running:
            loop_start = time.time()

            if not self.is_connected or cap is None or not cap.isOpened():
                # Provide dummy placeholder frame if camera disconnected
                frame_bgr = self._generate_disconnected_frame()
                self.keyboard_ctrl.release_all()
                time.sleep(0.1)
            else:
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    logger.warning("Camera read error or device disconnected.")
                    self.is_connected = False
                    self.keyboard_ctrl.release_all()
                    continue

            # Process hand tracking
            processed_bgr, hand_data = self.hand_tracker.process_frame(frame_bgr, mirror=True)

            # Safety check: if hand tracking is lost, immediately release all keys
            if not hand_data.is_detected:
                self.keyboard_ctrl.release_all()
                driving_state = DrivingState(gesture_name="NO HAND DETECTED", details="Hand not in view")
            else:
                # Handle calibration if active
                if self.calibration.is_calibrating:
                    done, msg, prog = self.calibration.update_calibration_sample(
                        hand_data.hand_center, hand_data.hand_size
                    )
                    self.calibration_feedback = (msg, prog)
                    driving_state = DrivingState(gesture_name="CALIBRATING", details=msg)
                else:
                    driving_state = self.gesture_detector.evaluate(hand_data)
                    # Apply keys if controller is enabled
                    if self.keyboard_ctrl.is_enabled():
                        self.keyboard_ctrl.apply_actions(driving_state.to_action_dict())
                    else:
                        self.keyboard_ctrl.release_all()

            # Render racing HUD telemetry overlay onto frame
            hud_bgr = self._render_racing_hud(processed_bgr, hand_data, driving_state)

            # Convert to PIL Image for CustomTkinter preview
            rgb_frame = cv2.cvtColor(hud_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            # Calculate FPS
            self._frame_count += 1
            now = time.time()
            if now - self._fps_timer >= 1.0:
                self.fps_actual = self._frame_count / (now - self._fps_timer)
                self._frame_count = 0
                self._fps_timer = now

            # Thread-safe telemetry update
            with self._lock:
                self.latest_frame_pil = pil_img
                self.current_state = driving_state
                self.current_hand = hand_data

            # Rate limit to match camera frame rate
            elapsed = time.time() - loop_start
            sleep_time = max(0.001, frame_duration - elapsed)
            time.sleep(sleep_time)

        # Cleanup on thread exit
        if cap:
            cap.release()
        self.keyboard_ctrl.release_all()
        logger.info("Camera thread exited.")

    def get_latest_snapshot(self) -> Tuple[Optional[Image.Image], DrivingState, HandData, float, bool]:
        """Thread-safe snapshot for UI polling."""
        with self._lock:
            return (
                self.latest_frame_pil,
                self.current_state,
                self.current_hand,
                self.fps_actual,
                self.is_connected
            )

    def _generate_disconnected_frame(self) -> np.ndarray:
        """Create a stylish dark placeholder when camera is offline."""
        img = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        # Charcoal background with grid lines
        img[:] = (20, 22, 26)
        cv2.putText(img, "CAMERA DISCONNECTED OR NOT FOUND", (60, self.target_height // 2 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (80, 80, 220), 1, cv2.LINE_AA)
        cv2.putText(img, "Select camera index in settings and click Reconnect", (80, self.target_height // 2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 150), 1, cv2.LINE_AA)
        return img

    def _render_racing_hud(self, frame: np.ndarray, hand: HandData, state: DrivingState) -> np.ndarray:
        """Draw an electric-blue racing HUD overlay with telemetry gauges."""
        h, w, c = frame.shape

        # 1. Neutral Steering Guide & Dead Zone Box
        neutral_x = int(self.calibration.neutral_x * w)
        deadzone_px = int(self.calibration.deadzone_x * w)

        # Translucent deadzone vertical corridor
        sub_img = frame.copy()
        cv2.rectangle(sub_img, (neutral_x - deadzone_px, 0), (neutral_x + deadzone_px, h), (0, 160, 255), -1)
        cv2.addWeighted(sub_img, 0.08, frame, 0.92, 0, frame)

        # Centerline guide (subtle dashed/dotted cyan line)
        cv2.line(frame, (neutral_x, 0), (neutral_x, h), (255, 200, 0), 1, cv2.LINE_AA)

        # Deadzone boundary lines
        cv2.line(frame, (neutral_x - deadzone_px, 0), (neutral_x - deadzone_px, h), (0, 140, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (neutral_x + deadzone_px, 0), (neutral_x + deadzone_px, h), (0, 140, 255), 1, cv2.LINE_AA)

        # 2. Draw Hand Landmarks with Neon Electric Blue Glow
        if hand.is_detected and hand.landmarks_pixel:
            # Connections
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),        # Index
                (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
                (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
                (0, 17)                                 # Palm base
            ]

            # Line color based on state
            if state.brake:
                line_color = (40, 40, 255)   # Red for brake
                joint_color = (80, 80, 255)
            elif state.handbrake:
                line_color = (255, 50, 200)  # Magenta/Purple for handbrake
                joint_color = (255, 120, 220)
            elif state.accelerate:
                line_color = (50, 240, 100)  # Neon green for throttle
                joint_color = (120, 255, 150)
            else:
                line_color = (255, 210, 0)   # Electric cyan-blue default
                joint_color = (255, 255, 255)

            for p1_idx, p2_idx in connections:
                if p1_idx < len(hand.landmarks_pixel) and p2_idx < len(hand.landmarks_pixel):
                    p1 = hand.landmarks_pixel[p1_idx]
                    p2 = hand.landmarks_pixel[p2_idx]
                    cv2.line(frame, p1, p2, line_color, 2, cv2.LINE_AA)

            for pt in hand.landmarks_pixel:
                cv2.circle(frame, pt, 4, joint_color, -1, cv2.LINE_AA)

            # Draw Hand Center Reticle
            cx = int(hand.hand_center[0] * w)
            cy = int(hand.hand_center[1] * h)
            cv2.circle(frame, (cx, cy), 8, line_color, 2, cv2.LINE_AA)
            cv2.line(frame, (cx - 12, cy), (cx + 12, cy), line_color, 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy - 12), (cx, cy + 12), line_color, 1, cv2.LINE_AA)

        # 3. Steering Bar Gauge at Bottom
        bar_w = int(w * 0.65)
        bar_h = 14
        bar_x = (w - bar_w) // 2
        bar_y = h - 28

        # Background track
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (35, 38, 44), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 75, 85), 1)

        # Center tick
        mid_x = bar_x + (bar_w // 2)
        cv2.line(frame, (mid_x, bar_y - 3), (mid_x, bar_y + bar_h + 3), (200, 200, 200), 2)

        # Active steering fill
        steer_fill = int((state.steering_val) * (bar_w // 2))
        if steer_fill < 0:
            # Steer left
            cv2.rectangle(frame, (mid_x + steer_fill, bar_y), (mid_x, bar_y + bar_h), (255, 180, 0), -1)
        elif steer_fill > 0:
            # Steer right
            cv2.rectangle(frame, (mid_x, bar_y), (mid_x + steer_fill, bar_y + bar_h), (255, 180, 0), -1)

        # 4. Top Telemetry Pill Overlay
        # Pill background
        cv2.rectangle(frame, (10, 10), (220, 65), (15, 18, 22), -1)
        cv2.rectangle(frame, (10, 10), (220, 65), (45, 52, 65), 1)

        # State text
        gesture_txt = state.gesture_name
        color_txt = (255, 255, 255)
        if state.brake:
            color_txt = (70, 70, 255)
        elif state.accelerate:
            color_txt = (70, 255, 100)
        elif state.handbrake:
            color_txt = (255, 100, 220)

        cv2.putText(frame, gesture_txt, (18, 32), cv2.FONT_HERSHEY_DUPLEX, 0.48, color_txt, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {self.fps_actual:.1f} | Sens: {self.calibration.steering_sensitivity:.1f}x",
                    (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 170, 185), 1, cv2.LINE_AA)

        # Active driving input indicator banner
        if self.keyboard_ctrl.is_enabled():
            cv2.putText(frame, "[DRIVING ACTIVE]", (w - 150, 26),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, (80, 255, 120), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "[STANDBY / PAUSED]", (w - 165, 26),
                        cv2.FONT_HERSHEY_DUPLEX, 0.42, (120, 120, 140), 1, cv2.LINE_AA)

        return frame
