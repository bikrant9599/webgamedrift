"""
Camera capture and processing thread.
Runs real-time OpenCV capture, MediaPipe tracking, gesture evaluation, and HUD rendering
in a high-performance background thread without blocking the CustomTkinter UI.
Includes an animated virtual steering wheel HUD, practice mode banner, and session telemetry.
"""

import sys
import time
import math
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
from controller.session_stats import SessionTracker

logger = logging.getLogger("burnout.camera")


def list_available_cameras(max_test: int = 2) -> List[int]:
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
                 session_tracker: SessionTracker,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30):
        super().__init__(name="BurnoutCameraThread", daemon=True)
        self.camera_index = camera_index
        self.hand_tracker = hand_tracker
        self.gesture_detector = gesture_detector
        self.calibration = calibration
        self.keyboard_ctrl = keyboard_ctrl
        self.session_tracker = session_tracker
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
                frame_bgr = self._generate_disconnected_frame()
                self.keyboard_ctrl.release_all()
                self.session_tracker.record_tracking_status(False, self.keyboard_ctrl.is_enabled())
                time.sleep(0.1)
            else:
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    logger.warning("Camera read error or device disconnected.")
                    self.is_connected = False
                    self.keyboard_ctrl.release_all()
                    self.keyboard_ctrl.notify_tracking_interrupted("Camera disconnected")
                    self.session_tracker.record_tracking_status(False, self.keyboard_ctrl.is_enabled())
                    continue

            # Process hand tracking (single or dual hand)
            processed_bgr, hand_data = self.hand_tracker.process_frame(frame_bgr, mirror=True)

            # Safety check: if hand tracking is lost, immediately release all keys and notify interruption
            if not hand_data.is_detected:
                if self.keyboard_ctrl.is_enabled():
                    self.keyboard_ctrl.notify_tracking_interrupted("Hand tracking lost")
                else:
                    self.keyboard_ctrl.release_all()

                self.session_tracker.record_tracking_status(False, self.keyboard_ctrl.is_enabled())
                driving_state = DrivingState(gesture_name="NO HAND DETECTED", details="Hand not in view")
            else:
                self.session_tracker.record_tracking_status(True, self.keyboard_ctrl.is_enabled())

                # Handle calibration if active (prevents accidental steering during calibration)
                if self.calibration.is_calibrating:
                    self.keyboard_ctrl.release_all()
                    done, msg, prog = self.calibration.update_calibration_sample(
                        hand_data.hand_center, hand_data.hand_size
                    )
                    self.calibration_feedback = (msg, prog)
                    driving_state = DrivingState(gesture_name="CALIBRATING", details=msg)
                else:
                    driving_state = self.gesture_detector.evaluate(hand_data)

                    # Record actions for session stats
                    self.session_tracker.record_actions(
                        driving_state.accelerate, driving_state.brake, driving_state.handbrake
                    )

                    # Apply inputs if enabled (or simulate in practice mode)
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

            elapsed = time.time() - loop_start
            sleep_time = max(0.001, frame_duration - elapsed)
            time.sleep(sleep_time)

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
        img[:] = (20, 22, 26)
        cv2.putText(img, "CAMERA DISCONNECTED OR NOT FOUND", (60, self.target_height // 2 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (80, 80, 220), 1, cv2.LINE_AA)
        cv2.putText(img, "Select camera index in settings and click Reconnect", (80, self.target_height // 2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 150), 1, cv2.LINE_AA)
        return img

    def _render_racing_hud(self, frame: np.ndarray, hand: HandData, state: DrivingState) -> np.ndarray:
        """Draw an electric-blue racing HUD overlay with telemetry and animated virtual steering wheel."""
        h, w, c = frame.shape

        # 1. Neutral Steering Guide & Dead Zone Corridor
        neutral_x = int(self.calibration.neutral_x * w)
        deadzone_px = int(self.calibration.deadzone_x * w)

        sub_img = frame.copy()
        cv2.rectangle(sub_img, (neutral_x - deadzone_px, 0), (neutral_x + deadzone_px, h), (0, 160, 255), -1)
        cv2.addWeighted(sub_img, 0.08, frame, 0.92, 0, frame)

        # Centerline guide & deadzone bounds
        cv2.line(frame, (neutral_x, 0), (neutral_x, h), (255, 200, 0), 1, cv2.LINE_AA)
        cv2.line(frame, (neutral_x - deadzone_px, 0), (neutral_x - deadzone_px, h), (0, 140, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (neutral_x + deadzone_px, 0), (neutral_x + deadzone_px, h), (0, 140, 255), 1, cv2.LINE_AA)

        # 2. Draw Hand Landmarks with Neon Electric Blue Glow
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
            (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
            (0, 17)                                 # Palm base
        ]

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

        # Primary Hand Skeleton
        if hand.is_detected and hand.landmarks_pixel:
            for p1_idx, p2_idx in connections:
                if p1_idx < len(hand.landmarks_pixel) and p2_idx < len(hand.landmarks_pixel):
                    cv2.line(frame, hand.landmarks_pixel[p1_idx], hand.landmarks_pixel[p2_idx], line_color, 2, cv2.LINE_AA)
            for pt in hand.landmarks_pixel:
                cv2.circle(frame, pt, 4, joint_color, -1, cv2.LINE_AA)

            cx = int(hand.hand_center[0] * w)
            cy = int(hand.hand_center[1] * h)
            cv2.circle(frame, (cx, cy), 7, line_color, 2, cv2.LINE_AA)

        # Secondary Hand Skeleton (for Two-Hand Wheel Mode)
        if hand.is_two_handed and hand.second_landmarks_pixel:
            for p1_idx, p2_idx in connections:
                if p1_idx < len(hand.second_landmarks_pixel) and p2_idx < len(hand.second_landmarks_pixel):
                    cv2.line(frame, hand.second_landmarks_pixel[p1_idx], hand.second_landmarks_pixel[p2_idx], line_color, 2, cv2.LINE_AA)
            for pt in hand.second_landmarks_pixel:
                cv2.circle(frame, pt, 4, joint_color, -1, cv2.LINE_AA)

            # Draw virtual connecting axle between the two hands
            if hand.left_hand_center and hand.right_hand_center:
                lx, ly = int(hand.left_hand_center[0] * w), int(hand.left_hand_center[1] * h)
                rx, ry = int(hand.right_hand_center[0] * w), int(hand.right_hand_center[1] * h)
                cv2.line(frame, (lx, ly), (rx, ry), (0, 220, 255), 2, cv2.LINE_AA)

        # 3. ANIMATED VIRTUAL STEERING WHEEL DIAL
        # Render a rotating virtual racing wheel in bottom center or right corner
        self._draw_virtual_steering_wheel(frame, hand, state)

        # 4. Steering Bar Gauge at Bottom
        bar_w = int(w * 0.65)
        bar_h = 14
        bar_x = (w - bar_w) // 2
        bar_y = h - 26

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (35, 38, 44), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 75, 85), 1)

        mid_x = bar_x + (bar_w // 2)
        cv2.line(frame, (mid_x, bar_y - 3), (mid_x, bar_y + bar_h + 3), (200, 200, 200), 2)

        steer_fill = int(state.steering_val * (bar_w // 2))
        if steer_fill < 0:
            cv2.rectangle(frame, (mid_x + steer_fill, bar_y), (mid_x, bar_y + bar_h), (255, 180, 0), -1)
        elif steer_fill > 0:
            cv2.rectangle(frame, (mid_x, bar_y), (mid_x + steer_fill, bar_y + bar_h), (255, 180, 0), -1)

        # 5. Top Left Telemetry Pill Overlay
        cv2.rectangle(frame, (10, 10), (235, 68), (15, 18, 22), -1)
        cv2.rectangle(frame, (10, 10), (235, 68), (45, 52, 65), 1)

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

        # 6. Mode & Driving Status Banner at Top Right
        if self.keyboard_ctrl.practice_mode:
            cv2.putText(frame, "[PRACTICE MODE - NO KEYS]", (w - 245, 26),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
        elif self.keyboard_ctrl.is_interrupted:
            cv2.putText(frame, "[INTERRUPTED - RE-ARM REQ]", (w - 250, 26),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, (50, 50, 255), 1, cv2.LINE_AA)
        elif self.keyboard_ctrl.is_enabled():
            cv2.putText(frame, "[LIVE DRIVING ACTIVE]", (w - 180, 26),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, (80, 255, 120), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "[STANDBY / PAUSED]", (w - 175, 26),
                        cv2.FONT_HERSHEY_DUPLEX, 0.42, (120, 120, 140), 1, cv2.LINE_AA)

        return frame

    def _draw_virtual_steering_wheel(self, frame: np.ndarray, hand: HandData, state: DrivingState):
        """Draw an animated racing virtual steering wheel that rotates with hand angle."""
        h, w, _ = frame.shape
        # Center position of steering wheel in bottom right corner
        wc_x = w - 80
        wc_y = h - 80
        wheel_radius = 42

        # Rotation angle in radians
        rot_deg = hand.wheel_angle if (hand.is_two_handed or self.calibration.two_hand_mode_enabled) else (state.steering_val * 45.0)
        rot_rad = math.radians(rot_deg)

        # Wheel rim
        cv2.circle(frame, (wc_x, wc_y), wheel_radius, (30, 34, 42), -1)
        cv2.circle(frame, (wc_x, wc_y), wheel_radius, (0, 210, 255), 3, cv2.LINE_AA)

        # Inner hub
        cv2.circle(frame, (wc_x, wc_y), 12, (0, 240, 255), -1, cv2.LINE_AA)

        # 3 Spokes rotated by rot_rad
        # Spoke 1: Left (-90 deg from top = left)
        spoke_angles = [rot_rad - math.pi / 2, rot_rad + math.pi / 2, rot_rad]
        for sa in spoke_angles:
            sx = int(wc_x + math.cos(sa) * (wheel_radius - 4))
            sy = int(wc_y + math.sin(sa) * (wheel_radius - 4))
            cv2.line(frame, (wc_x, wc_y), (sx, sy), (0, 210, 255), 2, cv2.LINE_AA)

        # Top Dead Center Alignment Marker (Red racing stripe on rim)
        top_sa = rot_rad - (math.pi / 2)
        mx = int(wc_x + math.cos(top_sa) * wheel_radius)
        my = int(wc_y + math.sin(top_sa) * wheel_radius)
        cv2.circle(frame, (mx, my), 4, (40, 40, 255), -1, cv2.LINE_AA)

        # Angle text
        angle_str = f"{rot_deg:+.0f}°"
        cv2.putText(frame, angle_str, (wc_x - 18, wc_y + wheel_radius + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 230, 255), 1, cv2.LINE_AA)

        # Mode label
        if hand.is_two_handed:
            mode_lbl = "2-Hand Wheel"
        elif hand.is_fallback:
            mode_lbl = "1-Hand Fallback"
        else:
            mode_lbl = "Virtual Wheel"
        cv2.putText(frame, mode_lbl, (wc_x - 38, wc_y - wheel_radius - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 160, 175), 1, cv2.LINE_AA)
