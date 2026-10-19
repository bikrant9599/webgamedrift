"""
Camera capture and processing thread for Burnout Camera Drive.
Runs OpenCV capture, MediaPipe dual-hand tracking, Notebook physical steering tracking,
gesture evaluation, and HUD telemetry rendering in a background thread.
"""

import sys
import time
import math
import threading
import logging
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from PIL import Image
import cv2

from controller.hand_tracking import HandTracker, HandData
from controller.notebook_tracking import NotebookTracker, NotebookData
from controller.gesture_detection import GestureDetector, DrivingState
from controller.calibration import CalibrationManager
from controller.keyboard_control import KeyboardController
from controller.session_stats import SessionTracker

logger = logging.getLogger("burnout.camera")


def list_available_cameras(max_test: int = 2) -> List[int]:
    """Discover available camera indices using DirectShow on Windows."""
    available = []
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
    Background worker thread running camera capture, notebook tracking, hand tracking, and HUD rendering.
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

        # Notebook Tracker
        hsv_l = self.calibration.config.get("notebook_hsv_lower", [7, 75, 120])
        hsv_u = self.calibration.config.get("notebook_hsv_upper", [28, 255, 255])
        self.notebook_tracker = NotebookTracker(hsv_lower=tuple(hsv_l), hsv_upper=tuple(hsv_u))
        self.notebook_tracker.neutral_angle = float(self.calibration.config.get("notebook_neutral_angle", 0.0))
        self.notebook_tracker.neutral_pitch = float(self.calibration.config.get("notebook_neutral_pitch", 0.0))
        self.notebook_tracker.neutral_height = float(self.calibration.config.get("notebook_neutral_height", 0.0))
        self.notebook_tracker.neutral_width = float(self.calibration.config.get("notebook_neutral_width", 0.0))
        self.notebook_tracker.brake_pitch_thresh = float(self.calibration.config.get("notebook_brake_pitch_thresh", 9.0))
        self.notebook_tracker.brake_height_thresh = float(self.calibration.config.get("notebook_brake_height_thresh", 0.10))

        self._running = False
        self._lock = threading.Lock()

        # Telemetry & latest state
        self.is_connected = False
        self.latest_frame_pil: Optional[Image.Image] = None
        self.current_state = DrivingState()
        self.current_hand = HandData()
        self.current_notebook = NotebookData()
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
        cap = cv2.VideoCapture(self.camera_index, backend)

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
                    self.is_connected = False
                    self.keyboard_ctrl.release_all()
                    self.keyboard_ctrl.notify_tracking_interrupted("Camera disconnected")
                    self.session_tracker.record_tracking_status(False, self.keyboard_ctrl.is_enabled())
                    continue

            # Process hand tracking with mirror view
            processed_bgr, hand_data = self.hand_tracker.process_frame(frame_bgr, mirror=True)

            # Process notebook tracking on the mirrored frame
            hand_roi = None
            if hand_data.left_hand_center and hand_data.right_hand_center:
                hand_roi = (hand_data.left_hand_center, hand_data.right_hand_center)

            self.notebook_tracker.neutral_angle = float(self.calibration.config.get("notebook_neutral_angle", 0.0))
            self.notebook_tracker.neutral_pitch = float(self.calibration.config.get("notebook_neutral_pitch", 0.0))
            self.notebook_tracker.neutral_height = float(self.calibration.config.get("notebook_neutral_height", 0.0))
            self.notebook_tracker.neutral_width = float(self.calibration.config.get("notebook_neutral_width", 0.0))
            self.notebook_tracker.brake_pitch_thresh = float(self.calibration.config.get("notebook_brake_pitch_thresh", 9.0))
            self.notebook_tracker.brake_height_thresh = float(self.calibration.config.get("notebook_brake_height_thresh", 0.10))
            if "notebook_hsv_lower" in self.calibration.config and "notebook_hsv_upper" in self.calibration.config:
                self.notebook_tracker.hsv_lower = np.array(self.calibration.config["notebook_hsv_lower"], dtype=np.uint8)
                self.notebook_tracker.hsv_upper = np.array(self.calibration.config["notebook_hsv_upper"], dtype=np.uint8)
            notebook_data = self.notebook_tracker.process_frame(processed_bgr, hand_roi=hand_roi)

            # Record tracking status
            tracking_healthy = hand_data.is_detected or notebook_data.is_detected
            self.session_tracker.record_tracking_status(tracking_healthy, self.keyboard_ctrl.is_enabled())

            # ---------------------------------------------------------
            # CALIBRATION MODE HANDLING
            # ---------------------------------------------------------
            if self.calibration.is_calibrating:
                self.keyboard_ctrl.release_all()
                if self.calibration.calibration_type == "notebook_wizard":
                    done, msg, prog = self.calibration.update_notebook_calibration(notebook_data, hand_data)
                    self.calibration_feedback = (msg, prog)
                    driving_state = DrivingState(gesture_name="CALIBRATING NOTEBOOK", details=msg)
                else:
                    done, msg, prog = self.calibration.update_calibration_sample(
                        hand_data.hand_center, hand_data.hand_size
                    )
                    self.calibration_feedback = (msg, prog)
                    driving_state = DrivingState(gesture_name="CALIBRATING HAND", details=msg)
            else:
                # ---------------------------------------------------------
                # LIVE DRIVING EVALUATION
                # ---------------------------------------------------------
                use_notebook = self.calibration.notebook_steering_enabled
                if use_notebook and not notebook_data.is_detected:
                    # Notebook tracking lost -> release steering
                    pass

                driving_state = self.gesture_detector.evaluate(hand_data, notebook=notebook_data)

                # Record actions
                self.session_tracker.record_actions(
                    driving_state.accelerate, driving_state.brake, driving_state.handbrake
                )

                if self.keyboard_ctrl.is_enabled():
                    self.keyboard_ctrl.apply_actions(driving_state.to_action_dict())
                else:
                    self.keyboard_ctrl.release_all()

            # Render live visual feedback HUD
            hud_bgr = self._render_live_hud(processed_bgr, hand_data, notebook_data, driving_state)

            # Convert to PIL Image for CustomTkinter preview
            rgb_frame = cv2.cvtColor(hud_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            # FPS calculation
            self._frame_count += 1
            now = time.time()
            if now - self._fps_timer >= 1.0:
                self.fps_actual = self._frame_count / (now - self._fps_timer)
                self._frame_count = 0
                self._fps_timer = now

            with self._lock:
                self.latest_frame_pil = pil_img
                self.current_state = driving_state
                self.current_hand = hand_data
                self.current_notebook = notebook_data

            elapsed = time.time() - loop_start
            sleep_time = max(0.001, frame_duration - elapsed)
            time.sleep(sleep_time)

        if cap:
            cap.release()
        self.keyboard_ctrl.release_all()
        logger.info("Camera thread exited.")

    def get_latest_snapshot(self) -> Tuple[Optional[Image.Image], DrivingState, HandData, NotebookData, float, bool]:
        """Thread-safe snapshot for UI polling."""
        with self._lock:
            return (
                self.latest_frame_pil,
                self.current_state,
                self.current_hand,
                self.current_notebook,
                self.fps_actual,
                self.is_connected
            )

    def _generate_disconnected_frame(self) -> np.ndarray:
        img = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        img[:] = (20, 22, 26)
        cv2.putText(img, "CAMERA DISCONNECTED OR NOT FOUND", (60, self.target_height // 2 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (80, 80, 220), 1, cv2.LINE_AA)
        cv2.putText(img, "Select camera index in settings and click Reconnect", (80, self.target_height // 2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 150), 1, cv2.LINE_AA)
        return img

    def _render_live_hud(self, frame: np.ndarray,
                         hand: HandData,
                         nb: NotebookData,
                         state: DrivingState) -> np.ndarray:
        """
        Renders rich visual feedback:
        - Notebook tracking outline & 4 corners
        - Center reticle & orientation vector
        - Right-hand index finger status badge
        - Live steering gauge & active key indicators
        """
        h, w, _ = frame.shape

        # -------------------------------------------------------------
        # 1. NOTEBOOK VISUAL TRACKING OVERLAY
        # -------------------------------------------------------------
        if nb.is_detected and len(nb.corners) == 4:
            pts = np.array(nb.corners, dtype=np.int32).reshape((-1, 1, 2))

            # Dynamic outline color: RED if braking, CYAN if steering, GREEN if neutral
            if nb.is_braking or state.brake:
                outline_color = (0, 0, 255)       # Red (Braking)
                fill_color = (0, 0, 220)
                status_tag = f"BRAKE ACTIVE (Pitch: {nb.pitch_angle:+.1f} deg)"
                tag_color = (0, 50, 255)
            elif state.steer_left or state.steer_right:
                outline_color = (0, 220, 255)     # Cyan (Steering)
                fill_color = (0, 140, 255)
                status_tag = f"STEER {'LEFT' if state.steer_left else 'RIGHT'} ({nb.rotation_angle:+.1f} deg)"
                tag_color = (0, 220, 255)
            else:
                outline_color = (0, 255, 120)     # Emerald Green (Neutral)
                fill_color = (0, 180, 80)
                status_tag = f"LEVEL / NEUTRAL ({nb.rotation_angle:+.1f} deg)"
                tag_color = (0, 255, 120)

            # Draw notebook polygon outline
            cv2.polylines(frame, [pts], isClosed=True, color=outline_color, thickness=3, lineType=cv2.LINE_AA)

            # Fill translucent highlight inside notebook
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], fill_color)
            cv2.addWeighted(overlay, 0.14, frame, 0.86, 0, frame)

            # Draw the 4 corner points with numbered tags
            corner_labels = ["1:TL", "2:TR", "3:BR", "4:BL"]
            for i, (cx, cy) in enumerate(nb.corners):
                cv2.circle(frame, (cx, cy), 6, outline_color, -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, corner_labels[i], (cx + 8, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 240, 255), 1, cv2.LINE_AA)

            # Center Point Reticle
            ncx, ncy = nb.center
            cv2.circle(frame, (ncx, ncy), 7, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.drawMarker(frame, (ncx, ncy), (255, 255, 255), cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)

            # Orientation Vector Line (top edge)
            tl = nb.corners[0]
            tr = nb.corners[1]
            cv2.line(frame, tl, tr, outline_color, 2, cv2.LINE_AA)

            # Display angle & status tag above notebook
            ang_txt = f"ROT: {nb.rotation_angle:+.1f} deg | PITCH: {nb.pitch_angle:+.1f} deg"
            cv2.putText(frame, ang_txt, (max(10, ncx - 100), max(25, ncy - int(nb.height // 2) - 18)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, status_tag, (max(10, ncx - 100), max(40, ncy - int(nb.height // 2) - 3)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, tag_color, 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 2. HAND TRACKING OVERLAYS & SKELETON
        # -------------------------------------------------------------
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
            (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
            (0, 17)                                 # Palm
        ]

        # Draw Right Hand
        if hand.right_hand_detected and hand.right_hand_landmarks:
            r_pts = hand.right_hand_landmarks
            r_color = (0, 255, 120) if hand.right_index_raised else (0, 180, 255)
            for p1_idx, p2_idx in connections:
                if p1_idx < len(r_pts) and p2_idx < len(r_pts):
                    cv2.line(frame, r_pts[p1_idx], r_pts[p2_idx], r_color, 2, cv2.LINE_AA)
            for pt in r_pts:
                cv2.circle(frame, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)

            # Highlight raised index tip with pulsating dot
            if hand.right_index_raised and len(r_pts) > 8:
                tip = r_pts[8]
                cv2.circle(frame, tip, 9, (0, 255, 80), -1, cv2.LINE_AA)
                cv2.circle(frame, tip, 13, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "INDEX UP (GAS)", (tip[0] + 12, tip[1] - 8),
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 120), 1, cv2.LINE_AA)

        # Draw Left Hand
        if hand.left_hand_detected and hand.left_hand_landmarks:
            l_pts = hand.left_hand_landmarks
            for p1_idx, p2_idx in connections:
                if p1_idx < len(l_pts) and p2_idx < len(l_pts):
                    cv2.line(frame, l_pts[p1_idx], l_pts[p2_idx], (0, 180, 255), 2, cv2.LINE_AA)
            for pt in l_pts:
                cv2.circle(frame, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 3. LIVE HUD TELEMETRY CARDS
        # -------------------------------------------------------------
        # Top Left Telemetry Box
        cv2.rectangle(frame, (10, 10), (320, 125), (15, 18, 22), -1)
        cv2.rectangle(frame, (10, 10), (320, 125), (45, 52, 65), 1)

        # Line 1: State title
        color_txt = (70, 255, 100) if state.accelerate else ((70, 70, 255) if state.brake else (255, 255, 255))
        cv2.putText(frame, state.gesture_name, (18, 30), cv2.FONT_HERSHEY_DUPLEX, 0.46, color_txt, 1, cv2.LINE_AA)

        # Line 2: Notebook & Calibration status
        cal_ok = "CALIB OK" if self.calibration.config.get("notebook_calibrated") else "UNCALIBRATED"
        nb_status = f"Notebook: {'LOCKED' if nb.is_detected else 'SEARCHING'} | {cal_ok}"
        cv2.putText(frame, nb_status, (18, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255) if nb.is_detected else (180, 180, 190), 1, cv2.LINE_AA)

        # Line 3: Steering direction & angle
        steer_dir = "LEFT" if state.steer_left else ("RIGHT" if state.steer_right else "CENTER")
        cv2.putText(frame, f"Steering: {steer_dir} ({nb.rotation_angle:+.1f} deg, neutral: {self.notebook_tracker.neutral_angle:.1f} deg)",
                    (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 205, 215), 1, cv2.LINE_AA)

        # Line 4: Brake & Acceleration status
        brk_lbl = "BRAKE: ACTIVE [DOWN]" if state.brake else f"Brake: Released (Pitch {nb.pitch_angle:+.1f} deg)"
        brk_col = (50, 70, 255) if state.brake else (140, 145, 155)
        cv2.putText(frame, brk_lbl, (18, 87), cv2.FONT_HERSHEY_SIMPLEX, 0.36, brk_col, 1, cv2.LINE_AA)

        # Line 5: Right Index finger & Active Keys
        active_keys = sorted(list(self.keyboard_ctrl.get_active_keys()))
        keys_str = "[" + ", ".join(k.upper() for k in active_keys) + "]" if active_keys else "[NONE]"
        gas_lbl = f"Gas: {'ACTIVE [UP]' if state.accelerate else 'COAST'} | Keys: {keys_str}"
        gas_col = (0, 255, 120) if state.accelerate else (140, 145, 155)
        cv2.putText(frame, gas_lbl, (18, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.36, gas_col, 1, cv2.LINE_AA)

        # Calibration warning banner if present
        if self.calibration.calibration_warning:
            cv2.rectangle(frame, (10, h - 55), (w - 10, h - 33), (20, 20, 60), -1)
            cv2.rectangle(frame, (10, h - 55), (w - 10, h - 33), (0, 160, 255), 1)
            cv2.putText(frame, f"NOTE: {self.calibration.calibration_warning}", (18, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 4. STEERING BAR GAUGE (BOTTOM)
        # -------------------------------------------------------------
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

        # Mode tag at top right
        if self.keyboard_ctrl.practice_mode:
            cv2.putText(frame, "[PRACTICE MODE - NO KEYS]", (w - 245, 26),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
        elif self.keyboard_ctrl.is_interrupted:
            cv2.putText(frame, "[INTERRUPTED - RE-ARM REQ]", (w - 250, 26),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, (50, 50, 255), 1, cv2.LINE_AA)
        elif self.keyboard_ctrl.is_enabled():
            cv2.putText(frame, "[LIVE NOTEBOOK DRIVE]", (w - 200, 26),
                        cv2.FONT_HERSHEY_DUPLEX, 0.45, (80, 255, 120), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "[STANDBY / PAUSED]", (w - 175, 26),
                        cv2.FONT_HERSHEY_DUPLEX, 0.42, (120, 120, 140), 1, cv2.LINE_AA)

        return frame
