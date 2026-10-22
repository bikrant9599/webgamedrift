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
        self._latest_raw_bgr: Optional[np.ndarray] = None
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

            with self._lock:
                self._latest_raw_bgr = frame_bgr.copy()

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

    def sample_notebook_color(self) -> Tuple[bool, str]:
        """Samples center region of latest frame to adaptively calibrate HSV bounds for any notebook."""
        with self._lock:
            frame = self._latest_raw_bgr.copy() if self._latest_raw_bgr is not None else None
        if frame is None:
            return False, "No camera frame available yet"

        lower, upper = self.notebook_tracker.calibrate_color_from_roi(frame)
        self.calibration.config["notebook_hsv_lower"] = list(lower)
        self.calibration.config["notebook_hsv_upper"] = list(upper)
        self.calibration.config["notebook_color_name"] = "Custom Sampled"
        self.calibration.save()
        return True, f"Color Sampled! (Hue {lower[0]}-{upper[0]})"

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

        # Draw hand skeletons
        hands_to_draw = []
        if hand.right_hand_landmarks:
            hands_to_draw.append((hand.right_hand_landmarks, True))
        if hand.left_hand_landmarks and hand.left_hand_landmarks != hand.right_hand_landmarks:
            hands_to_draw.append((hand.left_hand_landmarks, False))
        elif not hands_to_draw and hand.landmarks_pixel:
            hands_to_draw.append((hand.landmarks_pixel, True))

        for pts, is_right in hands_to_draw:
            line_color = (0, 220, 255) if is_right else (255, 180, 0)
            if state.accelerate:
                line_color = (0, 255, 120)
            elif state.brake:
                line_color = (60, 60, 255)
            elif state.handbrake:
                line_color = (255, 120, 200)

            for p1_idx, p2_idx in connections:
                if p1_idx < len(pts) and p2_idx < len(pts):
                    cv2.line(frame, pts[p1_idx], pts[p2_idx], line_color, 2, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(frame, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)

            # Highlight gestures on finger tips
            if state.accelerate and len(pts) > 8:
                tip = pts[8]
                cv2.circle(frame, tip, 10, (0, 255, 120), -1, cv2.LINE_AA)
                cv2.circle(frame, tip, 14, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "GAS (UP)", (tip[0] + 12, tip[1] - 8),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 120), 1, cv2.LINE_AA)

            elif state.handbrake and len(pts) > 12:
                t1, t2 = pts[8], pts[12]
                cv2.circle(frame, t1, 8, (255, 120, 200), -1, cv2.LINE_AA)
                cv2.circle(frame, t2, 8, (255, 120, 200), -1, cv2.LINE_AA)
                cv2.putText(frame, "DRIFT", (t1[0] + 12, t1[1] - 8),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 120, 200), 1, cv2.LINE_AA)

            elif state.brake and len(pts) > 9:
                center_pt = pts[9]
                cv2.circle(frame, center_pt, 16, (60, 60, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "BRAKE", (center_pt[0] + 12, center_pt[1] - 8),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (60, 60, 255), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 3. MODERN OVERLAY BADGE (TOP LEFT)
        # -------------------------------------------------------------
        if state.accelerate:
            badge_text = "GAS ACTIVE [W / UP]"
            badge_color = (0, 200, 80)
        elif state.brake:
            badge_text = "BRAKE ACTIVE [S / DOWN]"
            badge_color = (40, 40, 230)
        elif state.handbrake:
            badge_text = "DRIFT ACTIVE [SPACE]"
            badge_color = (200, 80, 180)
        elif state.steer_left:
            badge_text = "STEERING LEFT [A / LEFT]"
            badge_color = (0, 180, 240)
        elif state.steer_right:
            badge_text = "STEERING RIGHT [D / RIGHT]"
            badge_color = (0, 180, 240)
        elif hand.is_detected:
            badge_text = "COASTING (NEUTRAL)"
            badge_color = (160, 160, 170)
        else:
            badge_text = "SHOW HAND IN CAMERA"
            badge_color = (60, 140, 220)

        # Draw semi-transparent pill for badge
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
        bx, by = 16, 16
        bw, bh = tw + 24, th + 18

        sub_img = frame[by:by+bh, bx:bx+bw]
        rect_color = np.zeros_like(sub_img)
        rect_color[:] = (18, 20, 26)
        cv2.addWeighted(rect_color, 0.75, sub_img, 0.25, 0, sub_img)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), badge_color, 1, cv2.LINE_AA)
        cv2.putText(frame, badge_text, (bx + 12, by + th + 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, badge_color, 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 4. STEERING BAR GAUGE (BOTTOM)
        # -------------------------------------------------------------
        bar_w = int(w * 0.60)
        bar_h = 10
        bar_x = (w - bar_w) // 2
        bar_y = h - 22

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (25, 28, 35), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 65, 75), 1)

        mid_x = bar_x + (bar_w // 2)
        cv2.line(frame, (mid_x, bar_y - 2), (mid_x, bar_y + bar_h + 2), (200, 200, 200), 2)

        steer_fill = int(state.steering_val * (bar_w // 2))
        if steer_fill != 0:
            steer_color = (0, 190, 255) if steer_fill < 0 else (0, 220, 255)
            x1 = min(mid_x, mid_x + steer_fill)
            x2 = max(mid_x, mid_x + steer_fill)
            cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_h), steer_color, -1)

        return frame
