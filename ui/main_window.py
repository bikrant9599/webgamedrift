"""
Main window interface for Burnout Camera Drive.
Designed with a dark charcoal and electric-blue racing aesthetic.
Features live camera HUD, real-time telemetry, active key indicators,
calibration wizard, and direct game launch.
"""

import sys
import os
import time
import webbrowser
import threading
import logging
from typing import Optional, List

import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk

from controller.camera import CameraThread, list_available_cameras
from controller.hand_tracking import HandTracker
from controller.gesture_detection import GestureDetector
from controller.calibration import CalibrationManager
from controller.keyboard_control import KeyboardController
from ui.settings import SettingsWindow

logger = logging.getLogger("burnout.main_window")

# Theme Palette
BG_ROOT = "#0F1115"
BG_CARD = "#171A21"
BG_CARD_INNER = "#1F232D"
ACCENT_BLUE = "#00D2FF"
ACCENT_GREEN = "#00FF88"
ACCENT_RED = "#FF3366"
ACCENT_AMBER = "#FFB800"
TEXT_WHITE = "#FFFFFF"
TEXT_MUTED = "#8A94A6"
BORDER_SUBTLE = "#282E3B"

GAME_URL = "https://poki.com/en/g/burnout-drift-hilltop"


class MainWindow(ctk.CTk):
    """
    Main racing interface window.
    """

    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("BURNOUT CAMERA DRIVE — Hilltop Drift Controller")
        self.geometry("1180x740")
        self.minsize(1050, 680)
        self.configure(fg_color=BG_ROOT)

        # Set CustomTkinter theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Set application icon
        ico_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "app_icon.ico")
        if os.path.exists(ico_path):
            try:
                self.iconbitmap(ico_path)
            except Exception:
                pass

        # Core Engine Components
        self.calibration = CalibrationManager()
        self.hand_tracker = HandTracker()
        self.gesture_detector = GestureDetector(self.calibration)
        self.keyboard_ctrl = KeyboardController()

        # Camera management
        self.available_cameras = list_available_cameras()
        self.current_cam_index = self.available_cameras[0] if self.available_cameras else 0
        self.camera_thread: Optional[CameraThread] = None

        # Driving state
        self.is_driving = False

        # Build UI layout
        self._build_header()
        self._build_main_layout()

        # Keyboard safety: Bind global Escape key to Emergency Stop
        self.bind("<Escape>", lambda event: self._on_emergency_stop())

        # Start Camera Thread
        self._start_camera_thread()

        # Start telemetry & video update loop
        self._update_ui_loop()

        # Protocol for clean window exit
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_header(self):
        """Top navigation and command bar."""
        self.header = ctk.CTkFrame(self, fg_color=BG_CARD, height=64, corner_radius=0, border_width=1, border_color=BORDER_SUBTLE)
        self.header.pack(fill="x", side="top")

        # Left brand branding
        brand_frame = ctk.CTkFrame(self.header, fg_color="transparent")
        brand_frame.pack(side="left", padx=20, pady=10)

        title_lbl = ctk.CTkLabel(
            brand_frame,
            text="BURNOUT CAMERA DRIVE",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=TEXT_WHITE
        )
        title_lbl.pack(side="left")

        tag_badge = ctk.CTkLabel(
            brand_frame,
            text=" HILLTOP EDITION ",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            fg_color="#0D2E47",
            text_color=ACCENT_BLUE,
            corner_radius=4
        )
        tag_badge.pack(side="left", padx=10)

        # Right utility actions
        actions_frame = ctk.CTkFrame(self.header, fg_color="transparent")
        actions_frame.pack(side="right", padx=20, pady=10)

        # Open Game in Browser Button
        self.open_game_btn = ctk.CTkButton(
            actions_frame,
            text="🌐 OPEN GAME",
            fg_color="#1E3048",
            hover_color="#274266",
            text_color=ACCENT_BLUE,
            font=ctk.CTkFont(weight="bold", size=12),
            width=120,
            height=34,
            command=self._on_open_game
        )
        self.open_game_btn.pack(side="left", padx=6)

        # Settings Button
        self.settings_btn = ctk.CTkButton(
            actions_frame,
            text="⚙ SETTINGS",
            fg_color="#252A34",
            hover_color="#333A48",
            text_color=TEXT_WHITE,
            font=ctk.CTkFont(weight="bold", size=12),
            width=100,
            height=34,
            command=self._open_settings
        )
        self.settings_btn.pack(side="left", padx=6)

        # Emergency Stop Button (Always visible crimson red button)
        self.emg_btn = ctk.CTkButton(
            actions_frame,
            text="🛑 EMERGENCY STOP (ESC)",
            fg_color=ACCENT_RED,
            hover_color="#E02555",
            text_color=TEXT_WHITE,
            font=ctk.CTkFont(weight="bold", size=12),
            width=180,
            height=34,
            command=self._on_emergency_stop
        )
        self.emg_btn.pack(side="left", padx=6)

    def _build_main_layout(self):
        """Build two-column dashboard: Left = Video & HUD, Right = Telemetry & Controls."""
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=16, pady=12)

        # ----------------- LEFT COLUMN: VIDEO & STEERING HUD -----------------
        self.left_col = ctk.CTkFrame(self.main_container, fg_color=BG_CARD, corner_radius=12, border_width=1, border_color=BORDER_SUBTLE)
        self.left_col.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=0)

        # Video Frame Container
        self.video_box = ctk.CTkFrame(self.left_col, fg_color="#0B0C0E", corner_radius=8)
        self.video_box.pack(fill="both", expand=True, padx=12, pady=12)

        self.video_label = ctk.CTkLabel(self.video_box, text="Initializing Camera Stream...", text_color=TEXT_MUTED)
        self.video_label.pack(fill="both", expand=True, padx=2, pady=2)

        # Steering HUD Gauge underneath video
        hud_bar_frame = ctk.CTkFrame(self.left_col, fg_color=BG_CARD_INNER, corner_radius=8)
        hud_bar_frame.pack(fill="x", padx=12, pady=(0, 8))

        gauge_labels = ctk.CTkFrame(hud_bar_frame, fg_color="transparent")
        gauge_labels.pack(fill="x", padx=12, pady=(6, 2))

        ctk.CTkLabel(gauge_labels, text="◀ STEER LEFT", font=ctk.CTkFont(size=11, weight="bold"), text_color=ACCENT_BLUE).pack(side="left")
        self.steer_deg_lbl = ctk.CTkLabel(gauge_labels, text="CENTER (0%)", font=ctk.CTkFont(family="Consolas", size=12, weight="bold"), text_color=TEXT_WHITE)
        self.steer_deg_lbl.pack(side="top")
        ctk.CTkLabel(gauge_labels, text="STEER RIGHT ▶", font=ctk.CTkFont(size=11, weight="bold"), text_color=ACCENT_BLUE).pack(side="right")

        # Steering progress bar with center zero
        self.steer_prog = ctk.CTkProgressBar(hud_bar_frame, height=10, progress_color=ACCENT_BLUE, fg_color="#121418")
        self.steer_prog.pack(fill="x", padx=14, pady=(2, 8))
        self.steer_prog.set(0.5)  # 0.5 is center

        # Real-time Active Keyboard HUD Badges
        keys_card = ctk.CTkFrame(self.left_col, fg_color=BG_CARD_INNER, corner_radius=8)
        keys_card.pack(fill="x", padx=12, pady=(0, 12))

        k_title_row = ctk.CTkFrame(keys_card, fg_color="transparent")
        k_title_row.pack(fill="x", padx=12, pady=(6, 2))
        ctk.CTkLabel(k_title_row, text="SIMULATED KEYBOARD INPUTS (HARDWARE SCAN CODES):", font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT_MUTED).pack(side="left")

        badges_row = ctk.CTkFrame(keys_card, fg_color="transparent")
        badges_row.pack(fill="x", padx=12, pady=(2, 8))

        self.key_badges = {}
        badge_specs = [
            ("accelerate", "▲ ACCEL (UP)"),
            ("brake", "▼ BRAKE (DOWN)"),
            ("steer_left", "◀ LEFT"),
            ("steer_right", "▶ RIGHT"),
            ("handbrake", "⎵ DRIFT (SPACE)")
        ]

        for action_key, label_text in badge_specs:
            badge = ctk.CTkLabel(
                badges_row,
                text=label_text,
                font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
                fg_color="#171A21",
                text_color="#555E6D",
                corner_radius=6,
                height=26,
                padx=10
            )
            badge.pack(side="left", padx=4, expand=True, fill="x")
            self.key_badges[action_key] = badge

        # ----------------- RIGHT COLUMN: TELEMETRY & CONTROLS -----------------
        self.right_col = ctk.CTkScrollableFrame(self.main_container, fg_color=BG_CARD, corner_radius=12, width=380, border_width=1, border_color=BORDER_SUBTLE)
        self.right_col.pack(side="right", fill="both", padx=(8, 0), pady=0)

        # 1. DRIVING COMMAND CENTER
        self._create_panel_header(self.right_col, "DRIVING CONTROLLER STATUS")
        ctrl_card = ctk.CTkFrame(self.right_col, fg_color=BG_CARD_INNER, corner_radius=10)
        ctrl_card.pack(fill="x", padx=10, pady=(2, 10))

        # Main Start / Stop Buttons
        self.start_btn = ctk.CTkButton(
            ctrl_card,
            text="▶ START DRIVING",
            fg_color="#00A86B",
            hover_color="#00C87F",
            text_color=TEXT_WHITE,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=46,
            command=self._on_start_driving
        )
        self.start_btn.pack(fill="x", padx=12, pady=(12, 6))

        self.stop_btn = ctk.CTkButton(
            ctrl_card,
            text="⏸ STOP DRIVING (STANDBY)",
            fg_color="#2A2F3D",
            hover_color="#363C4E",
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38,
            command=self._on_stop_driving
        )
        self.stop_btn.pack(fill="x", padx=12, pady=(0, 12))

        # 2. LIVE TELEMETRY CARD
        self._create_panel_header(self.right_col, "SYSTEM TELEMETRY")
        telem_card = ctk.CTkFrame(self.right_col, fg_color=BG_CARD_INNER, corner_radius=10)
        telem_card.pack(fill="x", padx=10, pady=(2, 10))

        self.cam_status_lbl = self._create_telemetry_row(telem_card, "Camera Feed:", "CHECKING...", ACCENT_AMBER)
        self.tracking_status_lbl = self._create_telemetry_row(telem_card, "Hand Tracking:", "SEARCHING...", ACCENT_AMBER)
        self.gesture_lbl = self._create_telemetry_row(telem_card, "Current Gesture:", "IDLE", TEXT_WHITE)
        self.throttle_lbl = self._create_telemetry_row(telem_card, "Acceleration:", "RELEASED", TEXT_MUTED)
        self.brake_lbl = self._create_telemetry_row(telem_card, "Braking Status:", "RELEASED", TEXT_MUTED)
        self.fps_lbl = self._create_telemetry_row(telem_card, "Engine FPS:", "0.0 FPS", ACCENT_BLUE)

        # 3. QUICK CALIBRATION WIZARD
        self._create_panel_header(self.right_col, "CALIBRATION WIZARD")
        calib_card = ctk.CTkFrame(self.right_col, fg_color=BG_CARD_INNER, corner_radius=10)
        calib_card.pack(fill="x", padx=10, pady=(2, 10))

        calib_desc = ctk.CTkLabel(
            calib_card,
            text="Hold your hand in your natural resting steering position and click Calibrate to establish neutral center.",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
            wraplength=320,
            justify="left"
        )
        calib_desc.pack(padx=12, pady=(10, 8), anchor="w")

        self.calib_prog_bar = ctk.CTkProgressBar(calib_card, height=8, progress_color=ACCENT_BLUE, fg_color="#121418")
        self.calib_prog_bar.pack(fill="x", padx=12, pady=(0, 8))
        self.calib_prog_bar.set(0.0)

        self.calib_msg_lbl = ctk.CTkLabel(calib_card, text="Status: Ready to calibrate", font=ctk.CTkFont(size=11, weight="bold"), text_color=ACCENT_GREEN)
        self.calib_msg_lbl.pack(padx=12, pady=(0, 8), anchor="w")

        self.calib_btn = ctk.CTkButton(
            calib_card,
            text="🎯 CALIBRATE NEUTRAL POSITION",
            fg_color="#1A3A54",
            hover_color="#234E72",
            text_color=ACCENT_BLUE,
            font=ctk.CTkFont(size=12, weight="bold"),
            height=36,
            command=self._on_start_calibration
        )
        self.calib_btn.pack(fill="x", padx=12, pady=(0, 12))

        # 4. QUICK STEERING SENSITIVITY SLIDER
        self._create_panel_header(self.right_col, "QUICK TUNING")
        tune_card = ctk.CTkFrame(self.right_col, fg_color=BG_CARD_INNER, corner_radius=10)
        tune_card.pack(fill="x", padx=10, pady=(2, 10))

        tune_row = ctk.CTkFrame(tune_card, fg_color="transparent")
        tune_row.pack(fill="x", padx=12, pady=(10, 4))
        ctk.CTkLabel(tune_row, text="Steering Sensitivity:", font=ctk.CTkFont(size=12), text_color=TEXT_WHITE).pack(side="left")
        self.quick_sens_val = ctk.CTkLabel(
            tune_row,
            text=f"{self.calibration.steering_sensitivity:.1f}x",
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
            text_color=ACCENT_BLUE
        )
        self.quick_sens_val.pack(side="right")

        self.quick_sens_slider = ctk.CTkSlider(
            tune_card,
            from_=0.5, to=2.5,
            number_of_steps=20,
            progress_color=ACCENT_BLUE,
            button_color=ACCENT_BLUE,
            command=self._on_quick_sens_change
        )
        self.quick_sens_slider.pack(fill="x", padx=12, pady=(0, 12))
        self.quick_sens_slider.set(self.calibration.steering_sensitivity)

        # 5. GAME DRIVING INSTRUCTIONS CARD
        self._create_panel_header(self.right_col, "HOW TO PLAY WITH YOUR WEBCAM")
        guide_card = ctk.CTkFrame(self.right_col, fg_color=BG_CARD_INNER, corner_radius=10)
        guide_card.pack(fill="x", padx=10, pady=(2, 16))

        instructions = [
            ("1. OPEN GAME", "Click the 'OPEN GAME' button above to launch Poki."),
            ("2. FOCUS CANVAS", "Click once on the game screen in your browser."),
            ("3. START DRIVING", "Click 'START DRIVING' in this app."),
            ("4. STEER", "Move hand left or right of center to turn."),
            ("5. ACCELERATE", "Push open hand forward towards webcam."),
            ("6. BRAKE", "Form a tight fist to brake instantly."),
            ("7. DRIFT / HANDBRAKE", "Flash Peace sign ✌ for drift turn."),
            ("8. STOP / EMERGENCY", "Press Escape or red button to release all keys.")
        ]

        for step_title, step_desc in instructions:
            step_row = ctk.CTkFrame(guide_card, fg_color="transparent")
            step_row.pack(fill="x", padx=10, pady=3)
            ctk.CTkLabel(step_row, text=step_title, font=ctk.CTkFont(size=10, weight="bold"), text_color=ACCENT_BLUE, width=120, anchor="w").pack(side="left")
            ctk.CTkLabel(step_row, text=step_desc, font=ctk.CTkFont(size=10), text_color=TEXT_MUTED, anchor="w").pack(side="left", fill="x", expand=True)

    def _create_panel_header(self, parent, text: str):
        lbl = ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=ACCENT_BLUE
        )
        lbl.pack(anchor="w", padx=12, pady=(10, 2))

    def _create_telemetry_row(self, parent, label_text: str, default_val: str, val_color: str):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=4)
        ctk.CTkLabel(row, text=label_text, font=ctk.CTkFont(size=12), text_color=TEXT_MUTED).pack(side="left")
        val_lbl = ctk.CTkLabel(
            row,
            text=default_val,
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
            text_color=val_color
        )
        val_lbl.pack(side="right")
        return val_lbl

    def _start_camera_thread(self):
        """Initialize and start background worker."""
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.stop()
            self.camera_thread.join(timeout=1.0)

        self.camera_thread = CameraThread(
            camera_index=self.current_cam_index,
            hand_tracker=self.hand_tracker,
            gesture_detector=self.gesture_detector,
            calibration=self.calibration,
            keyboard_ctrl=self.keyboard_ctrl,
            width=640,
            height=480,
            fps=30
        )
        self.camera_thread.start()

    def _on_start_driving(self):
        self.is_driving = True
        self.keyboard_ctrl.set_enabled(True)
        self.start_btn.configure(fg_color="#008050", text="✔ DRIVING ACTIVE")
        self.stop_btn.configure(fg_color="#363C4E", text_color=TEXT_WHITE)
        logger.info("Driving enabled by user.")

    def _on_stop_driving(self):
        self.is_driving = False
        self.keyboard_ctrl.set_enabled(False)
        self.keyboard_ctrl.release_all()
        self.start_btn.configure(fg_color="#00A86B", text="▶ START DRIVING")
        self.stop_btn.configure(fg_color="#2A2F3D", text_color=TEXT_MUTED)
        logger.info("Driving stopped by user. All keys released.")

    def _on_emergency_stop(self):
        """Immediate killswitch: releases all keys and locks inputs."""
        self.is_driving = False
        self.keyboard_ctrl.emergency_stop()
        self.start_btn.configure(fg_color="#00A86B", text="▶ START DRIVING")
        self.stop_btn.configure(fg_color="#2A2F3D", text_color=TEXT_MUTED)
        self.gesture_lbl.configure(text="EMERGENCY STOPPED", text_color=ACCENT_RED)
        logger.warning("EMERGENCY STOP BUTTON PRESSED: All simulated inputs terminated.")

    def _on_open_game(self):
        """Open Poki game in user's default browser."""
        try:
            webbrowser.open(GAME_URL)
            logger.info(f"Opened game URL: {GAME_URL}")
        except Exception as e:
            logger.error(f"Error opening browser: {e}")

    def _on_start_calibration(self):
        self.calibration.start_quick_calibration()
        self.calib_btn.configure(text="HOLD STEADY...", state="disabled")

    def _on_quick_sens_change(self, val: float):
        self.calibration.set_steering_sensitivity(val)
        self.quick_sens_val.configure(text=f"{val:.1f}x")

    def _open_settings(self):
        SettingsWindow(
            parent=self,
            calibration_mgr=self.calibration,
            keyboard_ctrl=self.keyboard_ctrl,
            available_cameras=self.available_cameras,
            current_cam_index=self.current_cam_index,
            on_cam_change_cb=self._on_switch_camera,
            on_settings_applied_cb=self._on_settings_applied
        )

    def _on_switch_camera(self, new_idx: int):
        if new_idx != self.current_cam_index:
            self.current_cam_index = new_idx
            self._start_camera_thread()

    def _on_settings_applied(self):
        self.quick_sens_slider.set(self.calibration.steering_sensitivity)
        self.quick_sens_val.configure(text=f"{self.calibration.steering_sensitivity:.1f}x")

    def _update_ui_loop(self):
        """Periodic UI update (runs on main thread via Tkinter after)."""
        if self.camera_thread:
            pil_img, state, hand, fps, is_connected = self.camera_thread.get_latest_snapshot()

            # 1. Update Video Preview
            if pil_img:
                # Scale appropriately for display box
                box_w = max(400, self.video_box.winfo_width())
                box_h = max(300, self.video_box.winfo_height())
                # Maintain 4:3 aspect ratio
                scaled_w = box_w
                scaled_h = int(box_w * 0.75)
                if scaled_h > box_h:
                    scaled_h = box_h
                    scaled_w = int(box_h * (4.0 / 3.0))

                resized = pil_img.resize((scaled_w, scaled_h), Image.Resampling.BILINEAR)
                ctk_img = ctk.CTkImage(light_image=resized, dark_image=resized, size=(scaled_w, scaled_h))
                self.video_label.configure(image=ctk_img, text="")
                self.video_label.image = ctk_img

            # 2. Update Connection Status
            if is_connected:
                self.cam_status_lbl.configure(text=f"CONNECTED (CAM #{self.current_cam_index})", text_color=ACCENT_GREEN)
            else:
                self.cam_status_lbl.configure(text="DISCONNECTED", text_color=ACCENT_RED)

            # 3. Update Hand Status
            if hand.is_detected:
                self.tracking_status_lbl.configure(text=f"LOCKED ({hand.handedness})", text_color=ACCENT_GREEN)
            else:
                self.tracking_status_lbl.configure(text="SEARCHING HAND...", text_color=ACCENT_AMBER)

            # 4. Update Current Gesture
            self.gesture_lbl.configure(text=state.gesture_name)
            if state.brake:
                self.gesture_lbl.configure(text_color=ACCENT_RED)
            elif state.accelerate:
                self.gesture_lbl.configure(text_color=ACCENT_GREEN)
            elif state.handbrake:
                self.gesture_lbl.configure(text_color="#FF66CC")
            else:
                self.gesture_lbl.configure(text_color=TEXT_WHITE)

            # 5. Update Acceleration / Brake Status
            if state.accelerate:
                self.throttle_lbl.configure(text="ACTIVE (100%)", text_color=ACCENT_GREEN)
            else:
                self.throttle_lbl.configure(text="RELEASED", text_color=TEXT_MUTED)

            if state.brake:
                self.brake_lbl.configure(text="ACTIVE (FIST)", text_color=ACCENT_RED)
            else:
                self.brake_lbl.configure(text="RELEASED", text_color=TEXT_MUTED)

            # 6. Update FPS
            self.fps_lbl.configure(text=f"{fps:.1f} FPS")

            # 7. Update Steering Gauge
            # Map [-1.0 ... +1.0] to [0.0 ... 1.0] for CTkProgressBar
            prog_val = (state.steering_val + 1.0) / 2.0
            self.steer_prog.set(max(0.0, min(1.0, prog_val)))

            if state.steer_left:
                pct = int(abs(state.steering_val) * 100)
                self.steer_deg_lbl.configure(text=f"LEFT ({pct}%)", text_color=ACCENT_BLUE)
            elif state.steer_right:
                pct = int(abs(state.steering_val) * 100)
                self.steer_deg_lbl.configure(text=f"RIGHT ({pct}%)", text_color=ACCENT_BLUE)
            else:
                self.steer_deg_lbl.configure(text="CENTER (DEADZONE)", text_color=TEXT_MUTED)

            # 8. Update Active Keyboard Badges (HUD)
            active_keys = self.keyboard_ctrl.get_active_keys()
            mapping = self.keyboard_ctrl.get_mapping()

            for action_key, badge in self.key_badges.items():
                mapped_physical = mapping.get(action_key, "")
                is_active = (mapped_physical in active_keys) and self.is_driving

                if is_active:
                    if action_key == "brake":
                        badge.configure(fg_color=ACCENT_RED, text_color=TEXT_WHITE)
                    elif action_key == "accelerate":
                        badge.configure(fg_color="#00A86B", text_color=TEXT_WHITE)
                    elif action_key == "handbrake":
                        badge.configure(fg_color="#B5179E", text_color=TEXT_WHITE)
                    else:
                        badge.configure(fg_color="#0077BE", text_color=TEXT_WHITE)
                else:
                    badge.configure(fg_color="#171A21", text_color="#555E6D")

            # 9. Update Calibration Progress
            if self.calibration.is_calibrating:
                msg, prog = self.camera_thread.calibration_feedback
                self.calib_prog_bar.set(prog)
                self.calib_msg_lbl.configure(text=msg, text_color=ACCENT_BLUE)
            else:
                self.calib_prog_bar.set(1.0 if self.calibration.config.get("is_calibrated") else 0.0)
                self.calib_msg_lbl.configure(text="Neutral Centered & Calibrated", text_color=ACCENT_GREEN)
                self.calib_btn.configure(text="🎯 CALIBRATE NEUTRAL POSITION", state="normal")

        # Schedule next tick at ~30 FPS (33ms)
        self.after(33, self._update_ui_loop)

    def _on_close(self):
        """Safe exit: release all keys, stop camera thread, close app."""
        logger.info("Application closing...")
        self.keyboard_ctrl.shutdown()
        if self.camera_thread:
            self.camera_thread.stop()
        self.hand_tracker.close()
        try:
            self.destroy()
        except Exception:
            pass
        self.quit()
