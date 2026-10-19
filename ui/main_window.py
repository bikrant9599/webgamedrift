"""
Main window interface for Burnout Camera Drive.
Designed with a dark charcoal and electric-blue racing aesthetic.
Features live camera HUD with animated virtual steering wheel,
Practice Mode vs Live Driving selector, Smart Guided Calibration,
Sensitivity Presets (Beginner, Normal, Pro), Session Statistics, and Safety Re-Arm.
"""

import sys
import os
import time
import webbrowser
import threading
import logging
from typing import Optional, List, Dict, Any

import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk

from controller.camera import CameraThread, list_available_cameras
from controller.hand_tracking import HandTracker
from controller.gesture_detection import GestureDetector
from controller.calibration import CalibrationManager
from controller.keyboard_control import KeyboardController
from controller.session_stats import SessionTracker
from ui.settings import SettingsWindow

logger = logging.getLogger("burnout.main_window")

# Theme Palette
BG_ROOT = "#0F1115"
BG_CARD = "#171A21"
BG_CARD_INNER = "#1F232D"
ACCENT_BLUE = "#00D2FF"
ACCENT_CYAN = "#00F0FF"
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

        self.title("BURNOUT CAMERA DRIVE — Smart Virtual Wheel & Hilltop Controller")
        self.geometry("1220x780")
        self.minsize(1080, 700)
        self.configure(fg_color=BG_ROOT)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Set application icon if present
        ico_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "app_icon.ico")
        if os.path.exists(ico_path):
            try:
                self.iconbitmap(ico_path)
            except Exception:
                pass

        # Core Engine Components
        self.calibration = CalibrationManager()
        self.hand_tracker = HandTracker(two_handed_mode=self.calibration.two_hand_mode_enabled)
        self.gesture_detector = GestureDetector(self.calibration)
        self.keyboard_ctrl = KeyboardController()
        self.session_tracker = SessionTracker()

        # Camera management
        self.available_cameras = list_available_cameras()
        self.current_cam_index = self.available_cameras[0] if self.available_cameras else 0
        self.camera_thread: Optional[CameraThread] = None

        # Driving state
        self.is_driving = False
        self.is_practice_mode = False
        self._is_closing = False
        self._loop_after_id = None

        # Build UI layout
        self._build_header()
        self._build_main_layout()

        # Bind global Escape key to Emergency Stop
        self.bind("<Escape>", lambda event: self._on_emergency_stop())

        # Start Camera Thread & Session
        self.session_tracker.start_session(mode="live")
        self._start_camera_thread()

        # Start UI loop
        self._update_ui_loop()

        # Protocol for clean window exit
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_header(self):
        """Top navigation and command bar."""
        self.header = ctk.CTkFrame(self, fg_color=BG_CARD, height=64, corner_radius=0, border_width=1, border_color=BORDER_SUBTLE)
        self.header.pack(fill="x", side="top")

        brand_frame = ctk.CTkFrame(self.header, fg_color="transparent")
        brand_frame.pack(side="left", padx=16, pady=10)

        title_lbl = ctk.CTkLabel(
            brand_frame,
            text="BURNOUT CAMERA DRIVE",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=TEXT_WHITE
        )
        title_lbl.pack(side="left")

        self.mode_tag = ctk.CTkLabel(
            brand_frame,
            text=" LIVE MODE ",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            fg_color="#0D2E47",
            text_color=ACCENT_BLUE,
            corner_radius=4
        )
        self.mode_tag.pack(side="left", padx=10)

        actions_frame = ctk.CTkFrame(self.header, fg_color="transparent")
        actions_frame.pack(side="right", padx=16, pady=10)

        # Mode Selector (Practice vs Live)
        self.practice_btn = ctk.CTkButton(
            actions_frame,
            text="🎯 PRACTICE MODE: OFF",
            fg_color="#1E2430",
            hover_color="#2A3242",
            text_color=TEXT_WHITE,
            font=ctk.CTkFont(size=11, weight="bold"),
            width=165,
            height=34,
            command=self._toggle_practice_mode
        )
        self.practice_btn.pack(side="left", padx=5)

        # Open Game Button
        self.open_game_btn = ctk.CTkButton(
            actions_frame,
            text="🌐 OPEN GAME",
            fg_color="#1E3048",
            hover_color="#274266",
            text_color=ACCENT_BLUE,
            font=ctk.CTkFont(weight="bold", size=11),
            width=115,
            height=34,
            command=self._on_open_game
        )
        self.open_game_btn.pack(side="left", padx=5)

        # Settings Button
        self.settings_btn = ctk.CTkButton(
            actions_frame,
            text="⚙ SETTINGS",
            fg_color="#252A34",
            hover_color="#333A48",
            text_color=TEXT_WHITE,
            font=ctk.CTkFont(weight="bold", size=11),
            width=95,
            height=34,
            command=self._open_settings
        )
        self.settings_btn.pack(side="left", padx=5)

        # Emergency Stop Button
        self.emg_btn = ctk.CTkButton(
            actions_frame,
            text="🛑 EMERGENCY STOP (ESC)",
            fg_color=ACCENT_RED,
            hover_color="#E02555",
            text_color=TEXT_WHITE,
            font=ctk.CTkFont(weight="bold", size=11),
            width=175,
            height=34,
            command=self._on_emergency_stop
        )
        self.emg_btn.pack(side="left", padx=5)

    def _build_main_layout(self):
        """Build two-column dashboard: Left = Video Preview & HUD, Right = Telemetry & Controls."""
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=16, pady=10)

        # ----------------- LEFT COLUMN: VIDEO & STEERING HUD -----------------
        self.left_col = ctk.CTkFrame(self.main_container, fg_color=BG_CARD, corner_radius=12, border_width=1, border_color=BORDER_SUBTLE)
        self.left_col.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=0)

        # Safety Re-Arm Alert Banner (shown when tracking drops during live driving)
        self.rearm_banner = ctk.CTkFrame(self.left_col, fg_color="#4A1521", corner_radius=8, height=36)
        self.rearm_banner_lbl = ctk.CTkLabel(
            self.rearm_banner,
            text="⚠️ TRACKING INTERRUPTED — INPUTS SAFELY PAUSED",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_WHITE
        )
        self.rearm_banner_lbl.pack(side="left", padx=12, pady=6)

        self.rearm_banner_btn = ctk.CTkButton(
            self.rearm_banner,
            text="RESUME DRIVING",
            fg_color=ACCENT_RED,
            hover_color="#FF4D79",
            font=ctk.CTkFont(size=11, weight="bold"),
            width=130,
            height=26,
            command=self._on_rearm_click
        )
        self.rearm_banner_btn.pack(side="right", padx=8, pady=5)
        # Hidden by default
        self.rearm_banner.pack_forget()

        # Video Frame Container
        self.video_box = ctk.CTkFrame(self.left_col, fg_color="#0B0C0E", corner_radius=8)
        self.video_box.pack(fill="both", expand=True, padx=10, pady=(10, 8))

        self.video_label = ctk.CTkLabel(self.video_box, text="Initializing Camera Stream...", text_color=TEXT_MUTED)
        self.video_label.pack(fill="both", expand=True, padx=2, pady=2)

        # Steering HUD Gauge underneath video
        hud_bar_frame = ctk.CTkFrame(self.left_col, fg_color=BG_CARD_INNER, corner_radius=8)
        hud_bar_frame.pack(fill="x", padx=10, pady=(0, 6))

        gauge_labels = ctk.CTkFrame(hud_bar_frame, fg_color="transparent")
        gauge_labels.pack(fill="x", padx=12, pady=(4, 2))

        ctk.CTkLabel(gauge_labels, text="◀ STEER LEFT", font=ctk.CTkFont(size=11, weight="bold"), text_color=ACCENT_BLUE).pack(side="left")
        self.steer_deg_lbl = ctk.CTkLabel(gauge_labels, text="CENTER (0%)", font=ctk.CTkFont(family="Consolas", size=12, weight="bold"), text_color=TEXT_WHITE)
        self.steer_deg_lbl.pack(side="top")
        ctk.CTkLabel(gauge_labels, text="STEER RIGHT ▶", font=ctk.CTkFont(size=11, weight="bold"), text_color=ACCENT_BLUE).pack(side="right")

        self.steer_prog = ctk.CTkProgressBar(hud_bar_frame, height=10, progress_color=ACCENT_BLUE, fg_color="#121418")
        self.steer_prog.pack(fill="x", padx=14, pady=(2, 6))
        self.steer_prog.set(0.5)

        # Active Keyboard HUD Badges
        keys_card = ctk.CTkFrame(self.left_col, fg_color=BG_CARD_INNER, corner_radius=8)
        keys_card.pack(fill="x", padx=10, pady=(0, 10))

        k_title_row = ctk.CTkFrame(keys_card, fg_color="transparent")
        k_title_row.pack(fill="x", padx=12, pady=(4, 2))
        ctk.CTkLabel(k_title_row, text="SIMULATED KEYBOARD INPUTS (HARDWARE SCAN CODES):", font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT_MUTED).pack(side="left")

        badges_row = ctk.CTkFrame(keys_card, fg_color="transparent")
        badges_row.pack(fill="x", padx=12, pady=(2, 6))

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
                padx=8
            )
            badge.pack(side="left", padx=3, expand=True, fill="x")
            self.key_badges[action_key] = badge

        # ----------------- RIGHT COLUMN: TELEMETRY & CONTROLS -----------------
        self.right_col = ctk.CTkScrollableFrame(self.main_container, fg_color=BG_CARD, corner_radius=12, width=410, border_width=1, border_color=BORDER_SUBTLE)
        self.right_col.pack(side="right", fill="both", padx=(8, 0), pady=0)

        # 1. DRIVING COMMAND CENTER
        self._create_panel_header(self.right_col, "DRIVING CONTROLLER STATUS")
        ctrl_card = ctk.CTkFrame(self.right_col, fg_color=BG_CARD_INNER, corner_radius=10)
        ctrl_card.pack(fill="x", padx=8, pady=(2, 8))

        self.start_btn = ctk.CTkButton(
            ctrl_card,
            text="▶ START DRIVING",
            fg_color="#00A86B",
            hover_color="#00C87F",
            text_color=TEXT_WHITE,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=44,
            command=self._on_start_driving
        )
        self.start_btn.pack(fill="x", padx=12, pady=(10, 5))

        self.stop_btn = ctk.CTkButton(
            ctrl_card,
            text="⏸ STOP DRIVING (STANDBY)",
            fg_color="#2A2F3D",
            hover_color="#363C4E",
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(size=12, weight="bold"),
            height=36,
            command=self._on_stop_driving
        )
        self.stop_btn.pack(fill="x", padx=12, pady=(0, 10))

        # 2. SENSITIVITY PRESETS ROW (1-CLICK)
        self._create_panel_header(self.right_col, "STEERING SENSITIVITY PRESETS")
        presets_box = ctk.CTkFrame(self.right_col, fg_color=BG_CARD_INNER, corner_radius=10)
        presets_box.pack(fill="x", padx=8, pady=(2, 8))

        p_row = ctk.CTkFrame(presets_box, fg_color="transparent")
        p_row.pack(fill="x", padx=8, pady=8)

        self.p_btn_beg = ctk.CTkButton(
            p_row, text="🔰 Beginner", width=85, height=30, font=ctk.CTkFont(size=11),
            fg_color="#222631", hover_color="#2E3544", command=lambda: self._apply_preset_from_main("beginner")
        )
        self.p_btn_beg.pack(side="left", padx=2, expand=True, fill="x")

        self.p_btn_norm = ctk.CTkButton(
            p_row, text="🏎️ Normal", width=85, height=30, font=ctk.CTkFont(size=11),
            fg_color="#0077BE", hover_color="#0090E0", command=lambda: self._apply_preset_from_main("normal")
        )
        self.p_btn_norm.pack(side="left", padx=2, expand=True, fill="x")

        self.p_btn_pro = ctk.CTkButton(
            p_row, text="⚡ Pro", width=85, height=30, font=ctk.CTkFont(size=11),
            fg_color="#222631", hover_color="#2E3544", command=lambda: self._apply_preset_from_main("pro")
        )
        self.p_btn_pro.pack(side="left", padx=2, expand=True, fill="x")

        # Two-hand virtual wheel toggle button
        wheel_toggle_row = ctk.CTkFrame(presets_box, fg_color="transparent")
        wheel_toggle_row.pack(fill="x", padx=10, pady=(0, 8))
        self.wheel_mode_btn = ctk.CTkButton(
            wheel_toggle_row,
            text="🚗 Two-Hand Virtual Wheel: OFF",
            fg_color="#252A34",
            hover_color="#333A48",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._toggle_two_hand_wheel
        )
        self.wheel_mode_btn.pack(fill="x", pady=(0, 4))

        self.notebook_mode_btn = ctk.CTkButton(
            wheel_toggle_row,
            text=f"📓 Notebook Steering: {'ON' if self.calibration.notebook_steering_enabled else 'OFF'}",
            fg_color="#D97706" if self.calibration.notebook_steering_enabled else "#252A34",
            hover_color="#F59E0B" if self.calibration.notebook_steering_enabled else "#333A48",
            text_color="#000000" if self.calibration.notebook_steering_enabled else TEXT_WHITE,
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._toggle_notebook_mode
        )
        self.notebook_mode_btn.pack(fill="x")

        # 3. LIVE TELEMETRY CARD
        self._create_panel_header(self.right_col, "LIVE SYSTEM TELEMETRY")
        telem_card = ctk.CTkFrame(self.right_col, fg_color=BG_CARD_INNER, corner_radius=10)
        telem_card.pack(fill="x", padx=8, pady=(2, 8))

        self.cam_status_lbl = self._create_telemetry_row(telem_card, "Camera Feed:", "CHECKING...", ACCENT_AMBER)
        self.tracking_status_lbl = self._create_telemetry_row(telem_card, "Hand Tracking:", "SEARCHING...", ACCENT_AMBER)
        self.notebook_status_lbl = self._create_telemetry_row(telem_card, "Notebook Wheel:", "SEARCHING...", ACCENT_AMBER)
        self.right_index_lbl = self._create_telemetry_row(telem_card, "Right Index Finger:", "LOWERED", TEXT_MUTED)
        self.gesture_lbl = self._create_telemetry_row(telem_card, "Current Gesture:", "IDLE", TEXT_WHITE)
        self.throttle_lbl = self._create_telemetry_row(telem_card, "Acceleration:", "RELEASED", TEXT_MUTED)
        self.brake_lbl = self._create_telemetry_row(telem_card, "Braking Status:", "RELEASED", TEXT_MUTED)
        self.fps_lbl = self._create_telemetry_row(telem_card, "Engine FPS:", "0.0 FPS", ACCENT_BLUE)

        # 4. SMART CALIBRATION WIZARD CARD
        self._create_panel_header(self.right_col, "SMART CALIBRATION WIZARD")
        calib_card = ctk.CTkFrame(self.right_col, fg_color=BG_CARD_INNER, corner_radius=10)
        calib_card.pack(fill="x", padx=8, pady=(2, 8))

        calib_desc = ctk.CTkLabel(
            calib_card,
            text="Calibrate notebook steering wheel or hands. Steering is muted during calibration for safety.",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
            wraplength=340,
            justify="left"
        )
        calib_desc.pack(padx=12, pady=(8, 6), anchor="w")

        self.calib_prog_bar = ctk.CTkProgressBar(calib_card, height=8, progress_color=ACCENT_BLUE, fg_color="#121418")
        self.calib_prog_bar.pack(fill="x", padx=12, pady=(0, 6))
        self.calib_prog_bar.set(0.0)

        self.calib_msg_lbl = ctk.CTkLabel(calib_card, text="Status: Ready to calibrate", font=ctk.CTkFont(size=11, weight="bold"), text_color=ACCENT_GREEN)
        self.calib_msg_lbl.pack(padx=12, pady=(0, 6), anchor="w")

        # Row 1 of calibration: Notebook Calibration Wizard
        calib_nb_row = ctk.CTkFrame(calib_card, fg_color="transparent")
        calib_nb_row.pack(fill="x", padx=10, pady=(2, 4))

        self.calib_notebook_btn = ctk.CTkButton(
            calib_nb_row,
            text="📓 CALIBRATE NOTEBOOK WHEEL (4-STEP)",
            fg_color="#D97706",
            hover_color="#F59E0B",
            text_color="#000000",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=34,
            command=self._on_start_notebook_calibration
        )
        self.calib_notebook_btn.pack(fill="x")

        # Row 2 of calibration: Quick Hand & Guided Hand buttons
        calib_btn_row = ctk.CTkFrame(calib_card, fg_color="transparent")
        calib_btn_row.pack(fill="x", padx=10, pady=(0, 10))

        self.calib_quick_btn = ctk.CTkButton(
            calib_btn_row,
            text="🎯 Quick Hand (2.5s)",
            fg_color="#1A3A54",
            hover_color="#234E72",
            text_color=ACCENT_BLUE,
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._on_start_quick_calibration
        )
        self.calib_quick_btn.pack(side="left", padx=2, expand=True, fill="x")

        self.calib_guided_btn = ctk.CTkButton(
            calib_btn_row,
            text="📐 Hand Range (3-Step)",
            fg_color="#252A34",
            hover_color="#333A48",
            text_color=TEXT_WHITE,
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._on_start_guided_calibration
        )
        self.calib_guided_btn.pack(side="left", padx=2, expand=True, fill="x")

        # 5. SESSION STATISTICS CARD
        self._create_panel_header(self.right_col, "SESSION STATISTICS & HISTORY")
        stats_card = ctk.CTkFrame(self.right_col, fg_color=BG_CARD_INNER, corner_radius=10)
        stats_card.pack(fill="x", padx=8, pady=(2, 8))

        self.sess_timer_lbl = self._create_telemetry_row(stats_card, "Session Duration:", "00:00:00", TEXT_WHITE)
        self.sess_driving_lbl = self._create_telemetry_row(stats_card, "Driving Time:", "00:00:00", ACCENT_GREEN)
        self.sess_interrupt_lbl = self._create_telemetry_row(stats_card, "Tracking Interruptions:", "0", ACCENT_AMBER)

        hist_row = ctk.CTkFrame(stats_card, fg_color="transparent")
        hist_row.pack(fill="x", padx=10, pady=(4, 8))
        self.view_hist_btn = ctk.CTkButton(
            hist_row,
            text="📜 View Past Session History",
            fg_color="#222631",
            hover_color="#2E3544",
            font=ctk.CTkFont(size=11),
            command=self._show_session_history_dialog
        )
        self.view_hist_btn.pack(fill="x")

        # Privacy guarantee note
        priv_lbl = ctk.CTkLabel(
            stats_card,
            text="🔒 100% Local: No webcam frames or personal data are ever uploaded.",
            font=ctk.CTkFont(size=9),
            text_color=TEXT_MUTED
        )
        priv_lbl.pack(padx=10, pady=(0, 6), anchor="w")

        # 6. QUICK SENSITIVITY SLIDER
        self._create_panel_header(self.right_col, "SENSITIVITY & DEADZONE ADJUSTMENT")
        tune_card = ctk.CTkFrame(self.right_col, fg_color=BG_CARD_INNER, corner_radius=10)
        tune_card.pack(fill="x", padx=8, pady=(2, 12))

        tune_row = ctk.CTkFrame(tune_card, fg_color="transparent")
        tune_row.pack(fill="x", padx=12, pady=(8, 2))
        ctk.CTkLabel(tune_row, text="Steering Sensitivity:", font=ctk.CTkFont(size=12), text_color=TEXT_WHITE).pack(side="left")
        self.quick_sens_val = ctk.CTkLabel(
            tune_row,
            text=f"{self.calibration.steering_sensitivity:.1f}x",
            font=ctk.CTkFont(family="Consolas", size=12, weight="bold"),
            text_color=ACCENT_BLUE
        )
        self.quick_sens_val.pack(side="right")

        self.quick_sens_slider = ctk.CTkSlider(
            tune_card, from_=0.4, to=2.5, number_of_steps=21,
            progress_color=ACCENT_BLUE, button_color=ACCENT_BLUE,
            command=self._on_quick_sens_change
        )
        self.quick_sens_slider.pack(fill="x", padx=12, pady=(0, 10))
        self.quick_sens_slider.set(self.calibration.steering_sensitivity)

    def _create_panel_header(self, parent, text: str):
        lbl = ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color=ACCENT_BLUE)
        lbl.pack(anchor="w", padx=10, pady=(8, 2))

    def _create_telemetry_row(self, parent, label_text: str, default_val: str, val_color: str):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=3)
        ctk.CTkLabel(row, text=label_text, font=ctk.CTkFont(size=12), text_color=TEXT_MUTED).pack(side="left")
        val_lbl = ctk.CTkLabel(row, text=default_val, font=ctk.CTkFont(family="Consolas", size=12, weight="bold"), text_color=val_color)
        val_lbl.pack(side="right")
        return val_lbl

    def _start_camera_thread(self):
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.stop()
            self.camera_thread.join(timeout=1.0)

        self.camera_thread = CameraThread(
            camera_index=self.current_cam_index,
            hand_tracker=self.hand_tracker,
            gesture_detector=self.gesture_detector,
            calibration=self.calibration,
            keyboard_ctrl=self.keyboard_ctrl,
            session_tracker=self.session_tracker,
            width=640,
            height=480,
            fps=30
        )
        self.camera_thread.start()

    def _toggle_practice_mode(self):
        self.is_practice_mode = not self.is_practice_mode
        self.keyboard_ctrl.set_practice_mode(self.is_practice_mode)

        if self.is_practice_mode:
            self.practice_btn.configure(text="🎯 PRACTICE MODE: ON", fg_color="#0077BE", hover_color="#0090E0")
            self.mode_tag.configure(text=" PRACTICE MODE ", fg_color="#103B5C", text_color=ACCENT_CYAN)
        else:
            self.practice_btn.configure(text="🎯 PRACTICE MODE: OFF", fg_color="#1E2430", hover_color="#2A3242")
            self.mode_tag.configure(text=" LIVE MODE ", fg_color="#0D2E47", text_color=ACCENT_BLUE)

        logger.info(f"Practice mode toggled: {self.is_practice_mode}")

    def _apply_preset_from_main(self, preset_key: str):
        self.calibration.apply_preset(preset_key)
        self.quick_sens_slider.set(self.calibration.steering_sensitivity)
        self.quick_sens_val.configure(text=f"{self.calibration.steering_sensitivity:.1f}x")

        # Update button highlights
        self.p_btn_beg.configure(fg_color="#0077BE" if preset_key == "beginner" else "#222631")
        self.p_btn_norm.configure(fg_color="#0077BE" if preset_key == "normal" else "#222631")
        self.p_btn_pro.configure(fg_color="#0077BE" if preset_key == "pro" else "#222631")

    def _toggle_two_hand_wheel(self):
        curr = self.calibration.two_hand_mode_enabled
        new_val = not curr
        self.calibration.set_two_hand_mode(new_val)
        self.hand_tracker.set_two_handed_mode(new_val)
        status_str = "ON (with 1-hand fallback)" if new_val else "OFF"
        self.wheel_mode_btn.configure(text=f"🚗 Two-Hand Virtual Wheel: {status_str}", fg_color="#0077BE" if new_val else "#252A34")

    def _toggle_notebook_mode(self):
        curr = self.calibration.notebook_steering_enabled
        new_val = not curr
        self.calibration.set_notebook_steering_enabled(new_val)
        status_str = "ON" if new_val else "OFF"
        self.notebook_mode_btn.configure(
            text=f"📓 Notebook Steering: {status_str}",
            fg_color="#D97706" if new_val else "#252A34",
            hover_color="#F59E0B" if new_val else "#333A48",
            text_color="#000000" if new_val else TEXT_WHITE
        )
        logger.info(f"Notebook steering toggled: {status_str}")

    def _on_start_driving(self):
        self.is_driving = True
        self.keyboard_ctrl.set_enabled(True)
        self.keyboard_ctrl.rearm_driving()
        self.session_tracker.record_driving_state(True)
        self.start_btn.configure(fg_color="#008050", text="✔ DRIVING ACTIVE")
        self.stop_btn.configure(fg_color="#363C4E", text_color=TEXT_WHITE)
        self.rearm_banner.pack_forget()
        logger.info("Driving enabled by user.")

    def _on_stop_driving(self):
        self.is_driving = False
        self.keyboard_ctrl.set_enabled(False)
        self.keyboard_ctrl.release_all()
        self.session_tracker.record_driving_state(False)
        self.start_btn.configure(fg_color="#00A86B", text="▶ START DRIVING")
        self.stop_btn.configure(fg_color="#2A2F3D", text_color=TEXT_MUTED)
        self.rearm_banner.pack_forget()
        logger.info("Driving stopped by user.")

    def _on_emergency_stop(self):
        self.is_driving = False
        self.keyboard_ctrl.emergency_stop()
        self.session_tracker.record_driving_state(False)
        self.start_btn.configure(fg_color="#00A86B", text="▶ START DRIVING")
        self.stop_btn.configure(fg_color="#2A2F3D", text_color=TEXT_MUTED)
        self.gesture_lbl.configure(text="EMERGENCY STOPPED", text_color=ACCENT_RED)
        self.rearm_banner.pack_forget()
        logger.warning("Emergency Stop Triggered.")

    def _on_rearm_click(self):
        self.keyboard_ctrl.rearm_driving()
        self.rearm_banner.pack_forget()
        logger.info("User re-armed driving after interruption.")

    def _on_open_game(self):
        try:
            webbrowser.open(GAME_URL)
        except Exception as e:
            logger.error(f"Error opening browser: {e}")

    def _on_start_notebook_calibration(self):
        self.calibration.start_notebook_calibration()
        self.calib_notebook_btn.configure(text="📓 Calibrating Notebook...", state="disabled")

    def _on_start_quick_calibration(self):
        self.calibration.start_quick_calibration()
        self.calib_quick_btn.configure(text="Sampling...", state="disabled")

    def _on_start_guided_calibration(self):
        self.calibration.start_guided_calibration()
        self.calib_guided_btn.configure(text="Sampling...", state="disabled")

    def _on_quick_sens_change(self, val: float):
        self.calibration.set_steering_sensitivity(val)
        self.quick_sens_val.configure(text=f"{val:.1f}x")

    def _open_settings(self):
        SettingsWindow(
            parent=self,
            calibration_mgr=self.calibration,
            keyboard_ctrl=self.keyboard_ctrl,
            hand_tracker=self.hand_tracker,
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
        status_str = "ON" if self.calibration.two_hand_mode_enabled else "OFF"
        self.wheel_mode_btn.configure(
            text=f"🚗 Two-Hand Virtual Wheel: {status_str}",
            fg_color="#0077BE" if self.calibration.two_hand_mode_enabled else "#252A34"
        )
        nb_str = "ON" if self.calibration.notebook_steering_enabled else "OFF"
        self.notebook_mode_btn.configure(
            text=f"📓 Notebook Steering: {nb_str}",
            fg_color="#D97706" if self.calibration.notebook_steering_enabled else "#252A34",
            hover_color="#F59E0B" if self.calibration.notebook_steering_enabled else "#333A48",
            text_color="#000000" if self.calibration.notebook_steering_enabled else TEXT_WHITE
        )

    def _show_session_history_dialog(self):
        """Display past driving sessions in a clean modal dialog."""
        history = self.session_tracker.load_history()
        win = ctk.CTkToplevel(self)
        win.title("Driving Session History")
        win.geometry("520x460")
        win.configure(fg_color=BG_ROOT)
        win.transient(self)

        lbl = ctk.CTkLabel(win, text="LOCAL DRIVING SESSIONS (RECENT 25)", font=ctk.CTkFont(size=14, weight="bold"), text_color=ACCENT_BLUE)
        lbl.pack(pady=12)

        scr = ctk.CTkScrollableFrame(win, fg_color=BG_CARD)
        scr.pack(fill="both", expand=True, padx=14, pady=8)

        if not history:
            ctk.CTkLabel(scr, text="No completed sessions yet. Start driving to build history!", text_color=TEXT_MUTED).pack(pady=30)
        else:
            for s in history:
                row = ctk.CTkFrame(scr, fg_color=BG_CARD_INNER, corner_radius=6)
                row.pack(fill="x", padx=6, pady=4)
                ctk.CTkLabel(row, text=f"📅 {s.get('timestamp', 'Unknown')}", font=ctk.CTkFont(size=11, weight="bold"), text_color=TEXT_WHITE).pack(anchor="w", padx=10, pady=(6, 2))
                stats_str = f"Mode: {s.get('mode', 'live').upper()} | Duration: {s.get('duration_seconds', 0)}s | Driving Time: {s.get('driving_time_seconds', 0)}s | Interruptions: {s.get('interruptions', 0)}"
                ctk.CTkLabel(row, text=stats_str, font=ctk.CTkFont(size=10), text_color=TEXT_MUTED).pack(anchor="w", padx=10, pady=(0, 6))

        btn_row = ctk.CTkFrame(win, fg_color="transparent")
        btn_row.pack(fill="x", padx=14, pady=10)

        def clear_and_refresh():
            self.session_tracker.clear_history()
            win.destroy()

        ctk.CTkButton(btn_row, text="Clear History", fg_color="#363C4E", hover_color=ACCENT_RED, command=clear_and_refresh).pack(side="left")
        ctk.CTkButton(btn_row, text="Close", fg_color=ACCENT_BLUE, hover_color="#00B4DB", text_color="#000", command=win.destroy).pack(side="right")

    def _update_ui_loop(self):
        """Periodic UI update loop (~30 FPS)."""
        if self.camera_thread:
            pil_img, state, hand, nb, fps, is_connected = self.camera_thread.get_latest_snapshot()

            # 1. Video Preview
            if pil_img:
                box_w = max(400, self.video_box.winfo_width())
                box_h = max(300, self.video_box.winfo_height())
                scaled_w = box_w
                scaled_h = int(box_w * 0.75)
                if scaled_h > box_h:
                    scaled_h = box_h
                    scaled_w = int(box_h * (4.0 / 3.0))

                resized = pil_img.resize((scaled_w, scaled_h), Image.Resampling.BILINEAR)
                ctk_img = ctk.CTkImage(light_image=resized, dark_image=resized, size=(scaled_w, scaled_h))
                self.video_label.configure(image=ctk_img, text="")
                self.video_label.image = ctk_img

            # 2. Connection Status
            if is_connected:
                self.cam_status_lbl.configure(text=f"CONNECTED (CAM #{self.current_cam_index})", text_color=ACCENT_GREEN)
            else:
                self.cam_status_lbl.configure(text="DISCONNECTED", text_color=ACCENT_RED)

            # 3. Hand & Notebook Tracking Status & Indicators
            if hand.is_detected:
                if hand.is_two_handed:
                    self.tracking_status_lbl.configure(text="LOCKED [2 HANDS WHEEL]", text_color=ACCENT_GREEN)
                elif hand.is_fallback:
                    self.tracking_status_lbl.configure(text="LOCKED [1-HAND FALLBACK]", text_color=ACCENT_CYAN)
                else:
                    self.tracking_status_lbl.configure(text=f"LOCKED ({hand.handedness})", text_color=ACCENT_GREEN)
            else:
                self.tracking_status_lbl.configure(text="SEARCHING HAND...", text_color=ACCENT_AMBER)

            # Notebook Wheel Status
            if nb and nb.is_detected:
                nb_rot_str = f"{nb.rotation_angle:+.1f}°"
                self.notebook_status_lbl.configure(text=f"LOCKED ({nb_rot_str})", text_color=ACCENT_GREEN)
            else:
                self.notebook_status_lbl.configure(text="SEARCHING NOTEBOOK...", text_color=ACCENT_AMBER)

            # Right Index Finger Status
            if hand.right_index_raised:
                self.right_index_lbl.configure(text="▲ RAISED (GAS ACTIVE)", text_color=ACCENT_GREEN)
            elif hand.is_detected:
                self.right_index_lbl.configure(text="LOWERED (COAST)", text_color=TEXT_MUTED)
            else:
                self.right_index_lbl.configure(text="SEARCHING HAND...", text_color=TEXT_MUTED)

            # 4. Gesture Name
            self.gesture_lbl.configure(text=state.gesture_name)
            if state.brake:
                self.gesture_lbl.configure(text_color=ACCENT_RED)
            elif state.accelerate:
                self.gesture_lbl.configure(text_color=ACCENT_GREEN)
            elif state.handbrake:
                self.gesture_lbl.configure(text_color="#FF66CC")
            else:
                self.gesture_lbl.configure(text_color=TEXT_WHITE)

            # 5. Throttle & Brake
            self.throttle_lbl.configure(
                text="ACTIVE (100%)" if state.accelerate else "RELEASED",
                text_color=ACCENT_GREEN if state.accelerate else TEXT_MUTED
            )
            self.brake_lbl.configure(
                text="ACTIVE (BRAKING)" if state.brake else "RELEASED",
                text_color=ACCENT_RED if state.brake else TEXT_MUTED
            )

            # 6. FPS
            self.fps_lbl.configure(text=f"{fps:.1f} FPS")

            # 7. Steering Gauge
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

            # 8. Active Badges
            active_keys = self.keyboard_ctrl.get_active_keys()
            mapping = self.keyboard_ctrl.get_mapping()

            for action_key, badge in self.key_badges.items():
                mapped_physical = mapping.get(action_key, "")
                is_active = (mapped_physical in active_keys) and (self.is_driving or self.is_practice_mode)

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

            # 9. Calibration State & Feedback
            if self.calibration.is_calibrating:
                msg, prog = self.camera_thread.calibration_feedback
                self.calib_prog_bar.set(prog)
                self.calib_msg_lbl.configure(text=msg, text_color=ACCENT_BLUE)
            else:
                self.calib_prog_bar.set(1.0 if self.calibration.config.get("is_calibrated") else 0.0)
                if self.calibration.config.get("notebook_calibrated"):
                    neutral = float(self.calibration.config.get("notebook_neutral_angle", 0.0))
                    self.calib_msg_lbl.configure(text=f"Notebook Calibrated (Neutral: {neutral:+.1f}°)", text_color=ACCENT_GREEN)
                else:
                    self.calib_msg_lbl.configure(text="Neutral Centered & Calibrated", text_color=ACCENT_GREEN)
                self.calib_notebook_btn.configure(text="📓 CALIBRATE NOTEBOOK WHEEL (4-STEP)", state="normal")
                self.calib_quick_btn.configure(text="🎯 Quick Hand (2.5s)", state="normal")
                self.calib_guided_btn.configure(text="📐 Hand Range (3-Step)", state="normal")

            # 10. Safety Re-Arm Banner check
            if self.keyboard_ctrl.is_interrupted and self.is_driving and not self.is_practice_mode:
                self.rearm_banner.pack(fill="x", padx=10, pady=(6, 2), before=self.video_box)
            else:
                self.rearm_banner.pack_forget()

            # 11. Live Session Stats Update
            live_stats = self.session_tracker.get_live_stats()
            self.sess_timer_lbl.configure(text=live_stats["elapsed_str"])
            self.sess_driving_lbl.configure(text=live_stats["driving_time_str"])
            self.sess_interrupt_lbl.configure(text=str(live_stats["interruptions"]))

        if not getattr(self, '_is_closing', False):
            self._loop_after_id = self.after(33, self._update_ui_loop)

    def _on_close(self):
        logger.info("Application closing...")
        self._is_closing = True
        if getattr(self, '_loop_after_id', None):
            try:
                self.after_cancel(self._loop_after_id)
            except Exception:
                pass
        self.session_tracker.end_session()
        self.keyboard_ctrl.shutdown()
        if self.camera_thread:
            self.camera_thread.stop()
        self.hand_tracker.close()
        try:
            self.destroy()
        except Exception:
            pass
        self.quit()
