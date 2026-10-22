"""
Main window interface for Burnout Camera Drive.
Simple, sleek, and intuitive hand-gesture racing controller.
Designed with a dark carbon racing aesthetic and clear live visual feedback.
"""

import sys
import os
import time
import webbrowser
import logging
from typing import Optional, Dict

import tkinter as tk
import customtkinter as ctk
from PIL import Image

from controller.camera import CameraThread, list_available_cameras
from controller.hand_tracking import HandTracker
from controller.gesture_detection import GestureDetector
from controller.calibration import CalibrationManager
from controller.keyboard_control import KeyboardController
from controller.session_stats import SessionTracker

logger = logging.getLogger("burnout.main_window")

# Modern Carbon & Neon Palette
BG_ROOT = "#0C0E12"
BG_CARD = "#141720"
BG_CARD_INNER = "#1B202D"
BG_CARD_ALT = "#222736"

ACCENT_BLUE = "#00D2FF"
ACCENT_GREEN = "#00FF88"
ACCENT_RED = "#FF3366"
ACCENT_AMBER = "#FFB800"
ACCENT_PURPLE = "#D946EF"

TEXT_WHITE = "#FFFFFF"
TEXT_MUTED = "#8A94A6"
BORDER_SUBTLE = "#262C3D"

GAME_URL = "https://poki.com/en/g/burnout-drift-hilltop"


class MainWindow(ctk.CTk):
    """
    Simple, intuitive racing controller dashboard.
    """

    def __init__(self):
        super().__init__()

        self.title("BURNOUT CAMERA DRIVE — Hand Gesture Controller")
        self.geometry("1060x720")
        self.minsize(960, 640)
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
        self.keyboard_ctrl.require_manual_resume = False
        self.keyboard_ctrl.check_focus = False
        self.session_tracker = SessionTracker()

        # Camera management
        self.available_cameras = list_available_cameras()
        self.current_cam_index = self.available_cameras[0] if self.available_cameras else 0
        self.camera_thread: Optional[CameraThread] = None

        # Driving state
        self.is_driving = False
        self._is_closing = False
        self._loop_after_id = None
        self.focus_game_only_var = tk.BooleanVar(value=False)

        # Build clean UI layout
        self._build_header()
        self._build_main_layout()

        # Global hotkey Escape for Emergency Stop
        self.bind("<Escape>", lambda event: self._on_emergency_stop())

        # Start Camera Thread & Session
        self.session_tracker.start_session(mode="live")
        self._start_camera_thread()

        # Start UI loop (~30 FPS)
        self._update_ui_loop()

        # Protocol for clean window exit
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_header(self):
        """Top navigation bar: branding and quick action buttons."""
        self.header = ctk.CTkFrame(self, fg_color=BG_CARD, height=60, corner_radius=0, border_width=1, border_color=BORDER_SUBTLE)
        self.header.pack(fill="x", side="top")

        # Left Branding
        brand_frame = ctk.CTkFrame(self.header, fg_color="transparent")
        brand_frame.pack(side="left", padx=18, pady=10)

        title_lbl = ctk.CTkLabel(
            brand_frame,
            text="🏎️ BURNOUT CAMERA DRIVE",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=TEXT_WHITE
        )
        title_lbl.pack(side="left")

        self.mode_tag = ctk.CTkLabel(
            brand_frame,
            text=" ⏸ STANDBY ",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            fg_color="#1E2430",
            text_color=TEXT_MUTED,
            corner_radius=6,
            padx=8,
            pady=2
        )
        self.mode_tag.pack(side="left", padx=12)

        # Right Action Buttons
        actions_frame = ctk.CTkFrame(self.header, fg_color="transparent")
        actions_frame.pack(side="right", padx=16, pady=10)

        # Open Game Button
        self.open_game_btn = ctk.CTkButton(
            actions_frame,
            text="🌐 OPEN GAME",
            fg_color="#006699",
            hover_color="#0088CC",
            text_color=TEXT_WHITE,
            font=ctk.CTkFont(size=12, weight="bold"),
            width=120,
            height=34,
            command=self._on_open_game
        )
        self.open_game_btn.pack(side="left", padx=5)

        # Recenter Button
        self.recenter_btn = ctk.CTkButton(
            actions_frame,
            text="🎯 RECENTER",
            fg_color="#1F2836",
            hover_color="#2B374A",
            text_color=ACCENT_BLUE,
            font=ctk.CTkFont(size=12, weight="bold"),
            width=105,
            height=34,
            command=self._on_recenter_steering
        )
        self.recenter_btn.pack(side="left", padx=5)

        # Emergency Stop Button
        self.emg_btn = ctk.CTkButton(
            actions_frame,
            text="🛑 STOP (ESC)",
            fg_color=ACCENT_RED,
            hover_color="#E02555",
            text_color=TEXT_WHITE,
            font=ctk.CTkFont(size=12, weight="bold"),
            width=110,
            height=34,
            command=self._on_emergency_stop
        )
        self.emg_btn.pack(side="left", padx=5)

    def _build_main_layout(self):
        """Two-column layout: Left = Live Video & Active Action, Right = Controls & Key Badges."""
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True, padx=16, pady=12)

        # ----------------- LEFT COLUMN: CAMERA & STEERING HUD -----------------
        self.left_col = ctk.CTkFrame(self.main_container, fg_color=BG_CARD, corner_radius=12, border_width=1, border_color=BORDER_SUBTLE)
        self.left_col.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=0)

        # Top Big Live Action Banner
        self.action_banner = ctk.CTkLabel(
            self.left_col,
            text="👋 SHOW YOUR HAND TO THE CAMERA TO DRIVE",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color=TEXT_WHITE,
            fg_color="#1B2230",
            corner_radius=8,
            height=44
        )
        self.action_banner.pack(fill="x", padx=12, pady=(12, 8))

        # Fixed Video Preview Box (pack_propagate=False prevents recursive ballooning!)
        self.video_box = ctk.CTkFrame(self.left_col, fg_color="#08090C", corner_radius=10, width=640, height=480)
        self.video_box.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.video_box.pack_propagate(False)

        self.video_label = ctk.CTkLabel(self.video_box, text="Connecting Camera...", text_color=TEXT_MUTED)
        self.video_label.place(relx=0.5, rely=0.5, anchor="center")

        # Steering Gauge Bar underneath video
        hud_bar_frame = ctk.CTkFrame(self.left_col, fg_color=BG_CARD_INNER, corner_radius=8, height=44)
        hud_bar_frame.pack(fill="x", padx=12, pady=(0, 12))

        gauge_labels = ctk.CTkFrame(hud_bar_frame, fg_color="transparent")
        gauge_labels.pack(fill="x", padx=12, pady=(4, 2))

        ctk.CTkLabel(gauge_labels, text="◀ STEER LEFT", font=ctk.CTkFont(size=11, weight="bold"), text_color=ACCENT_BLUE).pack(side="left")
        self.steer_lbl = ctk.CTkLabel(gauge_labels, text="CENTER (0.0°)", font=ctk.CTkFont(family="Consolas", size=12, weight="bold"), text_color=TEXT_WHITE)
        self.steer_lbl.pack(side="top")
        ctk.CTkLabel(gauge_labels, text="STEER RIGHT ▶", font=ctk.CTkFont(size=11, weight="bold"), text_color=ACCENT_BLUE).pack(side="right")

        self.steer_prog = ctk.CTkProgressBar(hud_bar_frame, height=8, progress_color=ACCENT_BLUE, fg_color="#0F1218")
        self.steer_prog.pack(fill="x", padx=12, pady=(2, 6))
        self.steer_prog.set(0.5)

        # ----------------- RIGHT COLUMN: CONTROL PANEL -----------------
        self.right_col = ctk.CTkFrame(self.main_container, fg_color=BG_CARD, corner_radius=12, width=350, border_width=1, border_color=BORDER_SUBTLE)
        self.right_col.pack(side="right", fill="y", padx=(0, 0), pady=0)
        self.right_col.pack_propagate(False)

        # Card 1: Master Drive Switch
        ctrl_card = ctk.CTkFrame(self.right_col, fg_color=BG_CARD_INNER, corner_radius=10)
        ctrl_card.pack(fill="x", padx=12, pady=(12, 8))

        self.drive_btn = ctk.CTkButton(
            ctrl_card,
            text="▶ START DRIVING",
            fg_color="#00A86B",
            hover_color="#00C87F",
            text_color=TEXT_WHITE,
            font=ctk.CTkFont(size=15, weight="bold"),
            height=50,
            command=self._toggle_driving
        )
        self.drive_btn.pack(fill="x", padx=12, pady=(12, 6))

        self.drive_hint_lbl = ctk.CTkLabel(
            ctrl_card,
            text="Click Start Driving, then show signs to drive!",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED
        )
        self.drive_hint_lbl.pack(padx=12, pady=(0, 10))

        # Card 2: Live Active Hardware Keys
        keys_card = ctk.CTkFrame(self.right_col, fg_color=BG_CARD_INNER, corner_radius=10)
        keys_card.pack(fill="x", padx=12, pady=(0, 8))

        ctk.CTkLabel(
            keys_card,
            text="ACTIVE KEYBOARD KEYS (LIVE)",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_MUTED
        ).pack(anchor="w", padx=12, pady=(10, 6))

        badges_grid = ctk.CTkFrame(keys_card, fg_color="transparent")
        badges_grid.pack(fill="x", padx=10, pady=(0, 10))

        self.key_badges: Dict[str, ctk.CTkLabel] = {}
        specs = [
            ("accelerate", "▲ W / UP", ACCENT_GREEN),
            ("brake", "▼ S / DOWN", ACCENT_RED),
            ("steer_left", "◀ A / LEFT", ACCENT_AMBER),
            ("steer_right", "▶ D / RIGHT", ACCENT_AMBER),
            ("handbrake", "⎵ SPACEBAR (DRIFT)", ACCENT_BLUE),
        ]

        # Top row: Left, Accel, Right
        r1 = ctk.CTkFrame(badges_grid, fg_color="transparent")
        r1.pack(fill="x", pady=2)
        for action, text, col in [specs[2], specs[0], specs[3]]:
            lbl = ctk.CTkLabel(r1, text=text, font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
                               fg_color="#141822", text_color="#454D5D", corner_radius=6, height=30)
            lbl.pack(side="left", fill="x", expand=True, padx=2)
            self.key_badges[action] = lbl

        # Bottom row: Brake, Spacebar
        r2 = ctk.CTkFrame(badges_grid, fg_color="transparent")
        r2.pack(fill="x", pady=2)
        for action, text, col in [specs[1], specs[4]]:
            lbl = ctk.CTkLabel(r2, text=text, font=ctk.CTkFont(family="Consolas", size=11, weight="bold"),
                               fg_color="#141822", text_color="#454D5D", corner_radius=6, height=30)
            lbl.pack(side="left", fill="x", expand=True, padx=2)
            self.key_badges[action] = lbl

        # Card 3: Signs Quick Guide
        guide_card = ctk.CTkFrame(self.right_col, fg_color=BG_CARD_INNER, corner_radius=10)
        guide_card.pack(fill="x", padx=12, pady=(0, 8))

        ctk.CTkLabel(
            guide_card,
            text="HAND SIGNS GUIDE",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_MUTED
        ).pack(anchor="w", padx=12, pady=(10, 4))

        signs = [
            ("☝️ Point Up (Index)", "Accelerate (Gas W / UP)"),
            ("✊ Fist", "Brake / Reverse (S / DOWN)"),
            ("✌️ Peace Sign", "Handbrake / Drift (SPACE)"),
            ("👈 / 👉 Tilt Hand", "Steer Left / Right (A / D)"),
            ("✋ Open Palm", "Coast (Keys Released)"),
        ]
        for icon_txt, desc in signs:
            row = ctk.CTkFrame(guide_card, fg_color="transparent")
            row.pack(fill="x", padx=12, pady=2)
            ctk.CTkLabel(row, text=icon_txt, font=ctk.CTkFont(size=12, weight="bold"), text_color=TEXT_WHITE).pack(side="left")
            ctk.CTkLabel(row, text=desc, font=ctk.CTkFont(size=11), text_color=TEXT_MUTED).pack(side="right")

        ctk.CTkLabel(guide_card, text="", height=4).pack()

        # Card 4: Settings & Options
        opts_card = ctk.CTkFrame(self.right_col, fg_color=BG_CARD_INNER, corner_radius=10)
        opts_card.pack(fill="x", padx=12, pady=(0, 10))

        # Mode Selector
        ctk.CTkLabel(opts_card, text="CONTROL MODE", font=ctk.CTkFont(size=11, weight="bold"), text_color=TEXT_MUTED).pack(anchor="w", padx=12, pady=(10, 2))
        self.mode_selector = ctk.CTkSegmentedButton(
            opts_card,
            values=["✋ Hand Signs", "📓 Notebook"],
            command=self._on_mode_change
        )
        self.mode_selector.pack(fill="x", padx=12, pady=(2, 6))
        self.mode_selector.set("✋ Hand Signs" if not self.calibration.notebook_steering_enabled else "📓 Notebook")

        # Steering Sensitivity Preset
        ctk.CTkLabel(opts_card, text="SENSITIVITY PRESET", font=ctk.CTkFont(size=11, weight="bold"), text_color=TEXT_MUTED).pack(anchor="w", padx=12, pady=(4, 2))
        self.preset_selector = ctk.CTkSegmentedButton(
            opts_card,
            values=["🔰 Beginner", "🏎️ Normal", "⚡ Pro"],
            command=self._on_preset_change
        )
        self.preset_selector.pack(fill="x", padx=12, pady=(2, 10))
        cur_preset = self.calibration.config.get("active_preset", "normal").lower()
        if cur_preset == "beginner":
            self.preset_selector.set("🔰 Beginner")
        elif cur_preset == "pro":
            self.preset_selector.set("⚡ Pro")
        else:
            self.preset_selector.set("🏎️ Normal")

    def _start_camera_thread(self):
        """Starts background worker thread for camera and hand tracking."""
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

    def _toggle_driving(self):
        """Toggles driving on or off."""
        if self.is_driving:
            self._on_stop_driving()
        else:
            self._on_start_driving()

    def _on_start_driving(self):
        self.is_driving = True
        self.keyboard_ctrl.check_focus = self.focus_game_only_var.get()
        self.keyboard_ctrl.set_enabled(True)
        self.keyboard_ctrl.rearm_driving()
        self.session_tracker.record_driving_state(True)

        self.drive_btn.configure(fg_color="#CC2936", hover_color="#E03A48", text="⏸ STOP DRIVING")
        self.drive_hint_lbl.configure(text="✔ DRIVING ACTIVE! Show signs to steer and press keys.", text_color=ACCENT_GREEN)
        self.mode_tag.configure(text=" 🟢 DRIVING ACTIVE ", fg_color="#0A3822", text_color=ACCENT_GREEN)
        logger.info("Driving started by user.")

    def _on_stop_driving(self):
        self.is_driving = False
        self.keyboard_ctrl.set_enabled(False)
        self.keyboard_ctrl.release_all()
        self.session_tracker.record_driving_state(False)

        self.drive_btn.configure(fg_color="#00A86B", hover_color="#00C87F", text="▶ START DRIVING")
        self.drive_hint_lbl.configure(text="Click Start Driving, then show signs to drive!", text_color=TEXT_MUTED)
        self.mode_tag.configure(text=" ⏸ STANDBY ", fg_color="#1E2430", text_color=TEXT_MUTED)
        logger.info("Driving stopped by user.")

    def _on_emergency_stop(self):
        """Immediately release all simulated keys."""
        self._on_stop_driving()
        self.keyboard_ctrl.emergency_stop()
        self.action_banner.configure(text="🛑 EMERGENCY STOP — ALL KEYS RELEASED", fg_color=ACCENT_RED, text_color=TEXT_WHITE)
        logger.info("Emergency Stop triggered.")

    def _on_open_game(self):
        """Opens Burnout Drift Hilltop in browser."""
        try:
            webbrowser.open(GAME_URL)
        except Exception as e:
            logger.error(f"Error opening browser: {e}")

    def _on_recenter_steering(self):
        """Instant recenter of hand or notebook steering."""
        if self.camera_thread:
            _, _, hand, nb, _, _ = self.camera_thread.get_latest_snapshot()
            msg = self.calibration.recenter_steering(nb_data=nb, hand_data=hand)
            self.action_banner.configure(text=f"🎯 {msg}", fg_color="#1B3828", text_color=ACCENT_GREEN)

    def _on_mode_change(self, mode: str):
        if "Notebook" in mode:
            self.calibration.set_notebook_steering(True)
            self.calibration.set_steering_mode("notebook")
        else:
            self.calibration.set_notebook_steering(False)
            self.calibration.set_steering_mode("hand")
        self.calibration.save()

    def _on_preset_change(self, preset: str):
        if "Beginner" in preset:
            self.calibration.apply_preset("beginner")
        elif "Pro" in preset:
            self.calibration.apply_preset("pro")
        else:
            self.calibration.apply_preset("normal")
        self.calibration.save()

    def _update_ui_loop(self):
        """Periodic UI update loop (~30 FPS)."""
        if self._is_closing:
            return

        if self.camera_thread:
            pil_img, state, hand, nb, fps, is_connected = self.camera_thread.get_latest_snapshot()

            # 1. Update Video Preview
            if pil_img:
                bw = self.video_box.winfo_width()
                bh = self.video_box.winfo_height()
                if bw < 50:
                    bw, bh = 640, 480

                target_w = bw
                target_h = int(bw * 0.75)
                if target_h > bh:
                    target_h = bh
                    target_w = int(bh * (4.0 / 3.0))

                target_w = max(320, min(640, target_w))
                target_h = max(240, min(480, target_h))

                resized = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
                ctk_img = ctk.CTkImage(light_image=resized, dark_image=resized, size=(target_w, target_h))
                self.video_label.configure(image=ctk_img, text="")
                self.video_label.image = ctk_img

            # 2. Update Hero Action Banner
            if not self.is_driving:
                if hand.is_detected:
                    self.action_banner.configure(
                        text=f"✋ STANDBY — Sign: {state.gesture_name} (Click 'START DRIVING' to enable keys)",
                        fg_color="#1E2430",
                        text_color=ACCENT_BLUE
                    )
                else:
                    self.action_banner.configure(
                        text="👋 SHOW HAND IN CAMERA • CLICK 'START DRIVING' TO PLAY",
                        fg_color="#161A24",
                        text_color=TEXT_MUTED
                    )
            else:
                # Active driving banner with bright colors
                if state.accelerate:
                    self.action_banner.configure(
                        text="☝️ GAS ACTIVE — [ W / UP ] IS BEING PRESSED",
                        fg_color="#0D4722",
                        text_color=ACCENT_GREEN
                    )
                elif state.brake:
                    self.action_banner.configure(
                        text="✊ BRAKE ACTIVE — [ S / DOWN ] IS BEING PRESSED",
                        fg_color="#5C1420",
                        text_color="#FF4D6D"
                    )
                elif state.handbrake:
                    self.action_banner.configure(
                        text="✌️ DRIFT ACTIVE — [ SPACEBAR ] IS BEING PRESSED",
                        fg_color="#113F5C",
                        text_color=ACCENT_BLUE
                    )
                elif state.steer_left:
                    self.action_banner.configure(
                        text=f"◀ STEERING LEFT ({abs(state.steering_val)*100:.0f}%) — [ A / LEFT ] PRESSED",
                        fg_color="#473200",
                        text_color=ACCENT_AMBER
                    )
                elif state.steer_right:
                    self.action_banner.configure(
                        text=f"▶ STEERING RIGHT ({abs(state.steering_val)*100:.0f}%) — [ D / RIGHT ] PRESSED",
                        fg_color="#473200",
                        text_color=ACCENT_AMBER
                    )
                elif hand.is_detected:
                    self.action_banner.configure(
                        text="✋ COASTING — [ ALL KEYS RELEASED ]",
                        fg_color="#182030",
                        text_color="#A0AEC0"
                    )
                else:
                    self.action_banner.configure(
                        text="👋 SHOW HAND TO CAMERA TO DRIVE",
                        fg_color="#201A10",
                        text_color=ACCENT_AMBER
                    )

            # 3. Update Steering Gauge
            pct = state.steering_val
            self.steer_prog.set(0.5 + (pct * 0.5))
            if abs(pct) < 0.05:
                self.steer_lbl.configure(text="CENTER (0.0°)", text_color=TEXT_WHITE)
            elif pct < 0:
                self.steer_lbl.configure(text=f"LEFT ({abs(pct)*100:.0f}%)", text_color=ACCENT_BLUE)
            else:
                self.steer_lbl.configure(text=f"RIGHT ({pct*100:.0f}%)", text_color=ACCENT_BLUE)

            # 4. Update Key Badges Glow
            active_actions = state.to_action_dict()
            for action_key, badge in self.key_badges.items():
                is_active = active_actions.get(action_key, False) and (self.is_driving or self.keyboard_ctrl.practice_mode)
                if is_active:
                    if action_key == "accelerate":
                        badge.configure(fg_color="#00A86B", text_color="#FFFFFF")
                    elif action_key == "brake":
                        badge.configure(fg_color="#E02555", text_color="#FFFFFF")
                    elif action_key in ("steer_left", "steer_right"):
                        badge.configure(fg_color="#D97706", text_color="#FFFFFF")
                    elif action_key == "handbrake":
                        badge.configure(fg_color="#0284C7", text_color="#FFFFFF")
                else:
                    badge.configure(fg_color="#141822", text_color="#454D5D")

        self._loop_after_id = self.after(33, self._update_ui_loop)

    def _on_close(self):
        """Clean application shutdown."""
        self._is_closing = True
        logger.info("Application closing...")
        if self._loop_after_id:
            try:
                self.after_cancel(self._loop_after_id)
            except Exception:
                pass

        if self.keyboard_ctrl:
            self.keyboard_ctrl.emergency_stop()

        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.join(timeout=1.0)

        if self.session_tracker:
            self.session_tracker.end_session()

        self.destroy()


if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()
