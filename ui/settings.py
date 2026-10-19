"""
Settings Dialog for Burnout Camera Drive.
Allows tuning steering deadzone, sensitivity presets (Beginner, Normal, Pro),
two-hand virtual wheel mode, gesture triggers, keyboard controls,
safety re-arm behavior, camera selection, and Reset to Defaults with validation.
"""

import tkinter as tk
import customtkinter as ctk
from typing import Callable, Dict, Any, List

BG_DARK = "#121417"
PANEL_BG = "#1A1D24"
ACCENT_BLUE = "#00D2FF"
ACCENT_CYAN = "#00F0FF"
ACCENT_GREEN = "#00FF88"
ACCENT_RED = "#FF3366"
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#9BA3AF"
BORDER_COLOR = "#2A303C"
BUTTON_HOVER = "#00B4DB"


class SettingsWindow(ctk.CTkToplevel):
    """
    Comprehensive racing-styled configuration dialog.
    """

    def __init__(self, parent,
                 calibration_mgr,
                 keyboard_ctrl,
                 hand_tracker,
                 available_cameras: List[int],
                 current_cam_index: int,
                 on_cam_change_cb: Callable[[int], None],
                 on_settings_applied_cb: Callable[[], None] = None):
        super().__init__(parent)

        self.calibration = calibration_mgr
        self.keyboard = keyboard_ctrl
        self.hand_tracker = hand_tracker
        self.available_cameras = available_cameras
        self.current_cam_index = current_cam_index
        self.on_cam_change_cb = on_cam_change_cb
        self.on_settings_applied_cb = on_settings_applied_cb

        self.title("Burnout Camera Drive — Advanced Settings & Presets")
        self.geometry("620x760")
        self.minsize(580, 680)
        self.configure(fg_color=BG_DARK)

        self.transient(parent)
        self.grab_set()

        self._build_ui()
        self._load_current_values()

    def _build_ui(self):
        # Header banner
        header_frame = ctk.CTkFrame(self, fg_color=PANEL_BG, corner_radius=0, height=54)
        header_frame.pack(fill="x", side="top")

        title_lbl = ctk.CTkLabel(
            header_frame,
            text="SYSTEM CALIBRATION & DRIVING PRESETS",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            text_color=ACCENT_BLUE
        )
        title_lbl.pack(side="left", padx=20, pady=12)

        # Scrollable content frame
        self.scroll = ctk.CTkScrollableFrame(self, fg_color=BG_DARK, corner_radius=0)
        self.scroll.pack(fill="both", expand=True, padx=16, pady=10)

        # 1. QUICK SENSITIVITY PRESETS
        self._create_section_label(self.scroll, "SENSITIVITY PRESETS (1-CLICK TUNING)")
        preset_frame = ctk.CTkFrame(self.scroll, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER_COLOR)
        preset_frame.pack(fill="x", pady=4, padx=4)

        preset_row = ctk.CTkFrame(preset_frame, fg_color="transparent")
        preset_row.pack(fill="x", padx=12, pady=10)

        self.btn_beginner = ctk.CTkButton(
            preset_row, text="🔰 Beginner\nSmooth & Stable",
            fg_color="#232731", hover_color="#2D3340", font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: self._apply_preset_choice("beginner")
        )
        self.btn_beginner.pack(side="left", padx=3, expand=True, fill="x")

        self.btn_normal = ctk.CTkButton(
            preset_row, text="🏎️ Normal\nBalanced Drift",
            fg_color="#232731", hover_color="#2D3340", font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: self._apply_preset_choice("normal")
        )
        self.btn_normal.pack(side="left", padx=3, expand=True, fill="x")

        self.btn_pro = ctk.CTkButton(
            preset_row, text="⚡ Pro\nUltra Responsive",
            fg_color="#232731", hover_color="#2D3340", font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: self._apply_preset_choice("pro")
        )
        self.btn_pro.pack(side="left", padx=3, expand=True, fill="x")

        # 2. NOTEBOOK PHYSICAL STEERING WHEEL
        self._create_section_label(self.scroll, "NOTEBOOK PHYSICAL STEERING WHEEL")
        nb_frame = ctk.CTkFrame(self.scroll, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER_COLOR)
        nb_frame.pack(fill="x", pady=4, padx=4)

        nb_switch_row = ctk.CTkFrame(nb_frame, fg_color="transparent")
        nb_switch_row.pack(fill="x", padx=12, pady=8)
        self.nb_steering_switch = ctk.CTkSwitch(
            nb_switch_row,
            text="Enable Physical Notebook Steering (Webcam Tracking)",
            font=ctk.CTkFont(size=12, weight="bold"),
            progress_color="#D97706",
            command=self._on_nb_switch_toggle
        )
        self.nb_steering_switch.pack(side="left")

        # Deadzone slider
        nb_dz_row = ctk.CTkFrame(nb_frame, fg_color="transparent")
        nb_dz_row.pack(fill="x", padx=12, pady=(4, 2))
        self.nb_dz_lbl = ctk.CTkLabel(nb_dz_row, text="Notebook Steering Deadzone: 5.0°", font=ctk.CTkFont(size=12), text_color=TEXT_PRIMARY)
        self.nb_dz_lbl.pack(side="left")
        self.nb_dz_slider = ctk.CTkSlider(
            nb_frame, from_=1.0, to=15.0, number_of_steps=14,
            progress_color="#D97706", button_color="#F59E0B", command=self._on_nb_dz_change
        )
        self.nb_dz_slider.pack(fill="x", padx=14, pady=2)

        # Max Steering Angle slider
        nb_max_row = ctk.CTkFrame(nb_frame, fg_color="transparent")
        nb_max_row.pack(fill="x", padx=12, pady=(4, 2))
        self.nb_max_lbl = ctk.CTkLabel(nb_max_row, text="Max Steering Angle (100% Turn): 32.0°", font=ctk.CTkFont(size=12), text_color=TEXT_PRIMARY)
        self.nb_max_lbl.pack(side="left")
        self.nb_max_slider = ctk.CTkSlider(
            nb_frame, from_=15.0, to=50.0, number_of_steps=35,
            progress_color="#D97706", button_color="#F59E0B", command=self._on_nb_max_change
        )
        self.nb_max_slider.pack(fill="x", padx=14, pady=2)

        # Color Profile dropdown
        nb_color_row = ctk.CTkFrame(nb_frame, fg_color="transparent")
        nb_color_row.pack(fill="x", padx=12, pady=(4, 10))
        ctk.CTkLabel(nb_color_row, text="Notebook Color Profile:", font=ctk.CTkFont(size=12), text_color=TEXT_PRIMARY).pack(side="left")
        self.nb_color_menu = ctk.CTkOptionMenu(
            nb_color_row,
            values=["Orange (Default)", "Blue / Dark Cover", "Green / Bright Cover", "Red / Crimson Cover", "Any Color / Edge Adaptive"],
            fg_color="#232731", button_color="#D97706", button_hover_color="#F59E0B",
            command=self._on_nb_color_change
        )
        self.nb_color_menu.pack(side="right")

        # 3. VIRTUAL STEERING WHEEL & MODE
        self._create_section_label(self.scroll, "VIRTUAL STEERING WHEEL & INPUT MODES")
        mode_frame = ctk.CTkFrame(self.scroll, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER_COLOR)
        mode_frame.pack(fill="x", pady=4, padx=4)

        two_hand_row = ctk.CTkFrame(mode_frame, fg_color="transparent")
        two_hand_row.pack(fill="x", padx=12, pady=8)
        self.two_hand_switch = ctk.CTkSwitch(
            two_hand_row,
            text="Enable Two-Hand Virtual Steering Wheel (with 1-hand fallback)",
            font=ctk.CTkFont(size=12, weight="bold"),
            progress_color=ACCENT_BLUE,
            command=self._on_two_hand_toggle
        )
        self.two_hand_switch.pack(side="left")

        mode_row = ctk.CTkFrame(mode_frame, fg_color="transparent")
        mode_row.pack(fill="x", padx=12, pady=(0, 8))
        ctk.CTkLabel(mode_row, text="Single-Hand Mode Style:", font=ctk.CTkFont(size=12), text_color=TEXT_PRIMARY).pack(side="left")
        self.mode_menu = ctk.CTkOptionMenu(
            mode_row,
            values=["Horizontal Position (Left/Right)", "Steering Wheel Tilt (Angle)"],
            fg_color="#232731", button_color=ACCENT_BLUE, button_hover_color=BUTTON_HOVER,
            command=self._on_mode_change
        )
        self.mode_menu.pack(side="right")

        # 3. STEERING DYNAMICS & DEADZONES
        self._create_section_label(self.scroll, "STEERING SENSITIVITY & DEADZONES")
        steer_frame = ctk.CTkFrame(self.scroll, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER_COLOR)
        steer_frame.pack(fill="x", pady=4, padx=4)

        # Sensitivity
        sens_row = ctk.CTkFrame(steer_frame, fg_color="transparent")
        sens_row.pack(fill="x", padx=12, pady=(8, 2))
        self.sens_lbl = ctk.CTkLabel(sens_row, text="Steering Sensitivity: 1.0x", font=ctk.CTkFont(size=12), text_color=TEXT_PRIMARY)
        self.sens_lbl.pack(side="left")
        self.sens_slider = ctk.CTkSlider(
            steer_frame, from_=0.4, to=2.5, number_of_steps=21,
            progress_color=ACCENT_BLUE, button_color=ACCENT_CYAN, command=self._on_sens_change
        )
        self.sens_slider.pack(fill="x", padx=14, pady=2)

        # Deadzone
        dz_row = ctk.CTkFrame(steer_frame, fg_color="transparent")
        dz_row.pack(fill="x", padx=12, pady=(6, 2))
        self.dz_lbl = ctk.CTkLabel(dz_row, text="Center Deadzone: 8%", font=ctk.CTkFont(size=12), text_color=TEXT_PRIMARY)
        self.dz_lbl.pack(side="left")
        self.dz_slider = ctk.CTkSlider(
            steer_frame, from_=0.02, to=0.20, number_of_steps=18,
            progress_color=ACCENT_BLUE, button_color=ACCENT_CYAN, command=self._on_dz_change
        )
        self.dz_slider.pack(fill="x", padx=14, pady=(2, 10))

        # 4. GESTURE CONFIGURATION
        self._create_section_label(self.scroll, "GESTURE TRIGGERS & DEBOUNCING")
        gest_frame = ctk.CTkFrame(self.scroll, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER_COLOR)
        gest_frame.pack(fill="x", pady=4, padx=4)

        # Throttle Gesture
        acc_row = ctk.CTkFrame(gest_frame, fg_color="transparent")
        acc_row.pack(fill="x", padx=12, pady=6)
        ctk.CTkLabel(acc_row, text="Throttle (Acceleration):", font=ctk.CTkFont(size=12), text_color=TEXT_PRIMARY).pack(side="left")
        self.acc_menu = ctk.CTkOptionMenu(
            acc_row,
            values=["Forward Push (Depth)", "Forward Up (Y Lift)", "Index Point Forward"],
            fg_color="#232731", button_color=ACCENT_BLUE, button_hover_color=BUTTON_HOVER,
            command=self._on_acc_change
        )
        self.acc_menu.pack(side="right")

        # Brake Gesture
        brk_row = ctk.CTkFrame(gest_frame, fg_color="transparent")
        brk_row.pack(fill="x", padx=12, pady=6)
        ctk.CTkLabel(brk_row, text="Braking Trigger:", font=ctk.CTkFont(size=12), text_color=TEXT_PRIMARY).pack(side="left")
        self.brk_menu = ctk.CTkOptionMenu(
            brk_row,
            values=["Full Fist (All Fingers Curled)", "Thumbs Down"],
            fg_color="#232731", button_color=ACCENT_BLUE, button_hover_color=BUTTON_HOVER,
            command=self._on_brk_change
        )
        self.brk_menu.pack(side="right")

        # Handbrake Gesture
        hb_row = ctk.CTkFrame(gest_frame, fg_color="transparent")
        hb_row.pack(fill="x", padx=12, pady=6)
        ctk.CTkLabel(hb_row, text="Handbrake (Drift):", font=ctk.CTkFont(size=12), text_color=TEXT_PRIMARY).pack(side="left")
        self.hb_menu = ctk.CTkOptionMenu(
            hb_row,
            values=["Peace / Victory Sign (Two Fingers)", "Shaka (Thumb + Pinky)", "Index-Thumb Pinch", "Pinky Extended"],
            fg_color="#232731", button_color=ACCENT_BLUE, button_hover_color=BUTTON_HOVER,
            command=self._on_hb_change
        )
        self.hb_menu.pack(side="right")

        # Confirmation debounce
        deb_row = ctk.CTkFrame(gest_frame, fg_color="transparent")
        deb_row.pack(fill="x", padx=12, pady=6)
        ctk.CTkLabel(deb_row, text="Gesture Confirmation Delay:", font=ctk.CTkFont(size=12), text_color=TEXT_PRIMARY).pack(side="left")
        self.deb_menu = ctk.CTkOptionMenu(
            deb_row,
            values=["2 Frames (~66ms, Fast)", "3 Frames (~100ms, Standard)", "4 Frames (~133ms, High Stability)"],
            fg_color="#232731", button_color=ACCENT_BLUE, button_hover_color=BUTTON_HOVER,
            command=self._on_deb_change
        )
        self.deb_menu.pack(side="right")

        # 5. SAFETY & RE-ARM BEHAVIOR
        self._create_section_label(self.scroll, "SAFETY & RE-ARM POLICIES")
        saf_frame = ctk.CTkFrame(self.scroll, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER_COLOR)
        saf_frame.pack(fill="x", pady=4, padx=4)

        rearm_row = ctk.CTkFrame(saf_frame, fg_color="transparent")
        rearm_row.pack(fill="x", padx=12, pady=6)
        self.rearm_switch = ctk.CTkSwitch(
            rearm_row,
            text="Require manual click to resume driving after tracking drops",
            progress_color=ACCENT_BLUE,
            command=self._on_rearm_toggle
        )
        self.rearm_switch.pack(side="left")

        focus_row = ctk.CTkFrame(saf_frame, fg_color="transparent")
        focus_row.pack(fill="x", padx=12, pady=6)
        self.focus_switch = ctk.CTkSwitch(
            focus_row,
            text="Release all keys when game window loses focus",
            progress_color=ACCENT_BLUE,
            command=self._on_focus_toggle
        )
        self.focus_switch.pack(side="left")

        # 6. CAMERA HARDWARE SELECTION
        self._create_section_label(self.scroll, "CAMERA DEVICE SELECTION")
        cam_frame = ctk.CTkFrame(self.scroll, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER_COLOR)
        cam_frame.pack(fill="x", pady=4, padx=4)

        cam_row = ctk.CTkFrame(cam_frame, fg_color="transparent")
        cam_row.pack(fill="x", padx=12, pady=8)
        ctk.CTkLabel(cam_row, text="Active Webcam:", font=ctk.CTkFont(size=12), text_color=TEXT_PRIMARY).pack(side="left")
        cam_opts = [f"Camera Device #{idx}" for idx in (self.available_cameras if self.available_cameras else [0])]
        self.cam_dropdown = ctk.CTkOptionMenu(
            cam_row, values=cam_opts, fg_color="#232731", button_color=ACCENT_BLUE, button_hover_color=BUTTON_HOVER,
            command=self._on_cam_selected
        )
        self.cam_dropdown.pack(side="right")

        # 7. KEYBOARD CONTROLS MAPPING
        self._create_section_label(self.scroll, "KEYBOARD BINDINGS")
        key_frame = ctk.CTkFrame(self.scroll, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER_COLOR)
        key_frame.pack(fill="x", pady=4, padx=4)

        preset_row = ctk.CTkFrame(key_frame, fg_color="transparent")
        preset_row.pack(fill="x", padx=12, pady=8)

        ctk.CTkButton(preset_row, text="Arrow Keys (Default)", fg_color="#252A34", hover_color="#323846",
                      command=self._apply_arrow_preset).pack(side="left", padx=4, expand=True, fill="x")
        ctk.CTkButton(preset_row, text="WASD Controls", fg_color="#252A34", hover_color="#323846",
                      command=self._apply_wasd_preset).pack(side="left", padx=4, expand=True, fill="x")

        self.key_entries = {}
        for action, default_key in [("steer_left", "Left Arrow"), ("steer_right", "Right Arrow"),
                                    ("accelerate", "Up Arrow"), ("brake", "Down Arrow"), ("handbrake", "Spacebar")]:
            krow = ctk.CTkFrame(key_frame, fg_color="transparent")
            krow.pack(fill="x", padx=14, pady=3)
            ctk.CTkLabel(krow, text=f"{action.replace('_', ' ').title()}:", font=ctk.CTkFont(size=12), text_color=TEXT_SECONDARY).pack(side="left")
            val_lbl = ctk.CTkLabel(krow, text=f"[{default_key}]", font=ctk.CTkFont(family="Consolas", size=12, weight="bold"), text_color=ACCENT_CYAN)
            val_lbl.pack(side="right")
            self.key_entries[action] = val_lbl

        # Bottom Action Bar
        bottom_bar = ctk.CTkFrame(self, fg_color=PANEL_BG, corner_radius=0, height=60)
        bottom_bar.pack(fill="x", side="bottom")

        reset_btn = ctk.CTkButton(
            bottom_bar, text="↺ RESET TO DEFAULTS",
            fg_color="#323846", hover_color="#454D5E", text_color=TEXT_PRIMARY,
            command=self._on_reset_defaults
        )
        reset_btn.pack(side="left", padx=16, pady=12)

        close_btn = ctk.CTkButton(
            bottom_bar, text="SAVE & APPLY SETTINGS",
            fg_color=ACCENT_BLUE, hover_color=BUTTON_HOVER, text_color="#000000",
            font=ctk.CTkFont(weight="bold", size=13),
            command=self._close_dialog
        )
        close_btn.pack(side="right", padx=16, pady=12)

    def _create_section_label(self, parent, text: str):
        lbl = ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"), text_color=ACCENT_CYAN)
        lbl.pack(anchor="w", padx=6, pady=(10, 2))

    def _load_current_values(self):
        curr_str = f"Camera Device #{self.current_cam_index}"
        if curr_str in self.cam_dropdown._values:
            self.cam_dropdown.set(curr_str)

        # Notebook settings
        if self.calibration.notebook_steering_enabled:
            self.nb_steering_switch.select()
        else:
            self.nb_steering_switch.deselect()

        nb_dz = self.calibration.notebook_deadzone_angle
        self.nb_dz_slider.set(nb_dz)
        self.nb_dz_lbl.configure(text=f"Notebook Steering Deadzone: {nb_dz:.1f}°")

        nb_max = self.calibration.notebook_max_steer_angle
        self.nb_max_slider.set(nb_max)
        self.nb_max_lbl.configure(text=f"Max Steering Angle (100% Turn): {nb_max:.1f}°")

        color_name = self.calibration.config.get("notebook_color_name", "Orange (Default)")
        if color_name in self.nb_color_menu._values:
            self.nb_color_menu.set(color_name)

        # Two hand mode
        if self.calibration.two_hand_mode_enabled:
            self.two_hand_switch.select()
        else:
            self.two_hand_switch.deselect()

        mode = self.calibration.steering_mode
        self.mode_menu.set("Steering Wheel Tilt (Angle)" if mode == "tilt" else "Horizontal Position (Left/Right)")

        sens = self.calibration.steering_sensitivity
        self.sens_slider.set(sens)
        self.sens_lbl.configure(text=f"Steering Sensitivity: {sens:.1f}x")

        dz = self.calibration.deadzone_x
        self.dz_slider.set(dz)
        self.dz_lbl.configure(text=f"Center Deadzone: {dz*100:.0f}%")

        # Switches
        if self.keyboard.check_focus:
            self.focus_switch.select()
        else:
            self.focus_switch.deselect()

        if self.keyboard.require_manual_resume:
            self.rearm_switch.select()
        else:
            self.rearm_switch.deselect()

        # Highlight active preset button if matched
        self._highlight_preset_button(self.calibration.config.get("active_preset", "normal"))
        self._refresh_key_labels(self.keyboard.get_mapping())

    def _highlight_preset_button(self, preset_key: str):
        self.btn_beginner.configure(fg_color="#0077BE" if preset_key == "beginner" else "#232731")
        self.btn_normal.configure(fg_color="#0077BE" if preset_key == "normal" else "#232731")
        self.btn_pro.configure(fg_color="#0077BE" if preset_key == "pro" else "#232731")

    def _apply_preset_choice(self, preset_key: str):
        self.calibration.apply_preset(preset_key)
        self._load_current_values()
        if self.on_settings_applied_cb:
            self.on_settings_applied_cb()

    def _on_nb_switch_toggle(self):
        enabled = bool(self.nb_steering_switch.get())
        self.calibration.set_notebook_steering_enabled(enabled)

    def _on_nb_dz_change(self, val: float):
        self.calibration.set_notebook_deadzone_angle(val)
        self.nb_dz_lbl.configure(text=f"Notebook Steering Deadzone: {val:.1f}°")

    def _on_nb_max_change(self, val: float):
        self.calibration.set_notebook_max_steer_angle(val)
        self.nb_max_lbl.configure(text=f"Max Steering Angle (100% Turn): {val:.1f}°")

    def _on_nb_color_change(self, val: str):
        self.calibration.set_notebook_color_profile(val)

    def _on_two_hand_toggle(self):
        enabled = bool(self.two_hand_switch.get())
        self.calibration.set_two_hand_mode(enabled)
        self.hand_tracker.set_two_handed_mode(enabled)

    def _on_mode_change(self, val: str):
        new_mode = "tilt" if "Tilt" in val else "position"
        self.calibration.set_steering_mode(new_mode)

    def _on_sens_change(self, val: float):
        self.calibration.set_steering_sensitivity(val)
        self.sens_lbl.configure(text=f"Steering Sensitivity: {val:.1f}x")

    def _on_dz_change(self, val: float):
        self.calibration.set_deadzone_x(val)
        self.dz_lbl.configure(text=f"Center Deadzone: {val*100:.0f}%")

    def _on_acc_change(self, val: str):
        if "Up" in val:
            self.calibration.config["accel_mode"] = "forward_y"
        elif "Point" in val:
            self.calibration.config["accel_mode"] = "point"
        else:
            self.calibration.config["accel_mode"] = "forward_push"
        self.calibration.save()

    def _on_brk_change(self, val: str):
        self.calibration.config["brake_gesture_type"] = "thumbs_down" if "Thumbs" in val else "fist"
        self.calibration.save()

    def _on_hb_change(self, val: str):
        if "Peace" in val:
            self.calibration.config["handbrake_type"] = "peace"
        elif "Shaka" in val:
            self.calibration.config["handbrake_type"] = "shaka"
        elif "Pinch" in val:
            self.calibration.config["handbrake_type"] = "pinch"
        else:
            self.calibration.config["handbrake_type"] = "pinky"
        self.calibration.save()

    def _on_deb_change(self, val: str):
        if "2" in val:
            self.calibration.config["gesture_confirm_frames"] = 2
        elif "3" in val:
            self.calibration.config["gesture_confirm_frames"] = 3
        else:
            self.calibration.config["gesture_confirm_frames"] = 4
        self.calibration.save()

    def _on_rearm_toggle(self):
        self.keyboard.require_manual_resume = bool(self.rearm_switch.get())
        self.calibration.config["require_manual_resume"] = self.keyboard.require_manual_resume
        self.calibration.save()

    def _on_focus_toggle(self):
        self.keyboard.check_focus = bool(self.focus_switch.get())

    def _on_cam_selected(self, val: str):
        try:
            idx = int(val.split("#")[-1])
            if self.on_cam_change_cb:
                self.on_cam_change_cb(idx)
        except Exception:
            pass

    def _apply_arrow_preset(self):
        mapping = {"steer_left": "left", "steer_right": "right", "accelerate": "up", "brake": "down", "handbrake": "space"}
        self.keyboard.set_mapping(mapping)
        self._refresh_key_labels(mapping)

    def _apply_wasd_preset(self):
        mapping = {"steer_left": "a", "steer_right": "d", "accelerate": "w", "brake": "s", "handbrake": "space"}
        self.keyboard.set_mapping(mapping)
        self._refresh_key_labels(mapping)

    def _refresh_key_labels(self, mapping: Dict[str, str]):
        name_map = {"left": "Left Arrow", "right": "Right Arrow", "up": "Up Arrow", "down": "Down Arrow", "space": "Spacebar",
                    "a": "A Key", "d": "D Key", "w": "W Key", "s": "S Key"}
        for action, key in mapping.items():
            if action in self.key_entries:
                display = name_map.get(key.lower(), key.upper())
                self.key_entries[action].configure(text=f"[{display}]")

    def _on_reset_defaults(self):
        self.calibration.reset_to_defaults()
        self.keyboard.set_mapping({"steer_left": "left", "steer_right": "right", "accelerate": "up", "brake": "down", "handbrake": "space"})
        self.hand_tracker.set_two_handed_mode(False)
        self._load_current_values()
        if self.on_settings_applied_cb:
            self.on_settings_applied_cb()

    def _close_dialog(self):
        is_valid, msg = self.calibration.validate_config(self.calibration.config)
        if not is_valid:
            logger.warning(f"Settings validation failed: {msg}")
        self.calibration.save()
        if self.on_settings_applied_cb:
            self.on_settings_applied_cb()
        self.destroy()
