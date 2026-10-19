"""
Settings Dialog for Burnout Camera Drive.
Allows tuning steering deadzone, sensitivity, gesture mappings, keyboard controls,
camera selection, and focus safety.
"""

import tkinter as tk
import customtkinter as ctk
from typing import Callable, Dict, Any, List

# Racing theme color palette
BG_DARK = "#121417"
PANEL_BG = "#1A1D24"
ACCENT_BLUE = "#00D2FF"
ACCENT_CYAN = "#00F0FF"
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#9BA3AF"
BORDER_COLOR = "#2A303C"
BUTTON_HOVER = "#00B4DB"


class SettingsWindow(ctk.CTkToplevel):
    """
    Modular racing-styled settings dialog.
    """

    def __init__(self, parent,
                 calibration_mgr,
                 keyboard_ctrl,
                 available_cameras: List[int],
                 current_cam_index: int,
                 on_cam_change_cb: Callable[[int], None],
                 on_settings_applied_cb: Callable[[], None] = None):
        super().__init__(parent)

        self.calibration = calibration_mgr
        self.keyboard = keyboard_ctrl
        self.available_cameras = available_cameras
        self.current_cam_index = current_cam_index
        self.on_cam_change_cb = on_cam_change_cb
        self.on_settings_applied_cb = on_settings_applied_cb

        self.title("Burnout Camera Drive — Settings")
        self.geometry("560x680")
        self.resizable(False, False)
        self.configure(fg_color=BG_DARK)

        # Modal configuration
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
            text="TUNING & HARDWARE CONFIGURATION",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            text_color=ACCENT_BLUE
        )
        title_lbl.pack(side="left", padx=20, pady=12)

        # Scrollable content frame
        self.scroll = ctk.CTkScrollableFrame(self, fg_color=BG_DARK, corner_radius=0)
        self.scroll.pack(fill="both", expand=True, padx=16, pady=12)

        # 1. CAMERA SELECTION PANEL
        self._create_section_label(self.scroll, "VIDEO & HARDWARE INPUT")
        cam_frame = ctk.CTkFrame(self.scroll, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER_COLOR)
        cam_frame.pack(fill="x", pady=6, padx=4)

        cam_row = ctk.CTkFrame(cam_frame, fg_color="transparent")
        cam_row.pack(fill="x", padx=12, pady=10)

        ctk.CTkLabel(cam_row, text="Active Camera Device:", font=ctk.CTkFont(size=13), text_color=TEXT_PRIMARY).pack(side="left")

        cam_opts = [f"Camera Device #{idx}" for idx in (self.available_cameras if self.available_cameras else [0])]
        self.cam_dropdown = ctk.CTkOptionMenu(
            cam_row,
            values=cam_opts,
            fg_color="#232731",
            button_color=ACCENT_BLUE,
            button_hover_color=BUTTON_HOVER,
            text_color=TEXT_PRIMARY,
            command=self._on_cam_selected
        )
        self.cam_dropdown.pack(side="right")

        # 2. STEERING BEHAVIOR PANEL
        self._create_section_label(self.scroll, "STEERING RESPONSE & DYNAMICS")
        steer_frame = ctk.CTkFrame(self.scroll, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER_COLOR)
        steer_frame.pack(fill="x", pady=6, padx=4)

        # Steering Mode (Position vs Tilt)
        mode_row = ctk.CTkFrame(steer_frame, fg_color="transparent")
        mode_row.pack(fill="x", padx=12, pady=8)
        ctk.CTkLabel(mode_row, text="Steering Control Mode:", font=ctk.CTkFont(size=13), text_color=TEXT_PRIMARY).pack(side="left")
        self.mode_menu = ctk.CTkOptionMenu(
            mode_row,
            values=["Horizontal Position (Left/Right)", "Steering Wheel Tilt (Angle)"],
            fg_color="#232731",
            button_color=ACCENT_BLUE,
            button_hover_color=BUTTON_HOVER,
            command=self._on_mode_change
        )
        self.mode_menu.pack(side="right")

        # Sensitivity Slider
        sens_row = ctk.CTkFrame(steer_frame, fg_color="transparent")
        sens_row.pack(fill="x", padx=12, pady=6)
        self.sens_lbl = ctk.CTkLabel(sens_row, text="Steering Sensitivity: 1.0x", font=ctk.CTkFont(size=13), text_color=TEXT_PRIMARY)
        self.sens_lbl.pack(side="left")
        self.sens_slider = ctk.CTkSlider(
            steer_frame,
            from_=0.5, to=2.5,
            number_of_steps=20,
            progress_color=ACCENT_BLUE,
            button_color=ACCENT_CYAN,
            command=self._on_sens_change
        )
        self.sens_slider.pack(fill="x", padx=14, pady=4)

        # Deadzone Slider
        dz_row = ctk.CTkFrame(steer_frame, fg_color="transparent")
        dz_row.pack(fill="x", padx=12, pady=6)
        self.dz_lbl = ctk.CTkLabel(dz_row, text="Center Deadzone: 8%", font=ctk.CTkFont(size=13), text_color=TEXT_PRIMARY)
        self.dz_lbl.pack(side="left")
        self.dz_slider = ctk.CTkSlider(
            steer_frame,
            from_=0.02, to=0.20,
            number_of_steps=18,
            progress_color=ACCENT_BLUE,
            button_color=ACCENT_CYAN,
            command=self._on_dz_change
        )
        self.dz_slider.pack(fill="x", padx=14, pady=(4, 12))

        # 3. GESTURE TUNING PANEL
        self._create_section_label(self.scroll, "GESTURE TRIGGERS")
        gest_frame = ctk.CTkFrame(self.scroll, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER_COLOR)
        gest_frame.pack(fill="x", pady=6, padx=4)

        # Handbrake gesture option
        hb_row = ctk.CTkFrame(gest_frame, fg_color="transparent")
        hb_row.pack(fill="x", padx=12, pady=8)
        ctk.CTkLabel(hb_row, text="Handbrake (Drift) Gesture:", font=ctk.CTkFont(size=13), text_color=TEXT_PRIMARY).pack(side="left")
        self.hb_menu = ctk.CTkOptionMenu(
            hb_row,
            values=["Peace / Victory Sign (Two Fingers)", "Shaka (Thumb + Pinky)", "Index-Thumb Pinch"],
            fg_color="#232731",
            button_color=ACCENT_BLUE,
            button_hover_color=BUTTON_HOVER,
            command=self._on_hb_change
        )
        self.hb_menu.pack(side="right")

        # Accel sensitivity option
        acc_row = ctk.CTkFrame(gest_frame, fg_color="transparent")
        acc_row.pack(fill="x", padx=12, pady=8)
        ctk.CTkLabel(acc_row, text="Throttle (Forward Push) Threshold:", font=ctk.CTkFont(size=13), text_color=TEXT_PRIMARY).pack(side="left")
        self.acc_menu = ctk.CTkOptionMenu(
            acc_row,
            values=["Normal (+15% Push)", "Responsive (+10% Push)", "High (+20% Push)"],
            fg_color="#232731",
            button_color=ACCENT_BLUE,
            button_hover_color=BUTTON_HOVER,
            command=self._on_acc_change
        )
        self.acc_menu.pack(side="right")

        # 4. KEYBOARD MAPPINGS
        self._create_section_label(self.scroll, "KEYBOARD BINDINGS")
        key_frame = ctk.CTkFrame(self.scroll, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER_COLOR)
        key_frame.pack(fill="x", pady=6, padx=4)

        preset_row = ctk.CTkFrame(key_frame, fg_color="transparent")
        preset_row.pack(fill="x", padx=12, pady=10)

        ctk.CTkButton(
            preset_row,
            text="Preset: Arrow Keys (Default)",
            fg_color="#252A34",
            hover_color="#323846",
            text_color=TEXT_PRIMARY,
            command=self._apply_arrow_preset
        ).pack(side="left", padx=4, expand=True, fill="x")

        ctk.CTkButton(
            preset_row,
            text="Preset: WASD Controls",
            fg_color="#252A34",
            hover_color="#323846",
            text_color=TEXT_PRIMARY,
            command=self._apply_wasd_preset
        ).pack(side="left", padx=4, expand=True, fill="x")

        # Key fields
        self.key_entries = {}
        for action, default_key in [("steer_left", "Left Arrow"),
                                    ("steer_right", "Right Arrow"),
                                    ("accelerate", "Up Arrow"),
                                    ("brake", "Down Arrow"),
                                    ("handbrake", "Spacebar")]:
            krow = ctk.CTkFrame(key_frame, fg_color="transparent")
            krow.pack(fill="x", padx=14, pady=4)
            ctk.CTkLabel(krow, text=f"{action.replace('_', ' ').title()}:", font=ctk.CTkFont(size=12), text_color=TEXT_SECONDARY).pack(side="left")
            val_lbl = ctk.CTkLabel(krow, text=f"[{default_key}]", font=ctk.CTkFont(family="Consolas", size=12, weight="bold"), text_color=ACCENT_CYAN)
            val_lbl.pack(side="right")
            self.key_entries[action] = val_lbl

        # 5. SAFETY & GAME FOCUS
        self._create_section_label(self.scroll, "SAFETY & BACKGROUND PROTECTION")
        saf_frame = ctk.CTkFrame(self.scroll, fg_color=PANEL_BG, corner_radius=8, border_width=1, border_color=BORDER_COLOR)
        saf_frame.pack(fill="x", pady=6, padx=4)

        focus_row = ctk.CTkFrame(saf_frame, fg_color="transparent")
        focus_row.pack(fill="x", padx=12, pady=10)
        self.focus_switch = ctk.CTkSwitch(
            focus_row,
            text="Pause driving when browser/game loses focus",
            progress_color=ACCENT_BLUE,
            command=self._on_focus_toggle
        )
        self.focus_switch.pack(side="left")

        # Bottom Action Bar
        bottom_bar = ctk.CTkFrame(self, fg_color=PANEL_BG, corner_radius=0, height=60)
        bottom_bar.pack(fill="x", side="bottom")

        close_btn = ctk.CTkButton(
            bottom_bar,
            text="SAVE & APPLY SETTINGS",
            fg_color=ACCENT_BLUE,
            hover_color=BUTTON_HOVER,
            text_color="#000000",
            font=ctk.CTkFont(weight="bold", size=13),
            command=self._close_dialog
        )
        close_btn.pack(side="right", padx=16, pady=12)

    def _create_section_label(self, parent, text: str):
        lbl = ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=ACCENT_CYAN
        )
        lbl.pack(anchor="w", padx=6, pady=(12, 2))

    def _load_current_values(self):
        # Camera
        curr_str = f"Camera Device #{self.current_cam_index}"
        if curr_str in self.cam_dropdown._values:
            self.cam_dropdown.set(curr_str)

        # Mode
        mode = self.calibration.steering_mode
        self.mode_menu.set("Steering Wheel Tilt (Angle)" if mode == "tilt" else "Horizontal Position (Left/Right)")

        # Sensitivity & Deadzone
        sens = self.calibration.steering_sensitivity
        self.sens_slider.set(sens)
        self.sens_lbl.configure(text=f"Steering Sensitivity: {sens:.1f}x")

        dz = self.calibration.deadzone_x
        self.dz_slider.set(dz)
        self.dz_lbl.configure(text=f"Center Deadzone: {dz*100:.0f}%")

        # Focus switch
        if self.keyboard.check_focus:
            self.focus_switch.select()
        else:
            self.focus_switch.deselect()

        # Update key labels
        mapping = self.keyboard.get_mapping()
        self._refresh_key_labels(mapping)

    def _refresh_key_labels(self, mapping: Dict[str, str]):
        name_map = {"left": "Left Arrow", "right": "Right Arrow", "up": "Up Arrow", "down": "Down Arrow", "space": "Spacebar",
                    "a": "A Key", "d": "D Key", "w": "W Key", "s": "S Key"}
        for action, key in mapping.items():
            if action in self.key_entries:
                display = name_map.get(key.lower(), key.upper())
                self.key_entries[action].configure(text=f"[{display}]")

    def _on_cam_selected(self, val: str):
        try:
            idx = int(val.split("#")[-1])
            if self.on_cam_change_cb:
                self.on_cam_change_cb(idx)
        except Exception:
            pass

    def _on_mode_change(self, val: str):
        new_mode = "tilt" if "Tilt" in val else "position"
        self.calibration.set_steering_mode(new_mode)

    def _on_sens_change(self, val: float):
        self.calibration.set_steering_sensitivity(val)
        self.sens_lbl.configure(text=f"Steering Sensitivity: {val:.1f}x")

    def _on_dz_change(self, val: float):
        self.calibration.set_deadzone_x(val)
        self.dz_lbl.configure(text=f"Center Deadzone: {val*100:.0f}%")

    def _on_hb_change(self, val: str):
        if "Peace" in val:
            self.calibration.config["handbrake_type"] = "peace"
        elif "Shaka" in val:
            self.calibration.config["handbrake_type"] = "thumb"
        else:
            self.calibration.config["handbrake_type"] = "pinch"
        self.calibration.save()

    def _on_acc_change(self, val: str):
        if "Responsive" in val:
            thresh = 0.10
        elif "High" in val:
            thresh = 0.20
        else:
            thresh = 0.15
        self.calibration.config["accel_forward_threshold"] = thresh
        self.calibration.save()

    def _on_focus_toggle(self):
        self.keyboard.check_focus = bool(self.focus_switch.get())

    def _apply_arrow_preset(self):
        mapping = {
            "steer_left": "left",
            "steer_right": "right",
            "accelerate": "up",
            "brake": "down",
            "handbrake": "space"
        }
        self.keyboard.set_mapping(mapping)
        self._refresh_key_labels(mapping)

    def _apply_wasd_preset(self):
        mapping = {
            "steer_left": "a",
            "steer_right": "d",
            "accelerate": "w",
            "brake": "s",
            "handbrake": "space"
        }
        self.keyboard.set_mapping(mapping)
        self._refresh_key_labels(mapping)

    def _close_dialog(self):
        self.calibration.save()
        if self.on_settings_applied_cb:
            self.on_settings_applied_cb()
        self.destroy()
