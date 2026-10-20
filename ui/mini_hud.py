"""
Mini Floating HUD Window for Burnout Camera Drive.
A lightweight, always-on-top telemetry widget designed to hover in the corner
of the screen while playing Burnout Drift Hilltop in the browser.

Displays:
- Live steering deflection gauge
- Throttle (Gas ☝️), Brake (Forward Tilt 📐), and Drift (Space ✌️) badges
- Game window focus indicator
- Quick Stop, Recenter, and Expand controls
"""

import tkinter as tk
import customtkinter as ctk
from typing import Callable, Optional

BG_DARK = "#101318"
BG_CARD = "#181C24"
ACCENT_BLUE = "#00D2FF"
ACCENT_GREEN = "#00FF88"
ACCENT_RED = "#FF3366"
ACCENT_PURPLE = "#B5179E"
TEXT_MUTED = "#8A94A6"
TEXT_WHITE = "#FFFFFF"


class MiniHUDWindow(ctk.CTkToplevel):
    """
    Compact floating dashboard window (stays on top of browser).
    """

    def __init__(self, parent,
                 on_recenter_cb: Callable[[], None],
                 on_stop_cb: Callable[[], None],
                 on_expand_cb: Callable[[], None]):
        super().__init__(parent)

        self.on_recenter_cb = on_recenter_cb
        self.on_stop_cb = on_stop_cb
        self.on_expand_cb = on_expand_cb

        self.title("Burnout Mini HUD")
        self.geometry("310x175")
        self.minsize(280, 160)
        self.configure(fg_color=BG_DARK)

        # Keep on top of browser game
        self.attributes("-topmost", True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()

    def _build_ui(self):
        # Header bar
        header = ctk.CTkFrame(self, fg_color=BG_CARD, height=28, corner_radius=0)
        header.pack(fill="x", side="top")

        ctk.CTkLabel(
            header,
            text="🏎️ BURNOUT HUD",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=ACCENT_BLUE
        ).pack(side="left", padx=8, pady=3)

        self.focus_badge = ctk.CTkLabel(
            header,
            text="CHECKING FOCUS...",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color=ACCENT_GREEN
        )
        self.focus_badge.pack(side="right", padx=8, pady=3)

        # Steering Gauge Frame
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=8, pady=4)

        steer_row = ctk.CTkFrame(body, fg_color="transparent")
        steer_row.pack(fill="x", pady=(2, 1))

        ctk.CTkLabel(steer_row, text="◀ L", font=ctk.CTkFont(size=10, weight="bold"), text_color=ACCENT_BLUE).pack(side="left")
        self.steer_lbl = ctk.CTkLabel(steer_row, text="CENTER", font=ctk.CTkFont(family="Consolas", size=11, weight="bold"), text_color=TEXT_WHITE)
        self.steer_lbl.pack(side="top")
        ctk.CTkLabel(steer_row, text="R ▶", font=ctk.CTkFont(size=10, weight="bold"), text_color=ACCENT_BLUE).pack(side="right")

        self.steer_prog = ctk.CTkProgressBar(body, height=8, progress_color=ACCENT_BLUE, fg_color="#1F232D")
        self.steer_prog.pack(fill="x", pady=(1, 4))
        self.steer_prog.set(0.5)

        # Pedals / Actions Row
        badges_row = ctk.CTkFrame(body, fg_color="transparent")
        badges_row.pack(fill="x", pady=2)

        self.gas_badge = ctk.CTkLabel(
            badges_row, text="GAS ▲", font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#181C24", text_color="#555E6D", corner_radius=4, height=22, padx=6
        )
        self.gas_badge.pack(side="left", padx=2, expand=True, fill="x")

        self.brake_badge = ctk.CTkLabel(
            badges_row, text="BRAKE ▼", font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#181C24", text_color="#555E6D", corner_radius=4, height=22, padx=6
        )
        self.brake_badge.pack(side="left", padx=2, expand=True, fill="x")

        self.drift_badge = ctk.CTkLabel(
            badges_row, text="DRIFT ⎵", font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#181C24", text_color="#555E6D", corner_radius=4, height=22, padx=6
        )
        self.drift_badge.pack(side="left", padx=2, expand=True, fill="x")

        # Action Buttons
        btn_row = ctk.CTkFrame(body, fg_color="transparent")
        btn_row.pack(fill="x", pady=(4, 2))

        self.recenter_btn = ctk.CTkButton(
            btn_row, text="🎯 Center", font=ctk.CTkFont(size=10, weight="bold"),
            width=70, height=24, fg_color="#1F4E5B", hover_color="#2A6B7C",
            command=self.on_recenter_cb
        )
        self.recenter_btn.pack(side="left", padx=2, expand=True, fill="x")

        self.stop_btn = ctk.CTkButton(
            btn_row, text="🛑 Stop", font=ctk.CTkFont(size=10, weight="bold"),
            width=65, height=24, fg_color=ACCENT_RED, hover_color="#E02555",
            command=self.on_stop_cb
        )
        self.stop_btn.pack(side="left", padx=2, expand=True, fill="x")

        self.expand_btn = ctk.CTkButton(
            btn_row, text="🗖 Expand", font=ctk.CTkFont(size=10, weight="bold"),
            width=65, height=24, fg_color="#252A34", hover_color="#333A48",
            command=self._on_expand
        )
        self.expand_btn.pack(side="left", padx=2, expand=True, fill="x")

    def update_telemetry(self, state, is_focused: bool, is_driving: bool):
        """Update live status badges and steering gauge."""
        # Steering gauge
        prog_val = (state.steering_val + 1.0) / 2.0
        self.steer_prog.set(max(0.0, min(1.0, prog_val)))

        if state.steer_left:
            pct = int(abs(state.steering_val) * 100)
            self.steer_lbl.configure(text=f"LEFT ({pct}%)", text_color=ACCENT_BLUE)
        elif state.steer_right:
            pct = int(abs(state.steering_val) * 100)
            self.steer_lbl.configure(text=f"RIGHT ({pct}%)", text_color=ACCENT_BLUE)
        else:
            self.steer_lbl.configure(text="CENTER", text_color=TEXT_MUTED)

        # Pedals
        if state.accelerate and is_driving:
            self.gas_badge.configure(fg_color="#00A86B", text_color=TEXT_WHITE)
        else:
            self.gas_badge.configure(fg_color="#181C24", text_color="#555E6D")

        if state.brake and is_driving:
            self.brake_badge.configure(fg_color=ACCENT_RED, text_color=TEXT_WHITE)
        else:
            self.brake_badge.configure(fg_color="#181C24", text_color="#555E6D")

        if state.handbrake and is_driving:
            self.drift_badge.configure(fg_color=ACCENT_PURPLE, text_color=TEXT_WHITE)
        else:
            self.drift_badge.configure(fg_color="#181C24", text_color="#555E6D")

        # Focus badge
        if not is_driving:
            self.focus_badge.configure(text="STANDBY", text_color=TEXT_MUTED)
        elif is_focused:
            self.focus_badge.configure(text="🟢 FOCUSED", text_color=ACCENT_GREEN)
        else:
            self.focus_badge.configure(text="⚠️ CLICK GAME", text_color=ACCENT_RED)

    def _on_expand(self):
        self.on_expand_cb()
        self.destroy()

    def _on_close(self):
        self._on_expand()
