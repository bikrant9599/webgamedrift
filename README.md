# 🏎️ BURNOUT CAMERA DRIVE — Hilltop Edition

[![Windows Compatible](https://img.shields.io/badge/Platform-Windows%2010%20%7C%2011-0078D6.svg?logo=windows)](https://microsoft.com)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/AI-MediaPipe%20Hands-FF6F00.svg)](https://developers.google.com/mediapipe)
[![DirectInput](https://img.shields.io/badge/Input-DirectInput%20Hardware%20Scancodes-00D2FF.svg)](#keyboard-input-safety)

A high-performance Windows desktop application that lets you control the popular browser racing game **[Burnout Drift: Hilltop](https://poki.com/en/g/burnout-drift-hilltop)** using **webcam hand gestures** instead of a physical keyboard.

Built with **CustomTkinter** for a dark racing aesthetic, **MediaPipe** for real-time 21-point hand tracking with low-pass EMA smoothing, and **Windows SendInput DirectInput scan codes** for 100% reliable hardware-level response in WebGL and DirectX browser games.

---

## 🎮 Driving Gesture Controls

| Driving Action | Physical Gesture | Key Emulated | Description |
| :--- | :--- | :--- | :--- |
| **Steer Left** | Move hand left of neutral | `Left Arrow` (or `A`) | Proportional steering with center deadzone. |
| **Steer Right** | Move hand right of neutral | `Right Arrow` (or `D`) | Proportional steering with center deadzone. |
| **Accelerate (Gas)** | Push hand forward towards camera | `Up Arrow` (or `W`) | Deliberate forward thrust with hysteresis. |
| **Brake (Stop)** | Form a tight **Fist** ✊ | `Down Arrow` (or `S`) | Instant braking (mutually exclusive with gas). |
| **Handbrake (Drift)** | Flash **Peace / Victory** sign ✌️ | `Spacebar` | Immediate drift turn engagement. |
| **Coast / Neutral** | Relax open palm ✋ | None (All Released) | Neutral floating position; releases keys. |
| **Emergency Stop** | Press `Escape` or Red Button | Releases All | Immediate fail-safe input lock. |

> [!NOTE]
> Acceleration and braking are strictly **mutually exclusive**. Forming a fist immediately cuts throttle and applies brakes. Releasing your hand or looking away immediately releases all keys.

---

## ⚡ Key Features

- **DirectInput Hardware Scancode Engine**: Sends low-level hardware scan codes via `ctypes.windll.user32.SendInput` so that browser games and WebGL canvas games never miss a key.
- **Fail-Safe Input Protection**:
  - Never leaves a key stuck or held.
  - Releases keys when hand tracking is temporarily lost.
  - Releases keys when emergency stop is activated or when the app is minimized/closed.
  - Real-time active key visualizer HUD badges.
- **Interactive Neutral Calibration**: 2.5-second calibration establishes your ergonomic neutral center position and resting depth.
- **Live Telemetry & Steering HUD**: Shows steering angle deflection, FPS, tracking status, throttle, and brake intensity in real time.
- **Customizable Controls**: Switch between Arrow keys and WASD presets, adjust deadzones (2%–20%), tune steering sensitivity (0.5x–2.5x), and pick your preferred steering mode (*Horizontal Offset* vs *Steering Wheel Tilt*).

---

## 🚀 Installation & Launch Guide

### 1. Prerequisites
- **Operating System**: Windows 10 or Windows 11 (64-bit)
- **Python**: Python 3.10, 3.11, or 3.12 installed ([python.org](https://www.python.org/downloads/))
- **Webcam**: Any built-in laptop camera or USB webcam

### 2. Clone the Repository
```powershell
git clone https://github.com/bikrant9599/webgamedrift.git
cd webgamedrift
```

### 3. Create & Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1
```
*(If PowerShell displays an execution policy error, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once).*

### 4. Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Launch the Application
```powershell
python main.py
```

---

## 🕹️ How to Play Burnout Drift Hilltop

1. **Launch the Controller**: Run `python main.py`.
2. **Open the Game**: Click the **🌐 OPEN GAME** button in the top bar (or open `https://poki.com/en/g/burnout-drift-hilltop` in Chrome, Edge, Brave, or Firefox).
3. **Calibrate Neutral Position**:
   - Raise your hand in front of your webcam in a natural resting position.
   - Click **🎯 CALIBRATE NEUTRAL POSITION**.
   - Hold still for 2.5 seconds until the status turns green (*Neutral Centered*).
4. **Focus the Game Canvas**:
   - **Click once on the game screen** in your browser window to give it keyboard focus.
5. **Start Driving**:
   - In Burnout Camera Drive, click **▶ START DRIVING**.
   - Push your open hand forward to hit the gas!
   - Shift hand left or right to carve through mountain corners.
   - Flick a **Peace Sign** ✌️ to pop the handbrake and trigger a high-score drift!
   - Form a **Fist** ✊ to brake before hairpin turns.

---

## 🛡️ Input Safety Guarantees

The controller uses state tracking and Windows API hooks to protect against stuck keys:

1. **Auto-Release on Tracking Loss**: If your hand leaves the camera frame or tracking confidence dips, every driving key is released within 33 milliseconds.
2. **Emergency Killswitch**: Press `Esc` or click **🛑 EMERGENCY STOP** at any time to instantly zero out all active keyboard states.
3. **Graceful App Shutdown**: Safe cleanup hooks in `WM_DELETE_WINDOW` and Python `atexit` guarantee that all inputs are cleared when exiting.
4. **Browser Focus Awareness**: An optional toggle in Settings allows the app to automatically suspend inputs when switching away from the browser.

---

## 📦 Packaging as a Windows Executable (.exe)

You can build a standalone `.exe` using PyInstaller:

```powershell
pip install pyinstaller

pyinstaller --noconfirm --onedir --windowed `
    --name "BurnoutCameraDrive" `
    --icon "assets/app_icon.ico" `
    --add-data "assets;assets" `
    main.py
```

The resulting executable will be created inside the `dist/BurnoutCameraDrive/` folder.

---

## 🔧 Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **"Camera Disconnected" message** | Check if another app (Zoom, Teams, Discord) is locking your webcam. Select a different camera index (0, 1, 2) in **Settings**. |
| **Inputs not reaching the game** | Browser security requires user interaction. Make sure you **click inside the game canvas** in Chrome/Edge before driving so the browser receives keyboard events. |
| **Steering feels too jittery / twitchy** | Open **Settings** or use the slider on the main dashboard to increase **Center Deadzone** (e.g. 10%–12%) and decrease **Sensitivity** (e.g. 0.8x–1.0x). |
| **Throttle sticking** | Run **Calibrate Neutral Position** while sitting at your normal distance from the webcam so the app accurately benchmarks your hand depth. |

---

## 📁 Project Architecture

```
burnout_camera_drive/
├── main.py                     # High-DPI entry point and exception-safe loop
├── requirements.txt            # Python dependencies
├── calibration_config.json     # User calibration and steering thresholds
├── README.md                   # Documentation and gameplay instructions
├── controller/
│   ├── __init__.py
│   ├── camera.py               # Background threaded video pipeline & HUD renderer
│   ├── hand_tracking.py        # MediaPipe landmark detector with EMA smoothing
│   ├── gesture_detection.py    # Gesture classifier, hysteresis, and mutual exclusivity
│   ├── keyboard_control.py     # DirectInput SendInput engine with safety tracking
│   └── calibration.py          # Calibration manager and JSON configuration
├── ui/
│   ├── __init__.py
│   ├── main_window.py          # CustomTkinter racing dashboard & telemetry HUD
│   └── settings.py             # Hardware & steering tuning dialog
├── assets/
│   ├── app_icon.png            # Application graphics
│   ├── app_icon.ico            # Windows executable icon
│   └── hand_landmarker.task    # MediaPipe high-accuracy hand landmark model
└── tests/
    └── test_controllers.py     # 10 automated unit & safety tests
```

---

## ⚖️ License & Credits

- Free to use for personal gaming and education.
- Game created by Burnout Drift team, hosted on [Poki](https://poki.com).
- Hand tracking powered by Google [MediaPipe](https://github.com/google/mediapipe).
