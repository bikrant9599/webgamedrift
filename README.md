# 🏎️ BURNOUT CAMERA DRIVE — Smart Virtual Wheel & Hilltop Edition

[![Windows Compatible](https://img.shields.io/badge/Platform-Windows%2010%20%7C%2011-0078D6.svg?logo=windows)](https://microsoft.com)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python)](https://www.python.org/)
[![MediaPipe Hands](https://img.shields.io/badge/AI-MediaPipe%20Dual%20Hand-FF6F00.svg)](https://developers.google.com/mediapipe)
[![DirectInput](https://img.shields.io/badge/Input-DirectInput%20Hardware%20Scancodes-00D2FF.svg)](#keyboard-input-safety)
[![Local Privacy](https://img.shields.io/badge/Privacy-100%25%20Local-00FF88.svg)](#session-statistics--privacy)

A high-performance Windows desktop application that allows you to control the popular browser racing game **[Burnout Drift: Hilltop](https://poki.com/en/g/burnout-drift-hilltop)** using **webcam hand movements and two-hand virtual steering** instead of a physical keyboard.

Built with **CustomTkinter** for a sleek dark racing aesthetic, **MediaPipe** for real-time 21-point dual-hand landmark tracking, and **Windows SendInput DirectInput hardware scancodes** for 100% reliable hardware response in WebGL and DirectX browser games.

---

## 🎮 Driving Gestures & Controls

| Driving Action | Physical Hand Movement | Emulated Key | Logic & Protections |
| :--- | :--- | :--- | :--- |
| **Steer Left** | Move hand left / rotate wheel counter-clockwise | `Left Arrow` (or `A`) | Proportional steering beyond deadzone with anti-flutter debounce. |
| **Steer Right** | Move hand right / rotate wheel clockwise | `Right Arrow` (or `D`) | Proportional steering beyond deadzone with anti-flutter debounce. |
| **Accelerate (Gas)** | Push hand forward towards camera | `Up Arrow` (or `W`) | Deliberate forward thrust with hysteresis to prevent accidental gas. |
| **Brake (Stop)** | Form a tight **Fist** ✊ | `Down Arrow` (or `S`) | **Strictly mutually exclusive**: Instantly cuts throttle and applies brakes. |
| **Handbrake (Drift)** | Flash **Peace / Victory** sign ✌️ | `Spacebar` | Immediate drift engagement (also supports Shaka 🤙 or Pinch 🤏). |
| **Coast / Neutral** | Open, resting hand at neutral depth ✋ | *All Keys Released* | Returns car to coasting idle. |
| **Emergency Stop** | Press <kbd>Escape</kbd> or click Red Button | *All Keys Released* | Disables inputs and locks keyboard safely. |

---

## 🌟 Key Features & Improvements

### 1. 🎯 Smart Calibration & Guided Range
- **Quick Neutral (2.5s)**: Sets neutral resting position and hand depth.
- **Guided 3-Step Range Calibration**: Guides you through *Neutral -> Comfortable Left -> Comfortable Right* to set a natural steering boundary tailored to your arm length and camera distance.
- **Accidental Steering Protection**: Steering inputs are muted during calibration.
- **Recalibrate Anytime**: Trigger calibration at any moment with one click.

### 2. 🎚️ Smooth Steering & Sensitivity Presets
- **1-Click Presets**:
  - 🔰 **Beginner**: High deadzone (12%), gentle sensitivity (0.8x), heavy EMA smoothing (0.30), and 3-frame direction debounce.
  - 🏎️ **Normal**: Balanced drift (8% deadzone, 1.0x sensitivity, 0.45 smoothing).
  - ⚡ **Pro**: Ultra-responsive (4% deadzone, 1.5x sensitivity, 0.65 smoothing, instant direction shifts).
- **Zero-Crossing Anti-Flutter**: Direction debounce prevents twitchy, violent left-right oscillations when passing through center.
- **Clean Center Snapping**: Immediately returns steering to 0.0 and releases keys when inside deadzone.

### 3. 🚗 Two-Hand Virtual Steering Wheel
- **Dual-Hand Tracking**: Uses MediaPipe to track both hands simultaneously.
- **Wheel Angle Rotation**: Estimates steering wheel rotation angle from the vector between both hands.
- **Animated HUD Steering Wheel**: Renders a dynamic racing wheel with rim, spokes, and top dead center marker that turns as you tilt your hands.
- **Single-Hand Fallback**: If one hand drops out of the frame, the app seamlessly falls back to single-hand steering without crashing or interrupting inputs.
- **Optional Toggle**: Easily turn Two-Hand Mode ON/OFF from the dashboard or Settings.

### 4. ⚙️ Gesture Customization
- **Configurable Triggers**:
  - Throttle: *Forward Push (Depth)*, *Forward Up (Y Lift)*, or *Index Point*.
  - Brakes: *Full Fist ✊* or *Thumbs Down 👎*.
  - Handbrake: *Peace Sign ✌️*, *Shaka 🤙*, *Index-Thumb Pinch 🤏*, or *Pinky Extended*.
- **Confirmation Debouncing**: Configure 2 to 4 confirmation frames so accidental hand twitches don't trigger inputs.

### 5. 🛡️ Enhanced Safety & Manual Re-Arm
- **Zero Stuck Keys Guarantee**: All keys are released when stopping, closing the app, losing tracking, or losing window focus.
- **Anti-Auto-Resume Protection**: If tracking is lost or camera disconnects while driving, the system enters a safe paused state and displays an alert banner. **Throttle will NEVER automatically resume** until you click `RESUME DRIVING`.
- **Window Focus Safety**: Automatically pauses keyboard simulation when switching away from the browser.

### 6. 🧪 Practice Mode
- **Zero-Risk Testing**: Toggle `🎯 PRACTICE MODE: ON` to practice steering and gestures.
- Telemetry, steering gauges, and active key badges light up in real time, but **zero simulated keystrokes are sent to Windows**.

### 7. 📊 Session Statistics & Privacy
- **Live Metrics**: Shows active session duration, net driving time, and tracking interruptions count.
- **Local History**: Records summary metrics for your last 25 sessions in `session_history.json`.
- **100% Private**: **No webcam images or personal data are ever uploaded, saved, or transmitted**.

### 8. ↺ Settings Validation & Reset
- Validates all sensitivity and deadzone values before saving.
- `↺ RESET TO DEFAULTS` button resets all tuning and bindings to factory defaults.

---

## 🚀 Installation & Launch Guide

### 1. Prerequisites
- **Operating System**: Windows 10 or Windows 11 (64-bit)
- **Python**: Python 3.10, 3.11, or 3.12 ([python.org](https://www.python.org/downloads/))
- **Webcam**: Standard built-in laptop camera or USB webcam

### 2. Clone the Repository
```powershell
git clone https://github.com/bikrant9599/webgamedrift.git
cd webgamedrift
```

### 3. Create & Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the Application
```powershell
python main.py
```

---

## 🕹️ Quick Play Guide

1. **Launch App**: Run `python main.py`.
2. **Open Game**: Click **`🌐 OPEN GAME`** in the top bar to launch [Burnout Drift Hilltop](https://poki.com/en/g/burnout-drift-hilltop).
3. **Calibrate**: Hold your hand in your resting position and click **`🎯 Quick Neutral (2.5s)`** (or try Guided Range).
4. **Focus Browser Canvas**: **Click once inside the game screen** in Chrome/Edge to give it keyboard focus.
5. **Start Driving**: Click **`▶ START DRIVING`** (or toggle **Practice Mode** first to warm up).
6. **Drive**:
   - Push hand forward to hit the gas.
   - Steer by moving left/right or tilting the virtual steering wheel.
   - Flash a **Peace Sign** ✌️ to pop the handbrake into high-speed drifts!
   - Form a **Fist** ✊ to brake before tight turns.
   - Press <kbd>Escape</kbd> for Emergency Stop.

---

## 🧪 Automated Test Suite

Run the full suite of 13 unit tests:
```powershell
python tests/test_controllers.py
```
```
Ran 13 tests in 0.109s
OK
```

---

## 📁 Project Architecture

```
burnout_camera_drive/
├── main.py                     # High-DPI entry point and exception-safe loop
├── requirements.txt            # Python dependencies
├── calibration_config.json     # User calibration, presets, and steering thresholds
├── session_history.json        # Local session logs (no images/personal data)
├── README.md                   # Full documentation & gameplay guide
├── controller/
│   ├── __init__.py
│   ├── camera.py               # Background capture thread, virtual wheel HUD & telemetry
│   ├── hand_tracking.py        # Dual-hand MediaPipe landmark detector with EMA smoothing
│   ├── gesture_detection.py    # Gesture classifier, hysteresis, and anti-flutter debouncing
│   ├── keyboard_control.py     # DirectInput SendInput engine with safety tracking & practice mode
│   ├── calibration.py          # Smart guided range calibration, presets & validation
│   └── session_stats.py        # Local session statistics and interruption logger
├── ui/
│   ├── __init__.py
│   ├── main_window.py          # CustomTkinter dashboard, virtual wheel HUD, and telemetry
│   └── settings.py             # Advanced settings, presets, gesture triggers & validation
├── assets/
│   ├── app_icon.png            # Application graphic
│   ├── app_icon.ico            # Windows executable icon
│   └── hand_landmarker.task    # Bundled MediaPipe hand model
└── tests/
    └── test_controllers.py     # 13 automated unit & safety tests
```

---

## ⚖️ License & Privacy
- **Local Privacy**: All processing runs locally on CPU using MediaPipe and OpenCV. No video, images, or telemetry are ever sent to external servers.
- Free for personal gaming and educational use.
