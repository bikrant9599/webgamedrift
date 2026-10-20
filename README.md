# 🏎️ BURNOUT CAMERA DRIVE — Notebook Wheel & Hilltop Edition

[![Windows Compatible](https://img.shields.io/badge/Platform-Windows%2010%20%7C%2011-0078D6.svg?logo=windows)](https://microsoft.com)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python)](https://www.python.org/)
[![OpenCV Notebook Steering](https://img.shields.io/badge/CV-OpenCV%20Geometric%20Tracking-5C3EE8.svg?logo=opencv)](https://opencv.org/)
[![MediaPipe Hands](https://img.shields.io/badge/AI-MediaPipe%20Dual%20Hand-FF6F00.svg)](https://developers.google.com/mediapipe)
[![DirectInput](https://img.shields.io/badge/Input-DirectInput%20Hardware%20Scancodes-00D2FF.svg)](#keyboard-input-safety)
[![Local Privacy](https://img.shields.io/badge/Privacy-100%25%20Local-00FF88.svg)](#session-statistics--privacy)

A high-performance Windows desktop application that allows you to control the popular browser racing game **[Burnout Drift: Hilltop](https://poki.com/en/g/burnout-drift-hilltop)** using **a physical notebook held with both hands as a steering wheel** and **right-hand finger gestures** via your webcam instead of a physical keyboard.

Built with **CustomTkinter** for a sleek dark racing aesthetic, **OpenCV** for real-time 4-corner geometric notebook tracking, **MediaPipe** for handedness and finger landmark recognition, and **Windows SendInput DirectInput hardware scancodes** for 100% reliable hardware response in WebGL and DirectX browser games.

---

## 🎮 Driving Gestures & Physical Controls

| Driving Action | Physical Input | Emulated Key | Logic & Protections |
| :--- | :--- | :--- | :--- |
| **Steer Left** | Rotate notebook counter-clockwise ↺ | `Left Arrow` (or `A`) | Proportional steering beyond deadzone (default 5.0°) with EMA anti-jitter smoothing. |
| **Steer Right** | Rotate notebook clockwise ↻ | `Right Arrow` (or `D`) | Proportional steering beyond deadzone (default 5.0°) with EMA anti-jitter smoothing. |
| **Steer Center** | Return notebook to horizontal / neutral | *Keys Released* | Automatically snaps back to center deadzone; prevents drift or accidental turns. |
| **Brake (Forward Tilt)** | Tilt notebook forward out-of-plane 📐 | `Down Arrow` (or `S`) | **3D Pose & Foreshortening**: Distinguishes forward pitch from roll steering. Releases instantly when returned to neutral pitch. |
| **Accelerate (Gas)** | Raise **Right-Hand Index Finger** ☝️ | `Up Arrow` (or `W`) | **Handedness-aware**: Only physical right index finger triggers gas (middle/ring/pinky curled). Left hand is ignored. |
| **Steer + Accelerate** | Rotate notebook + Raise Right Index | `Left/Right + Up` | **Simultaneous multi-input**: Steer right/left while accelerating, or coast through corners. |
| **Steer + Brake** | Rotate notebook + Tilt Forward | `Left/Right + Down` | **Simultaneous control**: Valid steering is preserved while braking if movements are distinguished reliably. |
| **Brake Priority** | Tilt Forward + Right Index Raised | `Down Arrow` only | **Strict priority**: Braking always cancels acceleration; zero conflicting gas + brake inputs. |
| **Handbrake (Drift)** | Flash **Peace / Victory** sign ✌️ | `Spacebar` | Immediate drift engagement (also supports Shaka 🤙 or Pinch 🤏). |
| **Emergency Stop** | Press <kbd>Escape</kbd> or click Red Button | *All Keys Released* | Instantly releases all keys and halts steering inputs safely. |

---

## 🌟 Key Features & Improvements

### 1. 📓 Notebook-Based Physical Steering & Forward-Tilt Braking
- **Real-World Object Tracking**: Detects and tracks an orange (or custom colored) notebook held horizontally with both hands like a physical steering wheel.
- **4-Corner & Outline Tracking**: Locates and renders the notebook's rectangular boundaries, 4 labeled corners (TL, TR, BR, BL), center reticle, and orientation vector.
- **Roll Steering vs. Forward Pitch Braking**:
  - Uses `cv2.solvePnP` with square quadrilateral geometry to estimate full 3D Euler angles (Roll for steering, Pitch for braking).
  - Uses vertical foreshortening ($h / h_{neutral}$) and trapezoid ratio to verify physical forward tilt.
  - Distinguishes steering (in-plane roll) from braking (out-of-plane pitch) so steering never triggers accidental braking.
  - Dynamic visual outline: **RED** when braking, **CYAN** when steering, **EMERALD GREEN** when neutral.
- **Configurable Dynamics**: Adjustable deadzone angle (1°–15°), maximum steering angle (15°–50°), braking pitch threshold (4°–25°), and color profiles.

### 2. ☝️ Right-Hand Index Finger Forward Acceleration
- **Physical Handedness Detection**: Accurately distinguishes physical Right vs Left hand on mirrored selfie webcam feeds.
- **Gesture Distinction**: Checks that the right index finger is raised and extended upward while the remaining fingers (middle, ring, pinky) are curled, distinguishing it cleanly from an open palm.
- **Anti-Twitch Debounce**: Short confirmation delay prevents false activations from momentary hand repositioning.
- **Zero Accidental Resumption**: When tracking drops or finger is lowered, acceleration releases immediately.

### 3. 🎯 Guided 5-Step Notebook Calibration Wizard
Click **`📓 CALIBRATE NOTEBOOK (5-STEP WIZARD)`** to launch the comprehensive calibration:
1. **[Step 1/5] Hold Notebook Neutral**: Locks resting angle, 3D pitch, height, width, and corner bounds.
2. **[Step 2/5] Rotate Left and Right**: Measures comfortable left/right steering range and sets dynamic max steer angle.
3. **[Step 3/5] Perform Braking Movement**: Measures forward tilt pitch and height drop. **Rejects calibration** if camera cannot distinguish braking movement reliably (pitch delta < 5.0° or height drop < 6%).
4. **[Step 4/5] Return to Neutral**: Confirms that returning the notebook to neutral resting position cleanly releases the brake.
5. **[Step 5/5] Raise Right Index Finger**: Verifies deliberate right-hand index finger detection for gas acceleration.

### 4. 🎚️ Smooth Steering & Sensitivity Presets
- **1-Click Presets**:
  - 🔰 **Beginner**: High deadzone (8.0° notebook / 12% hand), gentle sensitivity (0.8x), heavy smoothing (0.30).
  - 🏎️ **Normal**: Balanced drift (5.0° notebook / 8% hand, 1.0x sensitivity, 0.45 smoothing).
  - ⚡ **Pro**: Ultra-responsive (3.0° notebook / 4% hand, 1.5x sensitivity, 0.65 smoothing).
- **Anti-Flutter Zero Crossing**: Direction debounce prevents twitchy, violent left-right oscillations when passing through center.

### 5. 🚗 Two-Hand Virtual Wheel & Single-Hand Modes (Preserved)
- **Two-Hand Mode**: Seamless dual-hand steering estimation without a physical object.
- **Single-Hand Fallback**: If one hand is obscured, system automatically falls back to single-hand steering without crashing or interrupting inputs.

### 6. 🛡️ Enhanced Safety & Manual Re-Arm
- **Zero Stuck Keys Guarantee**: All keys are released when stopping, closing the app, losing tracking, or losing window focus.
- **Anti-Auto-Resume Protection**: If tracking is lost while driving, the system pauses keyboard simulation and displays a safety banner. Throttle will NEVER automatically resume until you click `RESUME DRIVING`.
- **Window Focus Safety**: Automatically pauses keyboard simulation when switching away from the browser.

### 7. 🧪 Practice Mode
- **Zero-Risk Testing**: Toggle `🎯 PRACTICE MODE: ON` to test notebook rotation and finger gestures with live visual HUD feedback without sending any keystrokes to Windows.

### 9. 🚀 One-Click Play Now & Mini HUD Overlay
- **`🎮 1-CLICK PLAY NOW`**: Opens Burnout Drift Hilltop in your default browser, initializes camera and AI tracking, displays live game window focus status, and guides you to click Start Driving when ready.
- **`🗗 Mini HUD Mode`**: Compact always-on-top dashboard window you can keep floating over the game canvas while playing, displaying live steering gauge, gas/brake/drift badges, game focus detection, and quick Recenter/Stop controls.
- **`🎯 Recenter Wheel`**: 1-click neutral steering alignment anytime without repeating the full calibration wizard.

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
2. **Hold Notebook**: Hold your orange notebook horizontally with both hands in front of the camera like a steering wheel.
3. **One-Click Play**: Click **`🎮 1-CLICK PLAY NOW`** in the top bar to automatically launch [Burnout Drift: Hilltop](https://poki.com/en/g/burnout-drift-hilltop) in your browser.
4. **Recenter / Calibrate**: Hold your notebook level and click **`🎯 Recenter Wheel`** (or click **`📓 CALIBRATE NOTEBOOK WHEEL (5-STEP)`** for full calibration).
5. **Focus Browser Canvas**: Click inside the game canvas in Chrome or Edge so the green "GAME FOCUSED" banner appears.
6. **Start Driving**: Click **`▶ START DRIVING`** (or switch to **`🗗 Mini HUD Mode`** to keep a floating HUD over the game).
7. **Drive**:
   - Rotate notebook clockwise to steer right; counter-clockwise to steer left.
   - Raise your **right index finger** ☝️ to accelerate (lower it to stop).
   - Tilt notebook forward 📐 or form a fist ✊ to brake.
   - Flash a **Peace Sign** ✌️ to pop the handbrake into high-speed drifts!
   - Press <kbd>Escape</kbd> anytime for Emergency Stop.

---

## 🧪 Automated Test Suite

Run the full suite of 37 unit, safety, and integration tests:
```powershell
python -m unittest discover -s tests
```
```
Ran 37 tests in 6.162s
OK
```

---

## 📁 Project Architecture

```
webgamedrift/
├── main.py                     # High-DPI entry point and exception-safe loop
├── requirements.txt            # Python dependencies
├── calibration_config.json     # User calibration, presets, and steering thresholds
├── session_history.json        # Local session logs (no images/personal data)
├── README.md                   # Full documentation & gameplay guide
├── controller/
│   ├── __init__.py
│   ├── notebook_tracking.py    # OpenCV notebook detector, 4-corner sorter, rotation & EMA smoothing
│   ├── camera.py               # Background capture thread, notebook tracking HUD & telemetry
│   ├── hand_tracking.py        # Dual-hand MediaPipe landmark detector & anatomical right index gesture logic
│   ├── gesture_detection.py    # Simultaneous notebook steering & right index acceleration evaluator
│   ├── keyboard_control.py     # DirectInput SendInput engine with safety tracking, window focus, & practice mode
│   ├── calibration.py          # 5-step notebook wizard, guided range calibration, instant recentering & presets
│   └── session_stats.py        # Local session statistics and interruption logger
├── ui/
│   ├── __init__.py
│   ├── main_window.py          # CustomTkinter dashboard, 1-Click Play, notebook telemetry & calibration controls
│   ├── mini_hud.py             # Always-on-top floating mini HUD window with live steering gauge & controls
│   └── settings.py             # Advanced settings, notebook deadzone sliders & color profiles
├── assets/
│   ├── app_icon.png            # Application graphic
│   ├── app_icon.ico            # Windows executable icon
│   └── hand_landmarker.task    # Bundled MediaPipe hand model
└── tests/
    └── test_controllers.py     # 37 automated unit, safety, notebook, focus & gesture tests
```

---

## ⚖️ License & Privacy
- **Local Privacy**: All processing runs 100% locally on CPU using MediaPipe and OpenCV. No video, images, or telemetry are ever sent to external servers.
- Free for personal gaming and educational use.
