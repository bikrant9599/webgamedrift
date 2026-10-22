"""
BURNOUT CAMERA DRIVE
Main Application Entry Point.

Controls browser game 'Burnout Drift Hilltop' via webcam hand tracking.
Safely manages Windows keyboard inputs, telemetry HUD, and calibration.
"""

import os
import sys
import logging
import ctypes

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Silence MediaPipe C++ clearcut telemetry & TFLite warnings from polluting stderr
try:
    from controller.telemetry_suppress import silence_clearcut_telemetry
    silence_clearcut_telemetry()
except Exception:
    pass

# Configure logging
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(PROJECT_ROOT, "burnout_camera_drive.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger("burnout.main")


def enable_windows_high_dpi():
    """Enable crisp rendering on Windows high-DPI displays."""
    if sys.platform.startswith("win"):
        try:
            # Per-monitor DPI awareness
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


def main():
    logger.info("=" * 60)
    logger.info("STARTING BURNOUT CAMERA DRIVE - HILLTOP EDITION")
    logger.info("Game URL: https://poki.com/en/g/burnout-drift-hilltop")
    logger.info("=" * 60)

    enable_windows_high_dpi()

    # Safety wrapper: ensure keyboard keys are released on any fatal crash
    try:
        from ui.main_window import MainWindow
        app = MainWindow()
        app.mainloop()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down gracefully...")
    except Exception as e:
        logger.critical(f"Unhandled exception in main loop: {e}", exc_info=True)
    finally:
        # Ultimate fallback key safety
        try:
            from controller.keyboard_control import KeyboardController
            fallback_kb = KeyboardController()
            fallback_kb.release_all()
        except Exception:
            pass
        logger.info("BURNOUT CAMERA DRIVE has safely terminated.")


if __name__ == "__main__":
    main()
