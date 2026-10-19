"""
Unit and integration tests for Burnout Camera Drive.
Tests keyboard controller safety, gesture detector state machine,
smart calibration, presets, two-hand virtual wheel, practice mode, and session stats.
"""

import sys
import os
import unittest
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from controller.keyboard_control import KeyboardController
from controller.hand_tracking import HandData
from controller.calibration import CalibrationManager, PRESETS
from controller.gesture_detection import GestureDetector, DrivingState
from controller.session_stats import SessionTracker


class TestKeyboardSafetyAndPracticeMode(unittest.TestCase):
    def setUp(self):
        self.kb = KeyboardController()
        self.kb.set_enabled(False)

    def tearDown(self):
        self.kb.release_all()

    def test_default_mappings(self):
        mapping = self.kb.get_mapping()
        self.assertEqual(mapping["steer_left"], "left")
        self.assertEqual(mapping["steer_right"], "right")
        self.assertEqual(mapping["accelerate"], "up")
        self.assertEqual(mapping["brake"], "down")
        self.assertEqual(mapping["handbrake"], "space")

    def test_practice_mode_suppresses_real_keys(self):
        self.kb.set_practice_mode(True)
        self.kb.set_enabled(True)

        sent_keys = []
        self.kb._send_win_key = lambda code, ext, down: sent_keys.append((code, down))

        self.kb.apply_actions({"accelerate": True})
        # Practice mode suppresses OS key sending
        self.assertEqual(len(sent_keys), 0)
        # But active_keys is tracked for UI HUD badges
        self.assertIn("up", self.kb.get_active_keys())

    def test_manual_rearm_after_tracking_interruption(self):
        self.kb.set_enabled(True)
        self.kb.require_manual_resume = True

        # Simulate driving
        self.kb.apply_actions({"accelerate": True})
        self.assertIn("up", self.kb.get_active_keys())

        # Tracking interrupted
        self.kb.notify_tracking_interrupted("Hand lost")
        # All keys must be instantly released
        self.assertEqual(len(self.kb.get_active_keys()), 0)
        self.assertTrue(self.kb.is_interrupted)

        # Applying actions while interrupted must NOT press keys
        self.kb.apply_actions({"accelerate": True})
        self.assertEqual(len(self.kb.get_active_keys()), 0)

        # Manually re-arm
        self.kb.rearm_driving()
        self.assertFalse(self.kb.is_interrupted)
        self.kb.apply_actions({"accelerate": True})
        self.assertIn("up", self.kb.get_active_keys())

    def test_emergency_stop_clears_keys(self):
        self.kb.set_enabled(True)
        self.kb._active_keys.add("up")
        self.kb._active_keys.add("left")
        self.kb.emergency_stop()
        self.assertEqual(len(self.kb.get_active_keys()), 0)
        self.assertFalse(self.kb.is_enabled())

    def test_mutual_exclusivity(self):
        self.kb.set_enabled(True)
        self.kb._raw_press = lambda k: None
        self.kb._raw_release = lambda k: None

        self.kb.apply_actions({"accelerate": True, "brake": True})
        active = self.kb.get_active_keys()
        self.assertIn("down", active)
        self.assertNotIn("up", active)


class TestSmartCalibrationAndPresets(unittest.TestCase):
    def setUp(self):
        self.test_cfg = "test_calib_smart.json"
        self.calib = CalibrationManager(filepath=self.test_cfg)

    def tearDown(self):
        if os.path.exists(self.test_cfg):
            try:
                os.remove(self.test_cfg)
            except Exception:
                pass

    def test_presets_application(self):
        self.assertTrue(self.calib.apply_preset("beginner"))
        self.assertEqual(self.calib.steering_sensitivity, PRESETS["beginner"]["steering_sensitivity"])
        self.assertEqual(self.calib.deadzone_x, PRESETS["beginner"]["deadzone_x"])

        self.assertTrue(self.calib.apply_preset("pro"))
        self.assertEqual(self.calib.steering_sensitivity, PRESETS["pro"]["steering_sensitivity"])
        self.assertEqual(self.calib.deadzone_x, PRESETS["pro"]["deadzone_x"])

    def test_validation_and_reset(self):
        # Valid config
        is_v, _ = self.calib.validate_config({"steering_sensitivity": 1.2, "deadzone_x": 0.08, "neutral_x": 0.5})
        self.assertTrue(is_v)

        # Invalid config
        is_v, _ = self.calib.validate_config({"steering_sensitivity": -1.0})
        self.assertFalse(is_v)

        # Reset to defaults
        self.calib.reset_to_defaults()
        self.assertEqual(self.calib.steering_sensitivity, 1.0)
        self.assertEqual(self.calib.deadzone_x, 0.08)

    def test_guided_calibration_phases(self):
        self.calib.phase_duration = 0.01
        self.calib.start_guided_calibration()

        # Step 1: Neutral
        self.calib.update_calibration_sample((0.50, 0.50), 0.20)
        time.sleep(0.015)
        done, msg, prog = self.calib.update_calibration_sample((0.50, 0.50), 0.20)
        self.assertFalse(done)
        self.assertEqual(self.calib.calibration_phase, "sample_left")

        # Step 2: Left range
        time.sleep(0.015)
        done, msg, prog = self.calib.update_calibration_sample((0.28, 0.50), 0.20)
        self.assertFalse(done)
        self.assertEqual(self.calib.calibration_phase, "sample_right")

        # Step 3: Right range
        time.sleep(0.015)
        done, msg, prog = self.calib.update_calibration_sample((0.72, 0.50), 0.20)
        self.assertTrue(done)
        self.assertEqual(self.calib.calibration_phase, "done")
        self.assertAlmostEqual(self.calib.comfortable_range_x, 0.22, places=2)


class TestTwoHandVirtualWheelAndGestures(unittest.TestCase):
    def setUp(self):
        self.calib = CalibrationManager(filepath="test_temp_wheel.json")
        self.calib.config["neutral_x"] = 0.50
        self.calib.config["deadzone_x"] = 0.08
        self.calib.config["deadzone_angle"] = 12.0
        self.calib.config["gesture_confirm_frames"] = 2
        self.detector = GestureDetector(self.calib)

    def tearDown(self):
        if os.path.exists("test_temp_wheel.json"):
            try:
                os.remove("test_temp_wheel.json")
            except Exception:
                pass

    def test_two_hand_steering_wheel_rotation(self):
        # Two hands: Left hand at (0.35, 0.60), Right hand at (0.65, 0.40) -> Right hand is higher -> turning LEFT
        hand = HandData(
            is_detected=True,
            is_two_handed=True,
            left_hand_center=(0.35, 0.60),
            right_hand_center=(0.65, 0.40),
            wheel_angle=-25.0,  # rotated left > deadzone 12 deg
            fingers_extended=[True, True, True, True, True],
            hand_size=0.20
        )
        state = self.detector.evaluate(hand)
        self.assertTrue(state.steer_left)
        self.assertFalse(state.steer_right)
        self.assertLess(state.steering_val, 0.0)

    def test_two_hand_deadzone_return_to_center(self):
        # Hands level: angle = 3 deg < deadzone 12 deg
        hand = HandData(
            is_detected=True,
            is_two_handed=True,
            left_hand_center=(0.35, 0.50),
            right_hand_center=(0.65, 0.51),
            wheel_angle=3.0,
            fingers_extended=[True, True, True, True, True],
            hand_size=0.20
        )
        state = self.detector.evaluate(hand)
        self.assertFalse(state.steer_left)
        self.assertFalse(state.steer_right)
        self.assertEqual(state.steering_val, 0.0)

    def test_single_hand_fallback_in_wheel_mode(self):
        # Only 1 hand visible in wheel mode -> fallback engages smoothly
        hand = HandData(
            is_detected=True,
            is_two_handed=False,
            is_fallback=True,
            hand_center=(0.30, 0.50), # displaced left
            tilt_angle=-20.0,
            fingers_extended=[True, True, True, True, True],
            hand_size=0.20
        )
        state = self.detector.evaluate(hand)
        self.assertTrue(state.is_fallback)
        self.assertTrue(state.steer_left)

    def test_configurable_handbrake_gestures(self):
        # Shaka gesture: Thumb & Pinky extended, middle three curled
        self.calib.config["handbrake_type"] = "shaka"
        hand = HandData(
            is_detected=True,
            fingers_extended=[True, False, False, False, True],
            hand_center=(0.50, 0.50),
            hand_size=0.20
        )
        # Needs 2 frames for confirmation
        self.detector.evaluate(hand)
        state = self.detector.evaluate(hand)
        self.assertTrue(state.handbrake)


class TestSessionStatistics(unittest.TestCase):
    def setUp(self):
        self.hist_file = "test_sess_hist.json"
        self.tracker = SessionTracker(history_file=self.hist_file)

    def tearDown(self):
        if os.path.exists(self.hist_file):
            try:
                os.remove(self.hist_file)
            except Exception:
                pass

    def test_session_lifecycle_and_interruptions(self):
        self.tracker.start_session(mode="live")
        self.assertTrue(self.tracker.is_session_active)

        # Simulate tracking drop during driving
        self.tracker.record_tracking_status(True, is_driving=True)
        self.tracker.record_tracking_status(False, is_driving=True)
        self.assertEqual(self.tracker.interruption_count, 1)

        summary = self.tracker.end_session()
        self.assertIsNotNone(summary)
        self.assertEqual(summary["interruptions"], 1)

        # Verify history file loaded correctly
        loaded = self.tracker.load_history()
        self.assertEqual(len(loaded), 1)


if __name__ == "__main__":
    unittest.main()
