"""
Unit and integration tests for Burnout Camera Drive.
Tests keyboard controller safety, gesture detector state machine, and calibration logic.
"""

import sys
import os
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from controller.keyboard_control import KeyboardController
from controller.hand_tracking import HandData
from controller.calibration import CalibrationManager
from controller.gesture_detection import GestureDetector, DrivingState


class TestKeyboardSafety(unittest.TestCase):
    def setUp(self):
        self.kb = KeyboardController()
        # Ensure disabled during testing so no real keys are held
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

    def test_active_keys_tracking_when_disabled(self):
        # When controller is disabled, applying actions should result in 0 active keys
        self.kb.apply_actions({"accelerate": True, "steer_left": True})
        self.assertEqual(len(self.kb.get_active_keys()), 0)

    def test_emergency_stop_clears_keys(self):
        self.kb.set_enabled(True)
        # Directly simulate active key
        self.kb._active_keys.add("up")
        self.kb._active_keys.add("left")
        self.kb.emergency_stop()
        self.assertEqual(len(self.kb.get_active_keys()), 0)
        self.assertFalse(self.kb.is_enabled())

    def test_mutual_exclusivity(self):
        self.kb.set_enabled(True)
        # Mock raw press and release to avoid system key injection during headless tests
        press_log = []
        release_log = []
        self.kb._raw_press = lambda k: press_log.append(k)
        self.kb._raw_release = lambda k: release_log.append(k)

        # Both accelerate and brake requested simultaneously
        self.kb.apply_actions({"accelerate": True, "brake": True})
        active = self.kb.get_active_keys()

        # Brake must take priority over accelerate!
        self.assertIn("down", active)
        self.assertNotIn("up", active)


class TestGestureDetection(unittest.TestCase):
    def setUp(self):
        self.calib = CalibrationManager()
        self.calib.config["neutral_x"] = 0.50
        self.calib.config["deadzone_x"] = 0.08
        self.calib.config["steering_sensitivity"] = 1.0
        self.detector = GestureDetector(self.calib)

    def test_no_hand_releases_all(self):
        hand = HandData(is_detected=False)
        state = self.detector.evaluate(hand)
        self.assertFalse(state.accelerate)
        self.assertFalse(state.brake)
        self.assertFalse(state.steer_left)
        self.assertFalse(state.steer_right)
        self.assertFalse(state.handbrake)
        self.assertEqual(state.gesture_name, "NO HAND")

    def test_fist_triggers_brake(self):
        # All fingers curled = fist
        hand = HandData(
            is_detected=True,
            fingers_extended=[False, False, False, False, False],
            hand_center=(0.50, 0.55),
            hand_size=0.20
        )
        # Run 2 frames to clear debouncer
        self.detector.evaluate(hand)
        state = self.detector.evaluate(hand)

        self.assertTrue(state.brake)
        self.assertFalse(state.accelerate)
        self.assertIn("BRAKE", state.gesture_name)

    def test_peace_sign_triggers_handbrake(self):
        # Index & Middle extended, Ring & Pinky curled
        hand = HandData(
            is_detected=True,
            fingers_extended=[False, True, True, False, False],
            hand_center=(0.50, 0.55),
            hand_size=0.20
        )
        self.detector.evaluate(hand)
        state = self.detector.evaluate(hand)

        self.assertTrue(state.handbrake)
        self.assertFalse(state.accelerate)
        self.assertIn("HANDBRAKE", state.gesture_name)

    def test_forward_push_triggers_acceleration(self):
        # Open hand pushed closer: hand size 0.26 vs neutral 0.20 (+30%)
        hand = HandData(
            is_detected=True,
            fingers_extended=[True, True, True, True, True],
            hand_center=(0.50, 0.55),
            hand_size=0.26
        )
        state = self.detector.evaluate(hand)
        self.assertTrue(state.accelerate)
        self.assertFalse(state.brake)
        self.assertIn("ACCELERATING", state.gesture_name)

    def test_steering_deadzone_and_direction(self):
        # 1. Inside deadzone (0.50 neutral, position 0.52 -> dx = +0.02 < deadzone 0.08)
        hand_neutral = HandData(
            is_detected=True,
            fingers_extended=[True, True, True, True, True],
            hand_center=(0.52, 0.55),
            hand_size=0.20
        )
        s_neutral = self.detector.evaluate(hand_neutral)
        self.assertFalse(s_neutral.steer_left)
        self.assertFalse(s_neutral.steer_right)
        self.assertEqual(s_neutral.steering_val, 0.0)

        # 2. Steer Left (X = 0.35, dx = -0.15 < -deadzone 0.08)
        hand_left = HandData(
            is_detected=True,
            fingers_extended=[True, True, True, True, True],
            hand_center=(0.35, 0.55),
            hand_size=0.20
        )
        s_left = self.detector.evaluate(hand_left)
        self.assertTrue(s_left.steer_left)
        self.assertFalse(s_left.steer_right)
        self.assertLess(s_left.steering_val, 0.0)

        # 3. Steer Right (X = 0.68, dx = +0.18 > +deadzone 0.08)
        hand_right = HandData(
            is_detected=True,
            fingers_extended=[True, True, True, True, True],
            hand_center=(0.68, 0.55),
            hand_size=0.20
        )
        s_right = self.detector.evaluate(hand_right)
        self.assertFalse(s_right.steer_left)
        self.assertTrue(s_right.steer_right)
        self.assertGreater(s_right.steering_val, 0.0)


class TestCalibration(unittest.TestCase):
    def test_calibration_routine(self):
        calib = CalibrationManager(filepath="test_calib_temp.json")
        calib.phase_duration = 0.01  # fast test
        calib.start_quick_calibration()

        done, msg, prog = calib.update_calibration_sample((0.48, 0.52), 0.22)
        import time
        time.sleep(0.02)
        done, msg, prog = calib.update_calibration_sample((0.48, 0.52), 0.22)

        self.assertTrue(done)
        self.assertAlmostEqual(calib.neutral_x, 0.48, places=2)
        self.assertTrue(calib.config["is_calibrated"])

        if os.path.exists("test_calib_temp.json"):
            os.remove("test_calib_temp.json")


if __name__ == "__main__":
    unittest.main()
