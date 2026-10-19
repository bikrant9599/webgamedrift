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
from controller.notebook_tracking import NotebookTracker, NotebookData
from controller.calibration import CalibrationManager, PRESETS
from controller.gesture_detection import GestureDetector, DrivingState
from controller.session_stats import SessionTracker
import cv2
import numpy as np


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


class TestNotebookTrackingAndGeometry(unittest.TestCase):
    def setUp(self):
        self.tracker = NotebookTracker()

    def test_synthetic_horizontal_notebook(self):
        # Create an image with an orange rectangle in the center
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # BGR for orange: B=40, G=140, R=240
        cv2.rectangle(img, (180, 140), (460, 340), (40, 140, 240), -1)

        nb = self.tracker.process_frame(img)
        self.assertTrue(nb.is_detected)
        self.assertEqual(len(nb.corners), 4)
        # Check center is near (320, 240)
        self.assertAlmostEqual(nb.center[0], 320, delta=20)
        self.assertAlmostEqual(nb.center[1], 240, delta=20)
        # Horizontal rectangle -> rotation angle near 0.0
        self.assertAlmostEqual(nb.rotation_angle, 0.0, delta=5.0)

    def test_clockwise_and_counterclockwise_rotation(self):
        # Clockwise tilted rectangle
        img_cw = np.zeros((480, 640, 3), dtype=np.uint8)
        pts_cw = np.array([[220, 120], [480, 200], [420, 380], [160, 300]], dtype=np.int32)
        cv2.fillPoly(img_cw, [pts_cw], (40, 140, 240))

        tracker_cw = NotebookTracker()
        nb_cw = tracker_cw.process_frame(img_cw)
        self.assertTrue(nb_cw.is_detected)
        self.assertGreater(nb_cw.rotation_angle, 5.0)  # Clockwise -> positive angle

        # Counter-Clockwise tilted rectangle
        img_ccw = np.zeros((480, 640, 3), dtype=np.uint8)
        pts_ccw = np.array([[180, 220], [440, 140], [500, 320], [240, 400]], dtype=np.int32)
        cv2.fillPoly(img_ccw, [pts_ccw], (40, 140, 240))

        tracker_ccw = NotebookTracker()
        nb_ccw = tracker_ccw.process_frame(img_ccw)
        self.assertTrue(nb_ccw.is_detected)
        self.assertLess(nb_ccw.rotation_angle, -5.0)  # CCW -> negative angle

    def test_smoothing_measurement(self):
        # Tracker with smoothing
        tracker = NotebookTracker(smoothing=0.5)
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (200, 150), (440, 330), (40, 140, 240), -1)

        nb1 = tracker.process_frame(img)
        self.assertTrue(nb1.is_detected)
        first_angle = nb1.rotation_angle

        # Slight noise/jitter in frame
        img2 = np.zeros((480, 640, 3), dtype=np.uint8)
        pts_j = np.array([[202, 151], [442, 153], [440, 333], [200, 331]], dtype=np.int32)
        cv2.fillPoly(img2, [pts_j], (40, 140, 240))
        nb2 = tracker.process_frame(img2)
        self.assertTrue(nb2.is_detected)
        self.assertAlmostEqual(nb2.rotation_angle, first_angle, delta=3.0)

    def test_user_reference_image_detection(self):
        ref_path = r"C:\Users\Vikrant\.gemini\antigravity-ide\brain\8abb5107-83f3-49a6-b92d-cfa3075aeee9\.user_uploaded\media_1792431586452.png"
        if os.path.exists(ref_path):
            img = cv2.imread(ref_path)
            self.assertIsNotNone(img)
            tracker = NotebookTracker()
            nb = tracker.process_frame(img)
            self.assertTrue(nb.is_detected)
            self.assertEqual(len(nb.corners), 4)
            self.assertGreater(nb.area, 50000)
            self.assertAlmostEqual(nb.center[0], 400, delta=40)
            self.assertAlmostEqual(nb.center[1], 379, delta=40)


class TestRightHandIndexFingerGesture(unittest.TestCase):
    def setUp(self):
        self.calib = CalibrationManager(filepath="test_temp_right_idx.json")
        self.calib.config["accel_mode"] = "right_index"
        self.calib.config["gesture_confirm_frames"] = 2
        self.detector = GestureDetector(self.calib)

    def tearDown(self):
        if os.path.exists("test_temp_right_idx.json"):
            try:
                os.remove("test_temp_right_idx.json")
            except Exception:
                pass

    def test_right_hand_index_raised_triggers_acceleration(self):
        # Physical right hand with index raised
        hand = HandData(
            is_detected=True,
            handedness="Right",
            right_hand_detected=True,
            right_index_raised=True,
            hand_center=(0.7, 0.5),
            hand_size=0.20
        )
        # Two frames for confirmation debounce
        self.detector.evaluate(hand)
        state = self.detector.evaluate(hand)
        self.assertTrue(state.accelerate)
        self.assertIn("INDEX", state.gesture_name)

    def test_left_hand_index_does_not_trigger_acceleration(self):
        # Physical left hand with index raised: right_index_raised is False
        hand = HandData(
            is_detected=True,
            handedness="Left",
            right_hand_detected=False,
            right_index_raised=False,
            hand_center=(0.3, 0.5),
            hand_size=0.20
        )
        self.detector.evaluate(hand)
        state = self.detector.evaluate(hand)
        self.assertFalse(state.accelerate)

    def test_open_hand_does_not_trigger_acceleration(self):
        # Right hand open (palm, all fingers extended) -> right_index_raised is False
        hand = HandData(
            is_detected=True,
            handedness="Right",
            right_hand_detected=True,
            right_index_raised=False,
            hand_center=(0.7, 0.5),
            hand_size=0.20
        )
        self.detector.evaluate(hand)
        state = self.detector.evaluate(hand)
        self.assertFalse(state.accelerate)

    def test_immediate_release_on_lowering_finger(self):
        # First accelerate
        hand_up = HandData(
            is_detected=True,
            right_hand_detected=True,
            right_index_raised=True,
            hand_center=(0.7, 0.5),
            hand_size=0.20
        )
        self.detector.evaluate(hand_up)
        state1 = self.detector.evaluate(hand_up)
        self.assertTrue(state1.accelerate)

        # Lower finger
        hand_down = HandData(
            is_detected=True,
            right_hand_detected=True,
            right_index_raised=False,
            hand_center=(0.7, 0.5),
            hand_size=0.20
        )
        state2 = self.detector.evaluate(hand_down)
        self.assertFalse(state2.accelerate)


class TestCombinedSteeringAndAcceleration(unittest.TestCase):
    def setUp(self):
        self.calib = CalibrationManager(filepath="test_temp_comb.json")
        self.calib.config["notebook_steering_enabled"] = True
        self.calib.config["notebook_deadzone_angle"] = 5.0
        self.calib.config["notebook_max_steer_angle"] = 30.0
        self.calib.config["gesture_confirm_frames"] = 1
        self.detector = GestureDetector(self.calib)

    def tearDown(self):
        if os.path.exists("test_temp_comb.json"):
            try:
                os.remove("test_temp_comb.json")
            except Exception:
                pass

    def test_steer_left_while_accelerating(self):
        # Notebook tilted CCW (-20 deg) + Right index finger raised
        nb = NotebookData(
            is_detected=True,
            rotation_angle=-20.0,
            center=(320, 240)
        )
        hand = HandData(
            is_detected=True,
            right_hand_detected=True,
            right_index_raised=True,
            hand_center=(0.7, 0.5)
        )
        state = self.detector.evaluate(hand, notebook_data=nb)
        self.assertTrue(state.steer_left)
        self.assertFalse(state.steer_right)
        self.assertTrue(state.accelerate)
        self.assertLess(state.steering_val, 0.0)

    def test_steer_right_while_accelerating(self):
        # Notebook tilted CW (+20 deg) + Right index finger raised
        nb = NotebookData(
            is_detected=True,
            rotation_angle=20.0,
            center=(320, 240)
        )
        hand = HandData(
            is_detected=True,
            right_hand_detected=True,
            right_index_raised=True,
            hand_center=(0.7, 0.5)
        )
        state = self.detector.evaluate(hand, notebook_data=nb)
        self.assertTrue(state.steer_right)
        self.assertFalse(state.steer_left)
        self.assertTrue(state.accelerate)
        self.assertGreater(state.steering_val, 0.0)

    def test_straight_acceleration(self):
        # Notebook centered within deadzone (2 deg < 5 deg) + Right index raised
        nb = NotebookData(
            is_detected=True,
            rotation_angle=2.0,
            center=(320, 240)
        )
        hand = HandData(
            is_detected=True,
            right_hand_detected=True,
            right_index_raised=True,
            hand_center=(0.7, 0.5)
        )
        state = self.detector.evaluate(hand, notebook_data=nb)
        self.assertFalse(state.steer_left)
        self.assertFalse(state.steer_right)
        self.assertEqual(state.steering_val, 0.0)
        self.assertTrue(state.accelerate)

    def test_notebook_forward_tilt_braking(self):
        # Notebook tilted forward into braking zone
        nb = NotebookData(
            is_detected=True,
            rotation_angle=0.0,
            pitch_angle=16.0,
            is_braking=True,
            brake_deflection=0.85,
            center=(320, 240)
        )
        hand = HandData(is_detected=True)
        state = self.detector.evaluate(hand, notebook_data=nb)
        self.assertTrue(state.brake)
        self.assertFalse(state.accelerate)
        self.assertAlmostEqual(state.brake_val, 0.85, places=2)
        self.assertIn("BRAKE", state.gesture_name)

    def test_notebook_braking_overrides_acceleration(self):
        # User raises right index finger for gas BUT also tilts notebook forward to brake
        nb = NotebookData(
            is_detected=True,
            rotation_angle=0.0,
            pitch_angle=18.0,
            is_braking=True,
            brake_deflection=1.0,
            center=(320, 240)
        )
        hand = HandData(
            is_detected=True,
            right_hand_detected=True,
            right_index_raised=True,
            hand_center=(0.7, 0.5)
        )
        state = self.detector.evaluate(hand, notebook_data=nb)
        # Priority rule: Braking strictly overrides acceleration
        self.assertTrue(state.brake)
        self.assertFalse(state.accelerate)
        self.assertEqual(state.throttle_val, 0.0)

    def test_steer_left_while_braking(self):
        # Notebook rotated CCW (-22 deg) AND tilted forward to brake
        nb = NotebookData(
            is_detected=True,
            rotation_angle=-22.0,
            pitch_angle=15.0,
            is_braking=True,
            brake_deflection=0.75,
            center=(320, 240)
        )
        hand = HandData(is_detected=True)
        state = self.detector.evaluate(hand, notebook_data=nb)
        self.assertTrue(state.brake)
        self.assertTrue(state.steer_left)
        self.assertFalse(state.steer_right)
        self.assertFalse(state.accelerate)
        self.assertLess(state.steering_val, 0.0)

    def test_steer_right_while_braking(self):
        # Notebook rotated CW (+22 deg) AND tilted forward to brake
        nb = NotebookData(
            is_detected=True,
            rotation_angle=22.0,
            pitch_angle=15.0,
            is_braking=True,
            brake_deflection=0.75,
            center=(320, 240)
        )
        hand = HandData(is_detected=True)
        state = self.detector.evaluate(hand, notebook_data=nb)
        self.assertTrue(state.brake)
        self.assertTrue(state.steer_right)
        self.assertFalse(state.steer_left)
        self.assertFalse(state.accelerate)
        self.assertGreater(state.steering_val, 0.0)

    def test_brake_release_when_returning_to_neutral(self):
        # First brake
        nb_brake = NotebookData(
            is_detected=True,
            pitch_angle=16.0,
            is_braking=True,
            brake_deflection=0.8,
            center=(320, 240)
        )
        hand = HandData(is_detected=True)
        state1 = self.detector.evaluate(hand, notebook_data=nb_brake)
        self.assertTrue(state1.brake)

        # Return to neutral pitch
        nb_neutral = NotebookData(
            is_detected=True,
            pitch_angle=1.0,
            is_braking=False,
            brake_deflection=0.0,
            center=(320, 240)
        )
        state2 = self.detector.evaluate(hand, notebook_data=nb_neutral)
        self.assertFalse(state2.brake)
        self.assertEqual(state2.brake_val, 0.0)

    def test_tracking_loss_releases_all_inputs(self):
        # Notebook and hand not detected
        nb = NotebookData(is_detected=False)
        hand = HandData(is_detected=False)
        state = self.detector.evaluate(hand, notebook_data=nb)
        self.assertFalse(state.steer_left)
        self.assertFalse(state.steer_right)
        self.assertFalse(state.accelerate)
        self.assertFalse(state.brake)
        self.assertEqual(state.steering_val, 0.0)


class TestNotebookCalibrationWizard(unittest.TestCase):
    def setUp(self):
        self.test_cfg = "test_nb_wizard.json"
        self.calib = CalibrationManager(filepath=self.test_cfg)
        self.calib.phase_duration = 0.01  # fast testing

    def tearDown(self):
        if os.path.exists(self.test_cfg):
            try:
                os.remove(self.test_cfg)
            except Exception:
                pass

    def test_full_5_step_notebook_wizard_workflow(self):
        self.calib.start_notebook_calibration()
        self.assertTrue(self.calib.is_calibrating)
        self.assertEqual(self.calib.calibration_phase, "nb_neutral")

        # Step 1: Hold neutral position (0.5 deg roll, 2.0 deg pitch, height=180, width=280)
        nb_neutral = NotebookData(
            is_detected=True, raw_angle=0.5, raw_pitch=2.0, height=180.0, width=280.0, center=(320, 240)
        )
        hand = HandData(is_detected=True)
        self.calib.update_notebook_calibration(nb_neutral, hand)
        time.sleep(0.015)
        done, msg, prog = self.calib.update_notebook_calibration(nb_neutral, hand)
        self.assertFalse(done)
        self.assertEqual(self.calib.calibration_phase, "nb_steer_range")
        self.assertAlmostEqual(self.calib.config["notebook_neutral_angle"], 0.5, delta=0.5)

        # Step 2: Rotate Left and Right
        nb_l = NotebookData(is_detected=True, rotation_angle=-26.0, raw_pitch=2.0, height=180.0, width=280.0)
        nb_r = NotebookData(is_detected=True, rotation_angle=28.0, raw_pitch=2.0, height=180.0, width=280.0)
        self.calib.update_notebook_calibration(nb_l, hand)
        self.calib.update_notebook_calibration(nb_r, hand)
        time.sleep(0.015)
        done, msg, prog = self.calib.update_notebook_calibration(nb_r, hand)
        self.assertFalse(done)
        self.assertEqual(self.calib.calibration_phase, "nb_brake_pitch")

        # Step 3: Perform intended braking movement (Forward tilt: pitch=20.0 deg, height=140.0)
        nb_tilt = NotebookData(
            is_detected=True, rotation_angle=0.0, raw_pitch=20.0, height=140.0, width=280.0
        )
        self.calib.update_notebook_calibration(nb_tilt, hand)
        time.sleep(0.015)
        done, msg, prog = self.calib.update_notebook_calibration(nb_tilt, hand)
        self.assertFalse(done)
        self.assertEqual(self.calib.calibration_phase, "nb_return_neutral")
        self.assertGreater(self.calib.config["notebook_brake_pitch_thresh"], 5.0)

        # Step 4: Return notebook to neutral position
        self.calib.update_notebook_calibration(nb_neutral, hand)
        time.sleep(0.015)
        done, msg, prog = self.calib.update_notebook_calibration(nb_neutral, hand)
        self.assertFalse(done)
        self.assertEqual(self.calib.calibration_phase, "nb_right_index")

        # Step 5: Raise right index finger (acceleration verification)
        hand_idx = HandData(is_detected=True, right_hand_detected=True, right_index_raised=True)
        self.calib.update_notebook_calibration(nb_neutral, hand_idx)
        time.sleep(0.015)
        done, msg, prog = self.calib.update_notebook_calibration(nb_neutral, hand_idx)
        self.assertTrue(done)
        self.assertEqual(self.calib.calibration_phase, "done")
        self.assertTrue(self.calib.config.get("notebook_calibrated"))
        self.assertTrue(self.calib.notebook_steering_enabled)

    def test_rejection_when_braking_movement_is_too_subtle(self):
        self.calib.start_notebook_calibration()

        # Step 1: Neutral
        nb_neutral = NotebookData(is_detected=True, raw_angle=0.0, raw_pitch=0.0, height=180.0, width=280.0)
        hand = HandData(is_detected=True)
        self.calib.update_notebook_calibration(nb_neutral, hand)
        time.sleep(0.015)
        self.calib.update_notebook_calibration(nb_neutral, hand)
        self.assertEqual(self.calib.calibration_phase, "nb_steer_range")

        # Step 2: Range
        time.sleep(0.015)
        self.calib.update_notebook_calibration(nb_neutral, hand)
        self.assertEqual(self.calib.calibration_phase, "nb_brake_pitch")

        # Step 3: Braking movement that is too subtle (pitch delta = 1.0 deg < 5.0 deg, height drop = 0)
        nb_subtle = NotebookData(is_detected=True, raw_angle=0.0, raw_pitch=1.0, height=178.0, width=280.0)
        self.calib.update_notebook_calibration(nb_subtle, hand)
        time.sleep(0.015)
        done, msg, prog = self.calib.update_notebook_calibration(nb_subtle, hand)

        # Must reject calibration with a clear warning
        self.assertFalse(done)
        self.assertFalse(self.calib.is_calibrating)
        self.assertEqual(self.calib.calibration_phase, "idle")
        self.assertIn("Rejected", msg)
        self.assertIsNotNone(self.calib.calibration_warning)


if __name__ == "__main__":
    unittest.main()

