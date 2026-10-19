"""
Notebook Tracking Module for Burnout Camera Drive.
Detects and tracks a physical notebook (used as a physical steering wheel)
using a hybrid approach:
1. Adaptive HSV color segmentation with morphological filtering.
2. Geometric edge analysis and quadrilateral fitting (color-independent).
3. Hand spatial association (notebook bounded between left and right hands).

Tracks 4 corners, center, width, height, orientation, and relative rotation angle.
Includes Exponential Moving Average (EMA) smoothing to eliminate steering jitter.
"""

import math
import time
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

logger = logging.getLogger("burnout.notebook")


@dataclass
class NotebookData:
    is_detected: bool = False
    corners: List[Tuple[int, int]] = field(default_factory=list)  # [TL, TR, BR, BL]
    center: Tuple[int, int] = (0, 0)
    width: float = 0.0
    height: float = 0.0
    area: float = 0.0
    raw_angle: float = 0.0              # in-plane roll angle (-90 to +90)
    rotation_angle: float = 0.0         # relative to calibrated neutral angle (Steering)
    pitch_angle: float = 0.0            # relative out-of-plane forward/backward tilt (Braking)
    raw_pitch: float = 0.0              # absolute pitch angle from 3D pose
    height_ratio: float = 1.0           # current height / neutral height (Foreshortening)
    trapezoid_ratio: float = 0.0        # (W_top - W_bottom) / W_avg
    is_braking: bool = False            # True when forward tilt exceeds brake threshold
    brake_deflection: float = 0.0       # 0.0 to 1.0
    tracking_confidence: float = 0.0
    detection_method: str = "none"      # "color+contour", "geometric_edge", or "hand_prior"
    aspect_ratio: float = 1.0
    calibration_warning: Optional[str] = None


class NotebookTracker:
    """
    Tracks a physical notebook held as a steering wheel using OpenCV.
    Detects 2D roll (steering) and 3D forward pitch / foreshortening (braking).
    """

    def __init__(self,
                 hsv_lower: Optional[Tuple[int, int, int]] = None,
                 hsv_upper: Optional[Tuple[int, int, int]] = None,
                 smoothing_factor: float = 0.40,
                 smoothing: Optional[float] = None):
        # Default color range: Vibrant Orange (as in reference image)
        self.hsv_lower = np.array(hsv_lower if hsv_lower is not None else [8, 65, 65], dtype=np.uint8)
        self.hsv_upper = np.array(hsv_upper if hsv_upper is not None else [32, 255, 255], dtype=np.uint8)
        self.smoothing_factor = smoothing if smoothing is not None else smoothing_factor

        # Calibrated neutral references (for steering and braking)
        self.neutral_angle: float = 0.0
        self.neutral_pitch: float = 0.0
        self.neutral_height: float = 0.0
        self.neutral_width: float = 0.0
        self.calibrated_size: Optional[Tuple[float, float]] = None

        # Braking thresholds
        self.brake_pitch_thresh: float = 10.0      # degrees forward tilt to engage brake
        self.brake_height_thresh: float = 0.12     # foreshortening drop ratio to engage brake
        self._is_braking_latched: bool = False

        # Temporal smoothing buffers (EMA)
        self._smoothed_corners: Optional[np.ndarray] = None
        self._smoothed_center: Optional[Tuple[float, float]] = None
        self._smoothed_angle: Optional[float] = None
        self._smoothed_pitch: Optional[float] = None
        self._smoothed_height: Optional[float] = None
        self._consecutive_lost = 0

    def set_color_bounds(self, lower: Tuple[int, int, int], upper: Tuple[int, int, int]) -> None:
        self.hsv_lower = np.array(lower, dtype=np.uint8)
        self.hsv_upper = np.array(upper, dtype=np.uint8)
        logger.info(f"Updated notebook HSV bounds: Lower={lower}, Upper={upper}")

    def calibrate_color_from_roi(self, frame_bgr: np.ndarray, roi_box: Optional[Tuple[int, int, int, int]] = None) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Samples color from the center of the frame or provided ROI box (x, y, w, h)
        to calibrate for any notebook color (orange, blue, black, white, red, etc.).
        """
        h, w = frame_bgr.shape[:2]
        if roi_box is None:
            # Default center box 20% of frame width/height
            bw = int(w * 0.25)
            bh = int(h * 0.25)
            bx = (w - bw) // 2
            by = (h - bh) // 2
            roi_box = (bx, by, bw, bh)

        bx, by, bw, bh = roi_box
        bx = max(0, min(w - 1, bx))
        by = max(0, min(h - 1, by))
        bw = max(10, min(w - bx, bw))
        bh = max(10, min(h - by, bh))

        patch = frame_bgr[by:by+bh, bx:bx+bw]
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

        # Median HSV
        med_h = int(np.median(hsv_patch[:, :, 0]))
        med_s = int(np.median(hsv_patch[:, :, 1]))
        med_v = int(np.median(hsv_patch[:, :, 2]))

        # Calculate bounds
        h_min = max(0, med_h - 18)
        h_max = min(179, med_h + 18)
        s_min = max(40, med_s - 65)
        s_max = 255
        v_min = max(45, med_v - 75)
        v_max = 255

        lower = (h_min, s_min, v_min)
        upper = (h_max, s_max, v_max)
        self.set_color_bounds(lower, upper)
        logger.info(f"Calibrated notebook color from ROI. Center HSV=({med_h}, {med_s}, {med_v})")
        return lower, upper

    def process_frame(self, frame_bgr: np.ndarray,
                      hand_roi: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None) -> NotebookData:
        """
        Process a camera frame and detect the notebook's boundaries, corners, and angle.
        hand_roi: optional ((left_x, left_y), (right_x, right_y)) in normalized coordinates
                  providing spatial constraint between the user's two hands.
        """
        data = NotebookData()
        if frame_bgr is None:
            return data

        h, w = frame_bgr.shape[:2]
        frame_area = float(h * w)

        best_contour = None
        method = "none"
        confidence = 0.0

        # -------------------------------------------------------------
        # METHOD 1: COLOR SEGMENTATION + MORPHOLOGY
        # -------------------------------------------------------------
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Morphological opening and closing
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel_open)

        cnts, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_cnts = []
        for c in cnts:
            area = cv2.contourArea(c)
            # Area between 2.0% and 60% of image
            if (0.02 * frame_area) <= area <= (0.60 * frame_area):
                # Score contour based on centrality, aspect ratio, and area
                score = self._score_candidate_contour(c, w, h, hand_roi)
                valid_cnts.append((score, area, c))

        if valid_cnts:
            valid_cnts.sort(key=lambda x: (x[0], x[1]), reverse=True)
            best_contour = valid_cnts[0][2]
            method = "color+contour"
            confidence = min(1.0, valid_cnts[0][0])

        # -------------------------------------------------------------
        # METHOD 2: GEOMETRIC EDGE TRACKING FALLBACK
        # (Used if color detection yields low confidence or non-orange notebook)
        # -------------------------------------------------------------
        if best_contour is None or confidence < 0.40:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 35, 120)

            # If hands are available, mask only the region between hands
            if hand_roi and hand_roi[0] and hand_roi[1]:
                lx = int(min(hand_roi[0][0], hand_roi[1][0]) * w) - 20
                rx = int(max(hand_roi[0][0], hand_roi[1][0]) * w) + 20
                my = int((hand_roi[0][1] + hand_roi[1][1]) / 2.0 * h)
                ty = max(0, my - int(h * 0.35))
                by = min(h, my + int(h * 0.35))
                lx = max(0, lx)
                rx = min(w, rx)

                hand_mask = np.zeros_like(edges)
                cv2.rectangle(hand_mask, (lx, ty), (rx, by), 255, -1)
                edges = cv2.bitwise_and(edges, hand_mask)

            kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_edge)
            edge_cnts, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            edge_candidates = []
            for ec in edge_cnts:
                area = cv2.contourArea(ec)
                if (0.025 * frame_area) <= area <= (0.60 * frame_area):
                    peri = cv2.arcLength(ec, True)
                    approx = cv2.approxPolyDP(ec, 0.04 * peri, True)
                    if len(approx) in (4, 5, 6) and cv2.isContourConvex(approx):
                        score = self._score_candidate_contour(ec, w, h, hand_roi)
                        edge_candidates.append((score, area, ec))

            if edge_candidates:
                edge_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
                if best_contour is None or edge_candidates[0][0] > confidence:
                    best_contour = edge_candidates[0][2]
                    method = "geometric_edge"
                    confidence = 0.85

        # -------------------------------------------------------------
        # EXTRACT RECTANGLE, CORNERS, 3D POSE & FORWARD TILT BRAKING
        # -------------------------------------------------------------
        if best_contour is None:
            self._consecutive_lost += 1
            if self._consecutive_lost > 3:
                self._smoothed_corners = None
                self._smoothed_center = None
                self._smoothed_angle = None
                self._smoothed_pitch = None
                self._smoothed_height = None
                self._is_braking_latched = False
            data.is_detected = False
            return data

        self._consecutive_lost = 0
        min_rect = cv2.minAreaRect(best_contour)
        raw_box = cv2.boxPoints(min_rect)
        raw_box = np.array(raw_box, dtype=np.float32)

        # Sort corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        sorted_corners = self._order_corners(raw_box)

        # Center point
        raw_cx = float(min_rect[0][0])
        raw_cy = float(min_rect[0][1])

        # Width and height of notebook (long side vs short side)
        side_a = float(np.linalg.norm(sorted_corners[1] - sorted_corners[0]))
        side_b = float(np.linalg.norm(sorted_corners[3] - sorted_corners[0]))
        nb_w = max(side_a, side_b)
        nb_h = min(side_a, side_b)

        # Perspective analysis: Top width vs Bottom width
        top_w = float(np.linalg.norm(sorted_corners[1] - sorted_corners[0]))
        bot_w = float(np.linalg.norm(sorted_corners[2] - sorted_corners[3]))
        avg_w = max(1.0, (top_w + bot_w) / 2.0)
        trapezoid_ratio = (top_w - bot_w) / avg_w

        # Calculate in-plane rotation angle (Roll) from Top-Left to Top-Right
        tl = sorted_corners[0]
        tr = sorted_corners[1]
        dx = tr[0] - tl[0]
        dy = tr[1] - tl[1]
        raw_angle = math.degrees(math.atan2(dy, max(1.0, dx)))
        raw_angle = max(-65.0, min(65.0, raw_angle))

        # Estimate 3D pose (Roll, Pitch, Yaw)
        ref_w = self.neutral_width if self.neutral_width > 20.0 else nb_w
        ref_h = self.neutral_height if self.neutral_height > 20.0 else nb_h
        p_roll, raw_pitch, p_yaw = self.estimate_notebook_pose(sorted_corners, ref_w, ref_h, w, h)

        # Apply Exponential Moving Average (EMA) smoothing
        alpha = self.smoothing_factor
        if self._smoothed_corners is None:
            self._smoothed_corners = sorted_corners.copy()
            self._smoothed_center = (raw_cx, raw_cy)
            self._smoothed_angle = raw_angle
            self._smoothed_pitch = raw_pitch
            self._smoothed_height = nb_h
        else:
            self._smoothed_corners = alpha * sorted_corners + (1.0 - alpha) * self._smoothed_corners
            self._smoothed_center = (
                alpha * raw_cx + (1.0 - alpha) * self._smoothed_center[0],
                alpha * raw_cy + (1.0 - alpha) * self._smoothed_center[1]
            )
            self._smoothed_angle = alpha * raw_angle + (1.0 - alpha) * self._smoothed_angle
            self._smoothed_pitch = alpha * raw_pitch + (1.0 - alpha) * self._smoothed_pitch
            self._smoothed_height = alpha * nb_h + (1.0 - alpha) * self._smoothed_height

        # Foreshortening calculation (h / neutral_h)
        if self.neutral_height > 15.0:
            height_ratio = round(self._smoothed_height / self.neutral_height, 3)
        else:
            height_ratio = 1.0

        # Relative pitch calculation (delta from neutral)
        pitch_delta = abs(self._smoothed_pitch - self.neutral_pitch)

        # -------------------------------------------------------------
        # DISTINGUISH BRAKING (FORWARD PITCH) FROM STEERING (ROLL)
        # -------------------------------------------------------------
        is_braking = False
        brake_deflection = 0.0

        if self.neutral_height > 15.0:
            # Active calibrated braking: Forward tilt or significant foreshortening
            # Hysteresis: engagement threshold vs release threshold
            engage_pitch = self.brake_pitch_thresh
            release_pitch = max(3.0, self.brake_pitch_thresh - 3.5)
            engage_drop = self.brake_height_thresh
            release_drop = max(0.04, self.brake_height_thresh - 0.04)

            foreshorten_drop = 1.0 - height_ratio

            if not self._is_braking_latched:
                if (pitch_delta >= engage_pitch) or (foreshorten_drop >= engage_drop):
                    self._is_braking_latched = True
            else:
                if (pitch_delta >= release_pitch) or (foreshorten_drop >= release_drop):
                    self._is_braking_latched = True
                else:
                    self._is_braking_latched = False

            is_braking = self._is_braking_latched
            if is_braking:
                norm_p = max(0.0, min(1.0, (pitch_delta - release_pitch) / max(6.0, engage_pitch)))
                norm_h = max(0.0, min(1.0, (foreshorten_drop - release_drop) / 0.15))
                brake_deflection = max(norm_p, norm_h, 0.5)

        # Output data
        data.is_detected = True
        data.detection_method = method
        data.tracking_confidence = confidence
        data.center = (int(self._smoothed_center[0]), int(self._smoothed_center[1]))
        data.width = nb_w
        data.height = self._smoothed_height
        data.area = float(cv2.contourArea(best_contour))
        data.aspect_ratio = round(nb_w / max(1.0, self._smoothed_height), 2)
        data.raw_angle = self._smoothed_angle
        data.rotation_angle = self._smoothed_angle - self.neutral_angle
        data.raw_pitch = self._smoothed_pitch
        data.pitch_angle = pitch_delta
        data.height_ratio = height_ratio
        data.trapezoid_ratio = round(trapezoid_ratio, 3)
        data.is_braking = is_braking
        data.brake_deflection = round(brake_deflection, 2)
        data.corners = [(int(p[0]), int(p[1])) for p in self._smoothed_corners]
        return data

    def _score_candidate_contour(self, cnt: np.ndarray, frame_w: int, frame_h: int,
                                hand_roi: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None) -> float:
        """
        Scores a candidate contour to prioritize the notebook held in front of the user
        over background objects like furniture or walls.
        """
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        if M['m00'] <= 0:
            return 0.0

        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']

        # Ideal center: between hands if available, else center of camera frame
        if hand_roi and hand_roi[0] and hand_roi[1]:
            hx = (hand_roi[0][0] + hand_roi[1][0]) / 2.0 * frame_w
            hy = (hand_roi[0][1] + hand_roi[1][1]) / 2.0 * frame_h
        else:
            hx = frame_w / 2.0
            hy = frame_h * 0.55

        dist = math.hypot(cx - hx, cy - hy)
        max_dist = math.hypot(frame_w, frame_h) * 0.5
        center_score = max(0.0, 1.0 - (dist / max(1.0, max_dist)))

        # Aspect ratio score: notebooks typically have aspect ratio 1.1 to 2.4
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw, bh) / max(1.0, min(bw, bh))
        aspect_score = 1.0 if (1.1 <= aspect <= 2.5) else 0.4

        # Area score: notebook covers 3% to 45% of image
        norm_area = area / float(frame_w * frame_h)
        area_score = 1.0 if (0.03 <= norm_area <= 0.45) else 0.5

        # Avoid contours touching the very edge border x=0 or y=0
        border_penalty = 0.5 if (x <= 5 or (x + bw) >= frame_w - 5) else 1.0

        return (center_score * 0.45 + aspect_score * 0.30 + area_score * 0.25) * border_penalty

    def estimate_notebook_pose(self, corners: np.ndarray, ref_w: float, ref_h: float,
                                cam_w: int, cam_h: int) -> Tuple[float, float, float]:
        """
        Estimate 3D orientation (roll, pitch, yaw) of the planar rectangular notebook.
        Roll: In-plane steering rotation (degrees).
        Pitch: Out-of-plane forward/backward tilt (degrees, used for braking).
        Yaw: Left/right facing angle.
        """
        if len(corners) != 4 or ref_w <= 10.0 or ref_h <= 10.0:
            return 0.0, 0.0, 0.0

        object_points = np.array([
            [-ref_w / 2.0, -ref_h / 2.0, 0.0],
            [ ref_w / 2.0, -ref_h / 2.0, 0.0],
            [ ref_w / 2.0,  ref_h / 2.0, 0.0],
            [-ref_w / 2.0,  ref_h / 2.0, 0.0]
        ], dtype=np.float32)

        f = float(cam_w)
        camera_matrix = np.array([
            [f, 0.0, cam_w / 2.0],
            [0.0, f, cam_h / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        try:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                corners.astype(np.float32),
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not success:
                return 0.0, 0.0, 0.0

            R, _ = cv2.Rodrigues(rvec)
            # Roll (in-plane steering around optical Z axis)
            roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
            # Pitch (forward/backward tilt around X axis)
            pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
            yaw = math.degrees(math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2])))
            return roll, pitch, yaw
        except Exception as e:
            logger.debug(f"solvePnP pose fallback: {e}")
            return 0.0, 0.0, 0.0

    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
        """
        Sort 4 corner points in exact order:
        [0]: Top-Left (TL)
        [1]: Top-Right (TR)
        [2]: Bottom-Right (BR)
        [3]: Bottom-Left (BL)
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        # Sum of x + y: Top-Left has smallest sum, Bottom-Right has largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Difference of y - x: Top-Right has smallest diff, Bottom-Left has largest diff
        diff = pts[:, 1] - pts[:, 0]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
