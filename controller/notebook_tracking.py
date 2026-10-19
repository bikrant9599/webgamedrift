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
    raw_angle: float = 0.0              # degrees from horizontal (-90 to +90)
    rotation_angle: float = 0.0         # relative to calibrated neutral angle
    tracking_confidence: float = 0.0
    detection_method: str = "none"      # "color+contour", "geometric_edge", or "hand_prior"
    aspect_ratio: float = 1.0


class NotebookTracker:
    """
    Tracks a physical notebook held as a steering wheel using OpenCV.
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

        # Calibrated neutral reference angle
        self.neutral_angle: float = 0.0
        self.calibrated_size: Optional[Tuple[float, float]] = None

        # Temporal smoothing buffers
        self._smoothed_corners: Optional[np.ndarray] = None
        self._smoothed_center: Optional[Tuple[float, float]] = None
        self._smoothed_angle: Optional[float] = None
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
            # Area between 2.5% and 55% of image
            if (0.025 * frame_area) <= area <= (0.60 * frame_area):
                valid_cnts.append((area, c))

        if valid_cnts:
            valid_cnts.sort(key=lambda x: x[0], reverse=True)
            best_contour = valid_cnts[0][1]
            method = "color+contour"
            confidence = min(1.0, valid_cnts[0][0] / (0.12 * frame_area))

        # -------------------------------------------------------------
        # METHOD 2: GEOMETRIC EDGE TRACKING FALLBACK
        # (Used if color detection yields low confidence or non-orange notebook)
        # -------------------------------------------------------------
        if best_contour is None or confidence < 0.45:
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
                if (0.03 * frame_area) <= area <= (0.60 * frame_area):
                    peri = cv2.arcLength(ec, True)
                    approx = cv2.approxPolyDP(ec, 0.04 * peri, True)
                    if len(approx) in (4, 5, 6) and cv2.isContourConvex(approx):
                        edge_candidates.append((area, ec))

            if edge_candidates:
                edge_candidates.sort(key=lambda x: x[0], reverse=True)
                # If better than color contour, use it
                if best_contour is None or edge_candidates[0][0] > cv2.contourArea(best_contour):
                    best_contour = edge_candidates[0][1]
                    method = "geometric_edge"
                    confidence = 0.80

        # -------------------------------------------------------------
        # EXTRACT RECTANGLE, CORNERS, AND ORIENTATION
        # -------------------------------------------------------------
        if best_contour is None:
            self._consecutive_lost += 1
            if self._consecutive_lost > 3:
                self._smoothed_corners = None
                self._smoothed_center = None
                self._smoothed_angle = None
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

        # Calculate rotation angle of top edge
        # Vector from Top-Left to Top-Right
        tl = sorted_corners[0]
        tr = sorted_corners[1]
        dx = tr[0] - tl[0]
        dy = tr[1] - tl[1]
        # In image coords, Y is downward:
        # If right side is tilted down (clockwise turn), dy > 0 -> angle > 0 (Steer Right)
        # If left side is tilted down (counter-clockwise turn), dy < 0 -> angle < 0 (Steer Left)
        raw_angle = math.degrees(math.atan2(dy, max(1.0, dx)))
        raw_angle = max(-65.0, min(65.0, raw_angle))

        # Apply Exponential Moving Average (EMA) smoothing
        alpha = self.smoothing_factor
        if self._smoothed_corners is None:
            self._smoothed_corners = sorted_corners.copy()
            self._smoothed_center = (raw_cx, raw_cy)
            self._smoothed_angle = raw_angle
        else:
            self._smoothed_corners = alpha * sorted_corners + (1.0 - alpha) * self._smoothed_corners
            self._smoothed_center = (
                alpha * raw_cx + (1.0 - alpha) * self._smoothed_center[0],
                alpha * raw_cy + (1.0 - alpha) * self._smoothed_center[1]
            )
            self._smoothed_angle = alpha * raw_angle + (1.0 - alpha) * self._smoothed_angle

        # Output data
        data.is_detected = True
        data.detection_method = method
        data.tracking_confidence = confidence
        data.center = (int(self._smoothed_center[0]), int(self._smoothed_center[1]))
        data.width = nb_w
        data.height = nb_h
        data.area = float(cv2.contourArea(best_contour))
        data.aspect_ratio = round(nb_w / max(1.0, nb_h), 2)
        data.raw_angle = self._smoothed_angle
        # Relative rotation angle relative to neutral reference
        data.rotation_angle = self._smoothed_angle - self.neutral_angle

        data.corners = [(int(p[0]), int(p[1])) for p in self._smoothed_corners]
        return data

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
