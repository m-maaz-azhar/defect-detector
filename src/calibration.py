from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


DEFAULT_MARKER_HSV_LOWER = (35, 80, 80)
DEFAULT_MARKER_HSV_UPPER = (85, 255, 255)


@dataclass(frozen=True)
class CalibrationResult:
    method: str
    mm_per_px: float


def fixed_scale(mm_per_px: float) -> CalibrationResult:
    if mm_per_px <= 0:
        raise ValueError("fixed mm_per_px must be > 0")
    return CalibrationResult(method="fixed", mm_per_px=float(mm_per_px))


def marker_scale_from_image(
    image_bgr: np.ndarray,
    marker_size_mm: float,
    hsv_lower: tuple[int, int, int] = DEFAULT_MARKER_HSV_LOWER,
    hsv_upper: tuple[int, int, int] = DEFAULT_MARKER_HSV_UPPER,
    min_contour_area_px: float = 250.0,
) -> Optional[CalibrationResult]:
    if marker_size_mm <= 0:
        raise ValueError("marker_size_mm must be > 0")

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower, dtype=np.uint8), np.array(hsv_upper, dtype=np.uint8))

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < min_contour_area_px:
        return None

    (_, _), (width, height), _ = cv2.minAreaRect(contour)
    marker_px = float((width + height) / 2.0)
    if marker_px <= 0:
        return None

    mm_per_px = marker_size_mm / marker_px
    return CalibrationResult(method="marker", mm_per_px=float(mm_per_px))


def area_mm2(mask_area_px: int, mm_per_px: float) -> float:
    if mask_area_px < 0:
        raise ValueError("mask_area_px must be >= 0")
    if mm_per_px <= 0:
        raise ValueError("mm_per_px must be > 0")
    return float(mask_area_px) * float(mm_per_px**2)


def severity_bucket(mask_area_px: int, small_thresh_px: int = 800, medium_thresh_px: int = 3000) -> str:
    if mask_area_px < 0:
        raise ValueError("mask_area_px must be >= 0")
    if mask_area_px < small_thresh_px:
        return "small"
    if mask_area_px < medium_thresh_px:
        return "medium"
    return "large"

