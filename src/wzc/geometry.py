"""Pure geometry helpers. No I/O, no state."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

Point = tuple[float, float]
Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


def foot_point(xyxy: Box) -> Point:
    """Bottom-centre of a box.

    A high camera sees a person's waist drift into zones behind them, so the box centre
    is the wrong anchor. The feet are what touch the floor zone.
    """
    x1, _, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, float(y2))


def as_contour(polygon: Sequence[Sequence[float]]) -> np.ndarray:
    """Polygon as the float32 Nx1x2 array OpenCV expects."""
    return np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)


def inside(point: Point, polygon: Sequence[Sequence[float]] | np.ndarray) -> bool:
    """True if the point is inside the polygon or on its edge."""
    contour = polygon if isinstance(polygon, np.ndarray) else as_contour(polygon)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def polygon_area(polygon: Sequence[Sequence[float]]) -> float:
    """Absolute area in pixels (shoelace formula)."""
    pts = np.asarray(polygon, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)
