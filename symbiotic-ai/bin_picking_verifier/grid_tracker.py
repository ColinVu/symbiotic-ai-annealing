"""ArUco detection and homography-based virtual 4x6 bin polygons."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import cv2
import numpy as np
from shapely.geometry import Polygon

from config import VerifierConfig
from io_utils import ArucoMap

logger = logging.getLogger(__name__)


class GridTracker:
    def __init__(self, aruco_map: ArucoMap, config: VerifierConfig) -> None:
        self.aruco_map = aruco_map
        self.config = config

        if aruco_map.dictionary != "DICT_5X5_1000":
            logger.warning("Aruco map dictionary=%s, forcing DICT_5X5_1000 detector", aruco_map.dictionary)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshConstant = 7
        params.perspectiveRemoveIgnoredMarginPerCell = 0.13
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)

    def detect(self, frame_gray: np.ndarray) -> Dict[int, np.ndarray]:
        """Return marker_id -> corners(4x2) in image pixels."""
        corners, ids, _ = self.detector.detectMarkers(frame_gray)
        if ids is None or len(ids) == 0:
            return {}

        out: Dict[int, np.ndarray] = {}
        for marker_id, pts in zip(ids.flatten().tolist(), corners):
            out[int(marker_id)] = np.asarray(pts[0], dtype=np.float32)
        return out

    @staticmethod
    def _marker_center(corners: np.ndarray) -> np.ndarray:
        return np.mean(corners, axis=0).astype(np.float32)

    def _homography(self, detections: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
        src_pts = []
        dst_pts = []

        for marker_id, corners in detections.items():
            entry = self.aruco_map.marker_id_to_entry.get(marker_id)
            if entry is None:
                continue
            src_pts.append([float(entry.col - 1), float(entry.row - 1)])
            dst_pts.append(self._marker_center(corners).tolist())

        if len(src_pts) < 4:
            return None

        src = np.asarray(src_pts, dtype=np.float32)
        dst = np.asarray(dst_pts, dtype=np.float32)

        try:
            H, _ = cv2.findHomography(
                src,
                dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.config.homography_ransac_reproj_threshold,
            )
        except cv2.error as e:
            logger.warning("findHomography failed: %s", e)
            return None

        if H is None or not np.isfinite(H).all():
            return None
        return H

    def build_grid_polygons(self, detections: Dict[int, np.ndarray]) -> Optional[Dict[str, Polygon]]:
        """Return sku -> projected polygon for all bins, or None when homography fails."""
        try:
            H = self._homography(detections)
            if H is None:
                return None

            half_extent = float(self.config.polygon_half_extent_factor)
            polygons: Dict[str, Polygon] = {}
            for entry in self.aruco_map.bins:
                cx = float(entry.col - 1)
                cy = float(entry.row - 1)
                square = np.array(
                    [[[cx - half_extent, cy - half_extent],
                      [cx + half_extent, cy - half_extent],
                      [cx + half_extent, cy + half_extent],
                      [cx - half_extent, cy + half_extent]]],
                    dtype=np.float32,
                )
                warped = cv2.perspectiveTransform(square, H)[0]
                poly = Polygon([(float(x), float(y)) for x, y in warped])
                if poly.is_valid and not poly.is_empty:
                    polygons[entry.sku] = poly

            return polygons if polygons else None
        except Exception as e:
            logger.warning("build_grid_polygons failed: %s", e)
            return None
