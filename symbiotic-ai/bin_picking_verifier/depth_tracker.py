"""Monocular depth estimation via DepthAnything V2 (HuggingFace transformers)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from shapely.geometry import Polygon
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

logger = logging.getLogger(__name__)


@dataclass
class BinSurfacePlane:
    """Fitted 3D plane for the bin grid surface in camera coordinates."""

    normal: np.ndarray  # unit vector, oriented toward camera
    offset: float  # plane equation: normal·p + offset = 0
    centroid: np.ndarray
    marker_points_3d: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DepthEstimator:
    """Lazy-loaded DepthAnything V2 depth map predictor."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        infer_size: int = 518,
    ) -> None:
        self.model_id = model_id
        self.infer_size = infer_size
        self.device = resolve_device(device)
        self._processor: Optional[AutoImageProcessor] = None
        self._model: Optional[AutoModelForDepthEstimation] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading depth model %s on %s", self.model_id, self.device)
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return float32 depth map HxW aligned to frame (larger = farther for DA-V2)."""
        self._ensure_loaded()
        assert self._processor is not None and self._model is not None

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        inputs = self._processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(**inputs)
            depth = outputs.predicted_depth.squeeze(0).float().cpu().numpy()

        if depth.shape[0] != h or depth.shape[1] != w:
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        return depth.astype(np.float32)


def sample_depth(depth_map: np.ndarray, x: float, y: float, radius_px: int) -> Optional[float]:
    """Robust median depth in a disk around (x, y)."""
    h, w = depth_map.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return None

    r = max(0, int(radius_px))
    x0 = max(0, cx - r)
    x1 = min(w, cx + r + 1)
    y0 = max(0, cy - r)
    y1 = min(h, cy + r + 1)
    patch = depth_map[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    return float(np.median(patch))


def marker_centers(detections: dict) -> List[Tuple[float, float]]:
    centers: List[Tuple[float, float]] = []
    for corners in detections.values():
        pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        c = pts.mean(axis=0)
        centers.append((float(c[0]), float(c[1])))
    return centers


def approximate_intrinsics(width: int, height: int) -> Tuple[float, float, float, float]:
    """Rough pinhole intrinsics when camera calibration is unavailable."""
    fx = float(width)
    fy = float(width)
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def unproject_pixels(
    xs: np.ndarray,
    ys: np.ndarray,
    depths: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Unproject pixel arrays to 3D camera coordinates (N, 3)."""
    z = depths.astype(np.float64)
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def fit_plane_svd(points_3d: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """Fit plane ax + by + cz + d = 0; normal oriented toward camera at origin."""
    pts = np.asarray(points_3d, dtype=np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        raise ValueError("Degenerate plane fit")
    normal = normal / norm
    if float(np.dot(normal, centroid)) > 0.0:
        normal = -normal
    offset = float(-np.dot(normal, centroid))
    return normal, offset, centroid


def signed_plane_distances(points_3d: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    """Signed distance to plane. Negative = past plane toward bins (away from camera)."""
    return points_3d @ normal + offset


def fit_bin_surface_plane(
    depth_map: np.ndarray,
    marker_centers_xy: Sequence[Tuple[float, float]],
    radius_px: int,
    min_markers: int,
) -> Optional[BinSurfacePlane]:
    """Fit a 3D plane to ArUco marker centers using depth unprojection."""
    h, w = depth_map.shape[:2]
    fx, fy, cx, cy = approximate_intrinsics(w, h)

    xs: List[float] = []
    ys: List[float] = []
    ds: List[float] = []
    for x, y in marker_centers_xy:
        d = sample_depth(depth_map, x, y, radius_px)
        if d is not None:
            xs.append(x)
            ys.append(y)
            ds.append(d)

    if len(ds) < min_markers:
        return None

    points_3d = unproject_pixels(
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        np.asarray(ds, dtype=np.float64),
        fx,
        fy,
        cx,
        cy,
    )
    normal, offset, centroid = fit_plane_svd(points_3d)
    return BinSurfacePlane(
        normal=normal,
        offset=offset,
        centroid=centroid,
        marker_points_3d=points_3d,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )


def _polygon_sample_pixels(poly: Polygon, width: int, height: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (xs, ys) pixel coordinates inside polygon, subsampled by stride."""
    minx, miny, maxx, maxy = poly.bounds
    x0 = max(0, int(np.floor(minx)))
    y0 = max(0, int(np.floor(miny)))
    x1 = min(width, int(np.ceil(maxx)) + 1)
    y1 = min(height, int(np.ceil(maxy)) + 1)

    xs_list: List[int] = []
    ys_list: List[int] = []
    stride = max(1, int(stride))
    for y in range(y0, y1, stride):
        for x in range(x0, x1, stride):
            if poly.contains((float(x), float(y))):
                xs_list.append(x)
                ys_list.append(y)

    if not xs_list:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    return np.asarray(xs_list, dtype=np.int32), np.asarray(ys_list, dtype=np.int32)


def score_plane_intrusions(
    depth_map: np.ndarray,
    plane: BinSurfacePlane,
    polygons: Dict[str, Polygon],
    valid_skus: Optional[Set[str]],
    cross_margin: float,
    pixel_stride: int,
) -> Tuple[Dict[str, int], np.ndarray, int]:
    """Count pixels crossing the fitted bin surface plane inside each bin polygon.

    Returns per-SKU intrusion counts, full-frame intrusion mask, and total intrusion pixels.
    """
    h, w = depth_map.shape[:2]
    intrusion_mask = np.zeros((h, w), dtype=bool)
    counts: Dict[str, int] = {}

    for sku, poly in polygons.items():
        if valid_skus is not None and sku not in valid_skus:
            continue

        xs, ys = _polygon_sample_pixels(poly, w, h, pixel_stride)
        if xs.size == 0:
            counts[sku] = 0
            continue

        depths = depth_map[ys, xs]
        points_3d = unproject_pixels(xs, ys, depths, plane.fx, plane.fy, plane.cx, plane.cy)
        signed = signed_plane_distances(points_3d, plane.normal, plane.offset)
        crossed = signed < -cross_margin

        count = int(np.count_nonzero(crossed))
        counts[sku] = count
        if count > 0:
            intrusion_mask[ys[crossed], xs[crossed]] = True

    total = int(np.count_nonzero(intrusion_mask))
    return counts, intrusion_mask, total
