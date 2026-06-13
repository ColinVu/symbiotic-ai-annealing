"""Real-time OpenCV visualization for debugging inference."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
from shapely.geometry import Polygon

from config import VerifierConfig
from io_utils import ArucoMap


class FrameVisualizer:
    """Show annotated voting frames during processing (cv2.imshow)."""

    WINDOW_NAME = "bin_picking_verifier --visualized"

    def __init__(self, cfg: VerifierConfig, subtitle: str = "") -> None:
        self.cfg = cfg
        self.subtitle = subtitle
        self._paused = False
        self.quit_requested = False
        self._fps_ema = 0.0
        self._last_t = time.perf_counter()

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.WINDOW_NAME)
        except cv2.error:
            pass

    @staticmethod
    def _resize_for_display(frame: np.ndarray, target_w: int) -> tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        if w <= target_w:
            return frame, 1.0
        scale = target_w / float(w)
        nh = max(1, int(round(h * scale)))
        return cv2.resize(frame, (target_w, nh), interpolation=cv2.INTER_AREA), scale

    @staticmethod
    def _depth_inset(depth_map: np.ndarray, size: int = 200) -> np.ndarray:
        d = depth_map.astype(np.float32)
        lo, hi = float(np.percentile(d, 5)), float(np.percentile(d, 95))
        if hi <= lo:
            hi = lo + 1e-6
        norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
        gray = (norm * 255).astype(np.uint8)
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        return cv2.resize(colored, (size, size), interpolation=cv2.INTER_AREA)

    def draw_vote_frame(
        self,
        frame_idx: int,
        frame_bgr: np.ndarray,
        detections: Dict[int, np.ndarray],
        polygons: Optional[Dict[str, Polygon]],
        frame_state: Optional[Dict[str, Any]],
        aruco_map: ArucoMap,
    ) -> None:
        if self.quit_requested:
            return

        now = time.perf_counter()
        dt = max(1e-6, now - self._last_t)
        self._last_t = now
        inst_fps = 1.0 / dt
        self._fps_ema = inst_fps if self._fps_ema == 0.0 else (0.85 * self._fps_ema + 0.15 * inst_fps)

        out = frame_bgr.copy()

        # Draw ArUco markers and SKU labels
        for marker_id, corners in detections.items():
            pts = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], isClosed=True, color=(60, 200, 60), thickness=2)
            centroid = pts.reshape(-1, 2).mean(axis=0)
            entry = aruco_map.marker_id_to_entry.get(marker_id)
            label = entry.sku if entry is not None else f"id{marker_id}"
            lx, ly = int(centroid[0]), int(centroid[1])
            cv2.putText(
                out,
                label,
                (lx + 8, ly + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (40, 255, 180),
                2,
                cv2.LINE_AA,
            )

        # Draw bin polygons
        if polygons:
            for _sku, poly in polygons.items():
                pts = np.array(poly.exterior.coords, dtype=np.int32)
                cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, (80, 80, 110), 1, cv2.LINE_AA)

        gate_status = "NO DEPTH MODEL"
        intrusion_pixels = 0
        plane_normal = None
        plane_offset = None
        plane_broken = False
        depth_map = None
        intrusion_mask = None

        if frame_state is not None:
            gate_status = str(frame_state.get("gate_status", "NO DEPTH MODEL"))
            intrusion_pixels = int(frame_state.get("intrusion_pixels", 0))
            plane_normal = frame_state.get("plane_normal")
            plane_offset = frame_state.get("plane_offset")
            plane_broken = bool(frame_state.get("plane_broken", False))
            depth_map = frame_state.get("depth_map")
            intrusion_mask = frame_state.get("intrusion_mask")

            # Overlay intrusion mask (plane-breaking pixels)
            if intrusion_mask is not None and np.any(intrusion_mask):
                # Create colored overlay for intrusion pixels
                overlay = out.copy()
                overlay[intrusion_mask] = [60, 60, 255]  # Red tint for intrusion
                cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)

        plane_normal_str = "n/a"
        if plane_normal is not None:
            n = plane_normal
            plane_normal_str = f"[{n[0]:+.3f}, {n[1]:+.3f}, {n[2]:+.3f}]"

        lines = [
            f"frame {frame_idx}",
            self.subtitle,
            f"markers {len(detections)}  fps ~{self._fps_ema:.1f}" + ("  [PAUSED]" if self._paused else ""),
            f"gate: {gate_status}",
            f"plane_normal: {plane_normal_str}",
            f"plane_offset: {plane_offset:.4f}" if plane_offset is not None else "plane_offset: n/a",
            f"intrusion_pixels: {intrusion_pixels}",
            f"plane_broken: {plane_broken}",
        ]

        y0 = 22
        line_h = 22
        overlay_h = 18 + len(lines) * line_h
        cv2.rectangle(out, (4, 4), (820, overlay_h), (0, 0, 0), -1)
        cv2.rectangle(out, (4, 4), (820, overlay_h), (60, 60, 80), 1)
        for i, ln in enumerate(lines):
            cv2.putText(out, ln, (10, y0 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 245), 1, cv2.LINE_AA)

        if plane_broken:
            cv2.putText(
                out,
                f"PLANE BROKEN ({intrusion_pixels} px)",
                (out.shape[1] // 2 - 180, out.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (60, 60, 255),
                3,
                cv2.LINE_AA,
            )
        elif gate_status == "NO INTRUSION":
            cv2.putText(
                out,
                "NO INTRUSION",
                (out.shape[1] // 2 - 100, out.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (80, 200, 80),
                2,
                cv2.LINE_AA,
            )

        if self.cfg.viz_show_depth_inset and depth_map is not None:
            inset = self._depth_inset(depth_map)
            ih, iw = inset.shape[:2]
            x0 = out.shape[1] - iw - 12
            y1 = 12 + ih
            out[12:y1, x0 : x0 + iw] = inset
            cv2.rectangle(out, (x0 - 1, 11), (x0 + iw + 1, y1 + 1), (200, 200, 200), 1)

        disp, _scale = self._resize_for_display(out, self.cfg.viz_window_width)
        cv2.imshow(self.WINDOW_NAME, disp)

        key = cv2.waitKey(0 if self._paused else 1) & 0xFF
        if key == ord(" "):
            self._paused = not self._paused
        elif key in (27, ord("q")):
            self.quit_requested = True
