"""Debug frame overlay writer."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon


def draw_overlay(
    frame: np.ndarray,
    polygons: Dict[str, Polygon],
    interaction_point: Optional[Tuple[float, float]],
    detected_marker_ids: Iterable[int],
    predicted_sku: Optional[str],
    header_text: str,
) -> np.ndarray:
    out = frame.copy()

    for sku, poly in polygons.items():
        pts = np.array(poly.exterior.coords, dtype=np.int32)
        color = (160, 160, 160)
        thickness = 1
        if predicted_sku and sku == predicted_sku:
            color = (0, 200, 0)
            thickness = 2
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)

        c = poly.centroid
        cv2.putText(
            out,
            sku,
            (int(c.x), int(c.y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    if interaction_point is not None:
        cv2.circle(out, (int(interaction_point[0]), int(interaction_point[1])), 6, (0, 255, 255), -1)

    ids_text = "ids=" + ",".join(str(int(i)) for i in sorted(detected_marker_ids))
    cv2.putText(out, header_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, ids_text, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def write_debug_frame(
    output_dir: Path,
    video_stem: str,
    pick_index: int,
    image: np.ndarray,
) -> Path:
    p = output_dir / "debug" / video_stem
    p.mkdir(parents=True, exist_ok=True)
    out_path = p / f"pick_{pick_index:02d}.png"
    cv2.imwrite(str(out_path), image)
    return out_path
