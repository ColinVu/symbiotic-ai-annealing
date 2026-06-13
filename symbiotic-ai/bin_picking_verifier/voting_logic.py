"""Temporal scoring and picklist-constrained decision logic."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from config import VerifierConfig
from depth_tracker import (
    DepthEstimator,
    fit_bin_surface_plane,
    marker_centers,
    score_plane_intrusions,
)
from grid_tracker import GridTracker
from hand_tracker import HandTracker
from io_utils import ArucoMap, StateSegment
from video_reader import VideoReader

logger = logging.getLogger(__name__)

VisualizationFrameCallback = Optional[
    Callable[
        [
            int,
            np.ndarray,
            Dict[int, np.ndarray],
            Optional[Dict[str, Polygon]],
            Optional[Dict[str, Any]],
        ],
        bool,
    ]
]


@dataclass
class PickPrediction:
    pick_index: int
    picklist_block: int
    segment_frames: Tuple[int, int]
    voting_frames_used: int
    predicted_sku: Optional[str]
    predicted_marker_id: Optional[int]
    score: float
    second_best: Optional[dict]
    fallback_to_set_membership: bool
    occluded_baseline_markers: List[int]
    reason: Optional[str] = None
    debug: dict = field(default_factory=dict)


class PicklistState:
    def __init__(self, picklists: List[List[str]]) -> None:
        self.picklists = [list(x) for x in picklists]
        self.remaining = [list(x) for x in picklists]
        self.current_block = 0

    def _advance(self) -> None:
        while self.current_block < len(self.remaining) and len(self.remaining[self.current_block]) == 0:
            self.current_block += 1

    def choose(self, ranked_skus: List[str]) -> Tuple[int, Optional[str], bool]:
        self._advance()
        if self.current_block >= len(self.picklists):
            return max(0, len(self.picklists) - 1), None, False

        block = self.current_block
        remain = self.remaining[block]

        for sku in ranked_skus:
            if sku in remain:
                remain.remove(sku)
                self._advance()
                return block, sku, False

        valid_set = set(self.picklists[block])
        for sku in ranked_skus:
            if sku in valid_set:
                if sku in remain:
                    remain.remove(sku)
                self._advance()
                return block, sku, True

        if remain:
            chosen = remain[0]
            remain.remove(chosen)
            self._advance()
            logger.warning(
                "No ranked candidates in picklist block %d; defaulting to first remaining: %s",
                block,
                chosen,
            )
            return block, chosen, True

        if self.picklists[block]:
            chosen = self.picklists[block][0]
            logger.warning(
                "Picklist block %d exhausted; defaulting to first item: %s",
                block,
                chosen,
            )
            return block, chosen, True

        return block, None, True


def _window_frames(seg: StateSegment, cfg: VerifierConfig) -> Tuple[List[int], List[int]]:
    """Return all frames for both baseline and voting windows.
    
    With the new plane-breaking detection system, we no longer need to exclude
    the first 30% of frames. The baseline window is used to establish which
    markers are visible for the occlusion bonus, and the voting window is used
    for scoring. Both now encompass the entire segment.
    """
    s, e = seg.start_frame, seg.end_frame
    all_frames = list(range(s, e + 1))
    return all_frames, all_frames


def _score_plane_intrusion(
    depth_map: np.ndarray,
    plane,
    polygons: Dict[str, Polygon],
    scores: Dict[str, float],
    baseline_visible: Set[int],
    detected_ids: Set[int],
    aruco_map: ArucoMap,
    cfg: VerifierConfig,
) -> tuple[int, np.ndarray]:
    """Score bins by counting pixels that cross the fitted 3D bin surface plane."""
    valid_skus = set(scores.keys())
    counts, intrusion_mask, total = score_plane_intrusions(
        depth_map,
        plane,
        polygons,
        valid_skus,
        cfg.plane_cross_margin,
        cfg.intrusion_pixel_stride,
    )

    for sku, count in counts.items():
        if count > 0:
            scores[sku] += count * cfg.weight_intersection

        marker_id = aruco_map.sku_to_marker_id.get(sku)
        if marker_id is not None and marker_id in baseline_visible and marker_id not in detected_ids:
            scores[sku] += cfg.weight_occlusion

    return total, intrusion_mask


def _legacy_score_frame(
    hand_data: Dict[str, Any],
    polygons: Dict[str, Polygon],
    scores: Dict[str, float],
    baseline_visible: Set[int],
    detected_ids: Set[int],
    aruco_map: ArucoMap,
    cfg: VerifierConfig,
) -> None:
    """Legacy 2D point-in-polygon + occlusion (no palm/depth gates)."""
    ip = hand_data["interaction_point"]
    pt = Point(ip[0], ip[1])
    for sku, poly in polygons.items():
        if sku not in scores:
            continue
        if poly.buffer(cfg.intersection_buffer_px).contains(pt):
            scores[sku] += cfg.weight_intersection
        marker_id = aruco_map.sku_to_marker_id.get(sku)
        if marker_id is not None and marker_id in baseline_visible and marker_id not in detected_ids:
            scores[sku] += cfg.weight_occlusion


def score_pick_segment(
    video_reader: VideoReader,
    seg: StateSegment,
    grid_tracker: GridTracker,
    hand_tracker: HandTracker,  # Kept for backward compatibility but not used in depth mode
    aruco_map: ArucoMap,
    cfg: VerifierConfig,
    valid_skus: Optional[Set[str]] = None,
    visualization_callback: VisualizationFrameCallback = None,
    depth_estimator: Optional[DepthEstimator] = None,
) -> tuple[Dict[str, float], Set[int], int, dict]:
    baseline_frames, voting_frames = _window_frames(seg, cfg)

    baseline_visible: Set[int] = set()
    for f in baseline_frames:
        frame = video_reader.read_frame(f)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det = grid_tracker.detect(gray)
        baseline_visible.update(det.keys())

    if valid_skus is None:
        scores: Dict[str, float] = {b.sku: 0.0 for b in aruco_map.bins}
    else:
        scores: Dict[str, float] = {b.sku: 0.0 for b in aruco_map.bins if b.sku in valid_skus}

    use_depth = cfg.use_depth_gates and depth_estimator is not None
    used = 0
    last_debug: dict = {}
    viz_stop = False

    # Process every Nth frame to reduce computational cost
    frame_stride = max(1, cfg.frame_stride)
    sampled_frames = voting_frames[::frame_stride]

    for f in sampled_frames:
        frame = video_reader.read_frame(f)
        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det = grid_tracker.detect(gray)
        polygons = grid_tracker.build_grid_polygons(det)

        frame_state: Dict[str, Any] = {
            "gate_status": "NO DEPTH MODEL",
            "plane_broken": False,
            "intrusion_pixels": 0,
            "plane_normal": None,
            "plane_offset": None,
            "depth_map": None,
            "intrusion_mask": None,
        }

        if not use_depth:
            # Legacy mode: requires MediaPipe hand tracking
            hand_data = hand_tracker.get_hand_landmarks(frame)
            if hand_data is not None:
                frame_state["gate_status"] = "LEGACY"
                if polygons:
                    used += 1
                    _legacy_score_frame(
                        hand_data, polygons, scores, baseline_visible, set(det.keys()), aruco_map, cfg
                    )
        elif not polygons:
            frame_state["gate_status"] = "NO POLYGONS"
        else:
            # 3D plane-based intrusion detection (no MediaPipe)
            depth_map = depth_estimator.predict(frame)
            centers = marker_centers(det)
            plane = fit_bin_surface_plane(
                depth_map,
                centers,
                cfg.depth_sample_radius_px,
                cfg.min_markers_for_depth_ref,
            )

            frame_state["depth_map"] = depth_map

            if plane is None:
                frame_state["gate_status"] = "PLANE FIT FAILED"
            else:
                frame_state["plane_normal"] = plane.normal.tolist()
                frame_state["plane_offset"] = float(plane.offset)

                intrusion_count, intrusion_mask = _score_plane_intrusion(
                    depth_map,
                    plane,
                    polygons,
                    scores,
                    baseline_visible,
                    set(det.keys()),
                    aruco_map,
                    cfg,
                )

                frame_state["intrusion_pixels"] = intrusion_count
                frame_state["intrusion_mask"] = intrusion_mask
                frame_state["plane_broken"] = intrusion_count > 0
                frame_state["gate_status"] = "PLANE BROKEN" if intrusion_count > 0 else "NO INTRUSION"

                if intrusion_count > 0:
                    used += 1

        # Visualization callback
        if visualization_callback is not None:
            stopping = visualization_callback(f, frame, det, polygons, frame_state)
            if stopping:
                viz_stop = True
                last_debug = {
                    "frame_index": f,
                    "detected_marker_ids": sorted(int(x) for x in det.keys()),
                    "voting_frames": voting_frames,
                    "baseline_frames": baseline_frames,
                    "viz_requested_stop": True,
                    **frame_state,
                }
                break

        if frame_state.get("plane_broken"):
            last_debug = {
                "frame_index": f,
                "detected_marker_ids": sorted(int(x) for x in det.keys()),
                "voting_frames": voting_frames,
                "baseline_frames": baseline_frames,
                "viz_requested_stop": viz_stop,
                **frame_state,
            }

    return scores, baseline_visible, used, last_debug


def predict_for_segments(
    video_reader: VideoReader,
    pick_segments: List[StateSegment],
    grid_tracker: GridTracker,
    hand_tracker: HandTracker,
    aruco_map: ArucoMap,
    picklists: List[List[str]],
    cfg: VerifierConfig,
    visualization_callback: VisualizationFrameCallback = None,
    depth_estimator: Optional[DepthEstimator] = None,
) -> List[PickPrediction]:
    picker = PicklistState(picklists)
    out: List[PickPrediction] = []

    valid_skus: Set[str] = set()
    for block in picklists:
        valid_skus.update(block)
    logger.info("Constraining scoring to %d SKUs from picklists: %s", len(valid_skus), sorted(valid_skus))

    for i, seg in enumerate(pick_segments):
        scores, baseline_visible, used, dbg = score_pick_segment(
            video_reader,
            seg,
            grid_tracker,
            hand_tracker,
            aruco_map,
            cfg,
            valid_skus,
            visualization_callback=visualization_callback,
            depth_estimator=depth_estimator,
        )

        if dbg.get("viz_requested_stop"):
            logger.info("Visualization quit requested; skipping remaining picks for this video")
            break

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        ranked_skus = [sku for sku, score in ranked if score > 0]
        block, chosen, fallback = picker.choose(ranked_skus)

        if chosen is None:
            out.append(
                PickPrediction(
                    pick_index=i,
                    picklist_block=block,
                    segment_frames=(seg.start_frame, seg.end_frame),
                    voting_frames_used=used,
                    predicted_sku=None,
                    predicted_marker_id=None,
                    score=0.0,
                    second_best=None,
                    fallback_to_set_membership=fallback,
                    occluded_baseline_markers=[],
                    reason="no_valid_candidate",
                    debug=dbg,
                )
            )
            continue

        best_score = float(dict(ranked).get(chosen, 0.0))
        second = None
        for sku, score in ranked:
            if sku != chosen:
                second = {"sku": sku, "score": float(score)}
                break

        marker_id = aruco_map.sku_to_marker_id.get(chosen)
        occluded = []
        if marker_id is not None and marker_id in baseline_visible:
            occluded.append(int(marker_id))

        out.append(
            PickPrediction(
                pick_index=i,
                picklist_block=block,
                segment_frames=(seg.start_frame, seg.end_frame),
                voting_frames_used=used,
                predicted_sku=chosen,
                predicted_marker_id=int(marker_id) if marker_id is not None else None,
                score=best_score,
                second_best=second,
                fallback_to_set_membership=fallback,
                occluded_baseline_markers=occluded,
                debug=dbg,
            )
        )

    return out
