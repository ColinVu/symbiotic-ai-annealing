"""Tunable parameters for bin picking verification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VerifierConfig:
    """Processing and scoring hyperparameters."""

    target_height: int = 1080
    """Downscale video so height matches this (width scales proportionally)."""

    baseline_fraction: float = 0.30
    """DEPRECATED: With plane-breaking detection, entire segment is now used for baseline + scoring."""

    voting_fraction: float = 0.30
    """DEPRECATED: With plane-breaking detection, entire segment is now used for baseline + scoring."""

    intersection_buffer_px: float = 8.0
    """Expand bin polygons by this amount (Shapely buffer) for interaction tests."""

    weight_intersection: float = 1.0
    """Score added per voting frame when interaction point is inside a bin zone."""

    weight_occlusion: float = 0.5
    """Score added when baseline-visible marker for that bin is missing in the frame."""

    min_pick_segment_frames: int = 5
    """Minimum expected frames in a PICK segment (informational only with new full-segment scoring)."""

    homography_ransac_reproj_threshold: float = 5.0
    polygon_half_extent_factor: float = 0.45
    """Half-size of ideal-space bin square = mean_grid_spacing * this factor."""

    plane_breach_z_threshold: float = -0.05
    """DEPRECATED: Old MediaPipe z heuristic; replaced by depth-based plane break."""

    # Palm orientation gate (cross product)
    palm_down_normal_z_sign: float = 1.0
    """Multiply palm_normal_z by this; flip to -1.0 if gate polarity is wrong on your videos."""

    palm_down_min_abs_z: float = 0.002
    """Minimum |palm_normal_z| after sign flip to count as palm facing bins."""

    # Monocular depth (DepthAnything V2)
    depth_model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    depth_device: str = "auto"
    depth_infer_size: int = 518
    depth_sample_radius_px: int = 6
    min_markers_for_depth_ref: int = 3
    plane_cross_margin: float = 0.015
    """Signed 3D distance past fitted bin plane to count as intrusion (unprojected camera coords)."""

    plane_break_depth_margin: float = 0.02
    """DEPRECATED: replaced by plane_cross_margin (true 3D plane crossing)."""

    frame_stride: int = 3
    """Process every Nth frame in each PICK segment."""

    intrusion_pixel_stride: int = 2
    """Subsample pixels inside bin polygons when testing plane crossing."""

    use_depth_gates: bool = True
    """When False, legacy 2D point-in-polygon scoring without palm/depth gates."""

    viz_window_width: int = 1280
    """Resize displayed frame width for real-time viz (aspect preserved)."""

    viz_show_all_landmarks: bool = True
    viz_show_depth_inset: bool = True
    """Show normalized depth map inset in --visualized mode."""
