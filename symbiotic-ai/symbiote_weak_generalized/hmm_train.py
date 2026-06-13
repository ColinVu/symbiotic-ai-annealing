"""HMM Training Script.

Discovers all matching video/label pairs from the hmm-testing directories,
then trains the HTK HMM state detector using the existing pipeline.

Video files:  hmm-testing/picklist_videos/<name>.<ext>
Label files:  hmm-testing/picklist_labels/<name>.csv

Matching is purely by filename stem — the video extension can be anything
(mp4, mov, avi, …); the label must be a .csv file.  Any .eaf files found in
the label directory are automatically converted to CSV before pair discovery.

Usage (from the symbiotic-ai/ directory)::

    python -m symbiote.hmm_train

    # Custom paths
    python -m symbiote.hmm_train \\
        --video-dir  hmm-testing/picklist_videos \\
        --label-dir  hmm-testing/picklist_labels \\
        --output-dir models/htk \\
        --aruco-config config/aruco_bins.json \\
        --frame-skip 4

    # Override just the output directory
    python -m symbiote.hmm_train --output-dir models/htk_v2
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap: allow running as  python -m symbiote.hmm_train  OR directly
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # symbiote_weak/
_ROOT = _HERE.parent                             # symbiotic-ai/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from symbiote_weak.state_detection.feature_extraction import FeatureExtractor  # noqa: E402
from symbiote_weak.state_detection.training import train_state_detector  # noqa: E402
from symbiote_weak.eaf_to_csv import convert_directory as _convert_eaf_dir  # noqa: E402
from symbiote_weak.hmm_tune import tune as tune_decode_params  # noqa: E402

# ---------------------------------------------------------------------------
# Supported video extensions
# ---------------------------------------------------------------------------
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
_FEATURE_NAMES = [
    "hand_center_x",
    "hand_center_y",
    "velocity_x",
    "velocity_y",
    "accel_x",
    "accel_y",
    "bbox_width",
    "bbox_height",
    "bbox_dwidth",
    "bbox_dheight",
    "orientation_x",
    "orientation_y",
    "orientation_z",
    "object_confidence",
    *[f"hue_hist_{i}" for i in range(8)],
    *[f"sat_hist_{i}" for i in range(4)],
    "aruco_signed_context",
    "aruco_pick_proximity",
    "aruco_place_proximity",
]


def _parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> list[int]:
    return sorted({int(x.strip()) for x in s.split(",") if x.strip()})


def _resolve_feature_mask(output_dir: str, feature_mask: str | None, top_k: int, report_suffix: str = "") -> list[int] | None:
    if feature_mask:
        return _parse_int_list(feature_mask)
    if top_k <= 0:
        return None
    final_dir = Path(output_dir) / "models" / "hmm_final"
    report_path = final_dir / f"feature_reliability{report_suffix}.csv"
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Cannot build top-{top_k} feature mask because report is missing: {report_path}"
        )
    import pandas as pd

    df = pd.read_csv(report_path)
    if "feature_idx" not in df.columns:
        raise ValueError(f"{report_path} is missing required column 'feature_idx'")
    idxs = [int(v) for v in df["feature_idx"].head(top_k).tolist()]
    return sorted(set(idxs))


def _discover_pairs(video_dir: str, label_dir: str) -> list:
    """Return a sorted list of (video_path, label_path) pairs.

    A pair is included only when both a video file AND a matching .csv file
    with the same stem exist in their respective directories.
    """
    vd = Path(video_dir)
    ld = Path(label_dir)

    if not vd.is_dir():
        raise FileNotFoundError(f"Video directory not found: {vd}")
    if not ld.is_dir():
        raise FileNotFoundError(f"Label directory not found: {ld}")

    # Build stem -> video_path map
    video_map = {}
    for f in vd.iterdir():
        if f.suffix.lower() in _VIDEO_EXTS:
            video_map[f.stem] = f

    pairs = []
    unmatched_videos = []
    for stem, video_path in sorted(video_map.items()):
        label_path = ld / (stem + ".csv")
        if label_path.is_file():
            pairs.append((str(video_path), str(label_path)))
        else:
            unmatched_videos.append(stem)

    # Warn about videos with no matching label
    if unmatched_videos:
        print(
            f"[hmm_train] WARNING: {len(unmatched_videos)} video(s) have no "
            f"matching label file and will be skipped:"
        )
        for s in unmatched_videos:
            print(f"  - {s}")

    return pairs


def train(
    video_dir: str,
    label_dir: str,
    output_dir: str,
    aruco_config: str | None = None,
    frame_skip: int = 4,
    blur_threshold: float = 100.0,
    feature_mask: list[int] | None = None,
    coarse_feature_mask: list[int] | None = None,
    interact_feature_mask: list[int] | None = None,
    carry_feature_mask: list[int] | None = None,
    pipeline_mode: str = "two-stage",
    aruco_persistence_frames: int = 0,
    aruco_smoothing_window: int = 1,
    min_segment_seconds: float = 0.15,
    use_sequence_constraint: bool = True,
    verbose: bool = True,
) -> str:
    """Discover pairs and run HTK HMM training.

    Any ``.eaf`` files found in *label_dir* are automatically converted to
    ``.csv`` before pair discovery runs.

    Returns:
        Path to the final trained model directory.
    """
    # Auto-convert any .eaf files present in the label directory
    eaf_files = list(Path(label_dir).glob("*.eaf")) if Path(label_dir).is_dir() else []
    if eaf_files:
        if verbose:
            print(
                f"[hmm_train] Found {len(eaf_files)} .eaf file(s) in label dir — "
                "converting to CSV ..."
            )
        _convert_eaf_dir(label_dir, label_dir, verbose=verbose)

    pairs = _discover_pairs(video_dir, label_dir)

    if len(pairs) == 0:
        raise RuntimeError(
            f"No matched video/label pairs found.\n"
            f"  Video dir: {video_dir}\n"
            f"  Label dir: {label_dir}\n"
            "Ensure videos (mp4/mov/avi/…) and matching .csv labels share the "
            "same filename stem."
        )

    print(f"[hmm_train] Found {len(pairs)} matched pair(s):")
    for vp, lp in pairs:
        print(f"  video : {vp}")
        print(f"  label : {lp}")

    video_paths = [p[0] for p in pairs]
    annotation_paths = [p[1] for p in pairs]

    final_model_dir = train_state_detector(
        video_paths=video_paths,
        annotation_paths=annotation_paths,
        output_dir=output_dir,
        aruco_config_path=aruco_config,
        frame_skip=frame_skip,
        blur_threshold=blur_threshold,
        feature_mask=feature_mask,
        coarse_feature_mask=coarse_feature_mask,
        interact_feature_mask=interact_feature_mask,
        carry_feature_mask=carry_feature_mask,
        pipeline_mode=pipeline_mode,
        aruco_persistence_frames=aruco_persistence_frames,
        aruco_smoothing_window=aruco_smoothing_window,
        min_segment_seconds=min_segment_seconds,
        verbose=verbose,
    )

    if feature_mask is not None:
        meta_path = Path(final_model_dir) / "feature_mask.json"
        payload = {
            "selected_indices": feature_mask,
            "selected_features": [
                _FEATURE_NAMES[i] if 0 <= i < len(_FEATURE_NAMES) else f"feature_{i}"
                for i in feature_mask
            ],
            "mask_mode": "zero_out_unselected",
        }
        with open(meta_path, "w") as f:
            json.dump(payload, f, indent=2)
        if verbose:
            print(f"[hmm_train] Wrote feature mask: {meta_path}")
    stage_masks = {
        "coarse": coarse_feature_mask,
        "interact": interact_feature_mask,
        "carry": carry_feature_mask,
    }
    for stage_name, mask in stage_masks.items():
        if mask is None:
            continue
        stage_path = Path(final_model_dir) / f"feature_mask_{stage_name}.json"
        payload = {
            "selected_indices": mask,
            "selected_features": [
                _FEATURE_NAMES[i] if 0 <= i < len(_FEATURE_NAMES) else f"feature_{i}"
                for i in mask
            ],
            "mask_mode": "zero_out_unselected",
            "stage": stage_name,
        }
        with open(stage_path, "w") as f:
            json.dump(payload, f, indent=2)
        if verbose:
            print(f"[hmm_train] Wrote {stage_name} feature mask: {stage_path}")

    pipeline_cfg_path = Path(final_model_dir) / "pipeline_config.json"
    if pipeline_cfg_path.is_file():
        try:
            with open(pipeline_cfg_path, "r") as f:
                pipeline_cfg = json.load(f)
        except Exception:
            pipeline_cfg = {}
    else:
        pipeline_cfg = {}
    pipeline_cfg["feature_mask_file"] = "feature_mask.json" if feature_mask is not None else None
    pipeline_cfg["coarse_feature_mask_file"] = "feature_mask_coarse.json" if coarse_feature_mask is not None else None
    pipeline_cfg["interact_feature_mask_file"] = "feature_mask_interact.json" if interact_feature_mask is not None else None
    pipeline_cfg["carry_feature_mask_file"] = "feature_mask_carry.json" if carry_feature_mask is not None else None
    pipeline_cfg["use_sequence_constraint"] = bool(use_sequence_constraint)
    pipeline_cfg["min_segment_seconds"] = float(min_segment_seconds)
    pipeline_cfg["aruco_persistence_frames"] = int(aruco_persistence_frames)
    pipeline_cfg["aruco_smoothing_window"] = int(aruco_smoothing_window)
    with open(pipeline_cfg_path, "w") as f:
        json.dump(pipeline_cfg, f, indent=2)

    print(f"\n[hmm_train] Training complete.")
    print(f"[hmm_train] Model saved to: {final_model_dir}")
    return final_model_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m symbiote.hmm_train",
        description=(
            "Train the HTK HMM state detector from matched video/label pairs "
            "in the hmm-testing directories."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--video-dir",
        default=str(_ROOT / "hmm-testing" / "picklist_videos"),
        help="Directory containing training videos (default: hmm-testing/picklist_videos)",
    )
    p.add_argument(
        "--label-dir",
        default=str(_ROOT / "hmm-testing" / "picklist_labels"),
        help="Directory containing annotation CSVs/EAFs (default: hmm-testing/picklist_labels)",
    )
    p.add_argument(
        "--output-dir",
        default=str(_ROOT / "models" / "htk"),
        help="Output directory for the trained HTK model (default: models/htk)",
    )
    p.add_argument(
        "--aruco-config",
        default=str(_ROOT / "config" / "aruco_bins.json"),
        help="Path to aruco_bins.json (default: config/aruco_bins.json)",
    )
    p.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="Process every Nth frame during feature extraction (default: 4)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help="Laplacian blur detection threshold (default: 100.0)",
    )
    p.add_argument("--aruco-persistence-frames", type=int, default=0, help="Persist non-zero ARUCO context for N extracted frames.")
    p.add_argument("--aruco-smoothing-window", type=int, default=1, help="Temporal smoothing window for ARUCO context channel.")
    p.add_argument("--min-segment-seconds", type=float, default=0.15, help="Boundary cleanup minimum segment duration in seconds.")
    p.add_argument(
        "--no-sequence-constraint",
        action="store_true",
        help="Disable label-derived expected-sequence constraints for training metadata/tune/infer (default: enabled).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    p.add_argument(
        "--tune-decode",
        action="store_true",
        help="Run HVite decode parameter tuning immediately after training.",
    )
    p.add_argument(
        "--tune-penalties",
        default="-3,-1,0,1,2,3",
        help="Comma-separated sweep values for HVite -p when --tune-decode is set.",
    )
    p.add_argument(
        "--tune-scales",
        default="0.5,1,2,5,8",
        help="Comma-separated sweep values for HVite -s when --tune-decode is set.",
    )
    p.add_argument(
        "--tune-free-order",
        action="store_true",
        help="Use unconstrained order in tuning (default is strict cycle).",
    )
    p.add_argument(
        "--feature-mask",
        default=None,
        help="Comma-separated feature indices to keep active; others are zeroed.",
    )
    p.add_argument(
        "--feature-top-k",
        type=int,
        default=0,
        help="Auto-build mask from top-K rows in models/hmm_final/feature_reliability.csv.",
    )
    p.add_argument("--coarse-feature-mask", default=None, help="Mask indices for coarse INTERACT/CARRY stage.")
    p.add_argument("--coarse-feature-top-k", type=int, default=0, help="Top-K from feature_reliability_coarse.csv.")
    p.add_argument("--interact-feature-mask", default=None, help="Mask indices for interact PICK/PLACE stage.")
    p.add_argument("--interact-feature-top-k", type=int, default=0, help="Top-K from feature_reliability_interact.csv.")
    p.add_argument("--carry-feature-mask", default=None, help="Mask indices for carry CARRY_WITH/CARRY_EMPTY stage.")
    p.add_argument("--carry-feature-top-k", type=int, default=0, help="Top-K from feature_reliability_carry.csv.")
    p.add_argument(
        "--pipeline",
        choices=["two-stage", "legacy"],
        default="two-stage",
        help="Decoding/training pipeline mode (default: two-stage).",
    )
    p.add_argument(
        "--legacy",
        action="store_true",
        help="Shorthand for --pipeline legacy.",
    )
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    aruco_config = args.aruco_config if os.path.isfile(args.aruco_config) else None
    if args.aruco_config and aruco_config is None:
        print(f"[hmm_train] WARNING: aruco config not found at {args.aruco_config}, "
              "proceeding without ARUCO bin context.")

    try:
        pipeline_mode = "legacy" if args.legacy else args.pipeline
        resolved_mask = _resolve_feature_mask(args.output_dir, args.feature_mask, args.feature_top_k)
        resolved_coarse_mask = _resolve_feature_mask(
            args.output_dir, args.coarse_feature_mask, args.coarse_feature_top_k, report_suffix="_coarse"
        )
        resolved_interact_mask = _resolve_feature_mask(
            args.output_dir, args.interact_feature_mask, args.interact_feature_top_k, report_suffix="_interact"
        )
        resolved_carry_mask = _resolve_feature_mask(
            args.output_dir, args.carry_feature_mask, args.carry_feature_top_k, report_suffix="_carry"
        )
        _fd = FeatureExtractor.FEATURE_DIM
        if resolved_mask:
            if min(resolved_mask) < 0 or max(resolved_mask) >= _fd:
                raise ValueError(f"Feature indices must be in [0, {_fd - 1}].")
        for stage_name, stage_mask in [
            ("coarse", resolved_coarse_mask),
            ("interact", resolved_interact_mask),
            ("carry", resolved_carry_mask),
        ]:
            if stage_mask:
                if min(stage_mask) < 0 or max(stage_mask) >= _fd:
                    raise ValueError(f"{stage_name} feature indices must be in [0, {_fd - 1}].")
        if not args.quiet and resolved_mask is not None:
            print(f"[hmm_train] Using feature mask: {resolved_mask}")
        if not args.quiet and resolved_coarse_mask is not None:
            print(f"[hmm_train] Using coarse mask: {resolved_coarse_mask}")
        if not args.quiet and resolved_interact_mask is not None:
            print(f"[hmm_train] Using interact mask: {resolved_interact_mask}")
        if not args.quiet and resolved_carry_mask is not None:
            print(f"[hmm_train] Using carry mask: {resolved_carry_mask}")

        train(
            video_dir=args.video_dir,
            label_dir=args.label_dir,
            output_dir=args.output_dir,
            aruco_config=aruco_config,
            frame_skip=args.frame_skip,
            blur_threshold=args.threshold,
            feature_mask=resolved_mask,
            coarse_feature_mask=resolved_coarse_mask,
            interact_feature_mask=resolved_interact_mask,
            carry_feature_mask=resolved_carry_mask,
            pipeline_mode=pipeline_mode,
            aruco_persistence_frames=args.aruco_persistence_frames,
            aruco_smoothing_window=args.aruco_smoothing_window,
            min_segment_seconds=args.min_segment_seconds,
            use_sequence_constraint=not args.no_sequence_constraint,
            verbose=not args.quiet,
        )
        if args.tune_decode:
            print("\n[hmm_train] Starting decode tuning (cache-aware) ...")
            tune_decode_params(
                video_dir=args.video_dir,
                label_dir=args.label_dir,
                model_dir=args.output_dir,
                aruco_config=aruco_config,
                frame_skip=args.frame_skip,
                blur_threshold=args.threshold,
                penalties=_parse_float_list(args.tune_penalties),
                scales=_parse_float_list(args.tune_scales),
                strict_cycle=not args.tune_free_order,
                feature_mask=resolved_mask,
                coarse_feature_mask=resolved_coarse_mask,
                interact_feature_mask=resolved_interact_mask,
                carry_feature_mask=resolved_carry_mask,
                pipeline_mode=pipeline_mode,
                min_segment_seconds=args.min_segment_seconds,
                aruco_persistence_frames=args.aruco_persistence_frames,
                aruco_smoothing_window=args.aruco_smoothing_window,
                use_sequence_constraint=not args.no_sequence_constraint,
                lock_sequence_constraint=True,
                verbose=not args.quiet,
            )
    except Exception as exc:
        print(f"\n[hmm_train] ERROR: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
