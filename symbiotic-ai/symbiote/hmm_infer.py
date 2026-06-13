"""HMM Inference Script.

Runs the trained HTK HMM state detector on a new input video and produces:

1. A CSV file with predicted state timestamps::

       timestamp_start,timestamp_end,state
       0.00,1.73,CARRY_EMPTY
       1.73,3.12,PICK
       3.12,8.40,CARRY_WITH
       8.40,10.05,PLACE
       ...

2. An annotated copy of the original video with the current predicted state
   overlaid on every frame as a coloured banner.

   State colour coding:
     PICK         → green
     CARRY_WITH   → yellow
     PLACE        → red
     CARRY_EMPTY  → grey

Usage (from the symbiotic-ai/ directory)::

    python -m symbiote.hmm_infer \\
        --video path/to/input.mp4 \\
        --model-dir models/htk \\
        --output-csv results/predicted_states.csv \\
        --output-video results/input_annotated.mp4 \\
        --aruco-config config/aruco_bins.json

    # Minimal (CSV and annotated video auto-named from input)
    python -m symbiote.hmm_infer --video path/to/input.mp4

    # Skip annotated video output
    python -m symbiote.hmm_infer --video path/to/input.mp4 --no-video
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # symbiote/
_ROOT = _HERE.parent                             # symbiotic-ai/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from symbiote.state_detection.aruco_detection import ArucoDetector  # noqa: E402
from symbiote.state_detection.config import DEFAULT_HTK_CONFIG       # noqa: E402
from symbiote.state_detection.feature_extraction import FeatureExtractor  # noqa: E402
from symbiote.state_detection.htk_interface import HTKStateDetector  # noqa: E402
from symbiote.state_detection.two_stage import (  # noqa: E402
    CARRY_STATES,
    COARSE_STATES,
    INTERACT_STATES,
    apply_feature_mask,
    decode_subtype_with_runs,
    frame_labels_to_segments,
    segments_to_frame_labels,
)

# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------
_STATE_COLORS = {
    "PICK":         (0,   220,  0),    # green  (BGR)
    "CARRY_WITH":   (0,   220, 220),   # yellow (BGR)
    "PLACE":        (0,    30, 220),   # red    (BGR)
    "CARRY_EMPTY":  (140, 140, 140),   # grey   (BGR)
}
_DEFAULT_COLOR = (200, 200, 200)
_BANNER_HEIGHT = 60   # pixels


def _parse_int_list(s: str) -> list[int]:
    return sorted({int(x.strip()) for x in s.split(",") if x.strip()})


def _state_at(timestamp: float, segments: pd.DataFrame) -> Optional[str]:
    """Return the state label at *timestamp*, or None if outside all segments."""
    for _, row in segments.iterrows():
        if row["timestamp_start"] <= timestamp <= row["timestamp_end"]:
            return row["state"]
    return None


def _draw_state_banner(frame: np.ndarray, state: Optional[str]) -> np.ndarray:
    """Draw a solid coloured banner at the top of *frame* showing *state*."""
    out = frame.copy()
    h, w = out.shape[:2]
    label = state if state else "UNKNOWN"
    color = _STATE_COLORS.get(label, _DEFAULT_COLOR)

    # Filled banner
    cv2.rectangle(out, (0, 0), (w, _BANNER_HEIGHT), color, thickness=-1)

    # State text centred in banner
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    tx = (w - tw) // 2
    ty = (_BANNER_HEIGHT + th) // 2

    # Shadow for readability
    cv2.putText(out, label, (tx + 2, ty + 2), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(out, label, (tx, ty), font, font_scale, (255, 255, 255), thickness)

    return out


def run_inference(
    video_path: str,
    model_dir: str,
    output_csv: str,
    output_video: Optional[str],
    aruco_config: Optional[str] = None,
    frame_skip: int = 4,
    blur_threshold: float = 100.0,
    word_penalty: Optional[float] = None,
    grammar_scale: Optional[float] = None,
    strict_cycle: bool = True,
    feature_mask: Optional[list[int]] = None,
    coarse_feature_mask: Optional[list[int]] = None,
    interact_feature_mask: Optional[list[int]] = None,
    carry_feature_mask: Optional[list[int]] = None,
    pipeline_mode: str = "two-stage",
    verbose: bool = True,
) -> dict:
    """Run HMM inference on *video_path*.

    Args:
        video_path:      Input video.
        model_dir:       Trained HTK model directory (output of hmm_train.py).
        output_csv:      Where to write the predicted-state CSV.
        output_video:    Where to write the annotated video.  Pass ``None`` to
                         skip video rendering.
        aruco_config:    Path to ``aruco_bins.json`` (optional but recommended).
        frame_skip:      Feature-extraction frame stride.
        blur_threshold:  Laplacian blur filter threshold.
        verbose:         Print progress.

    Returns:
        dict with keys ``csv``, ``video`` (paths written) and ``segments``
        (the decoded DataFrame).
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Resolve model directory — accept either the top-level output_dir or the
    # inner models/hmm_final directory directly.
    resolved_model_dir = model_dir
    candidate_final = os.path.join(model_dir, "models", "hmm_final")
    if os.path.isdir(candidate_final):
        resolved_model_dir = model_dir  # HTKStateDetector resolves internally

    if not os.path.isdir(resolved_model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Optional tuned decode params auto-loaded from model dir.
    final_dir = os.path.join(resolved_model_dir, "models", "hmm_final")
    if not os.path.isdir(final_dir):
        final_dir = resolved_model_dir
    tuned_params_path = os.path.join(final_dir, "infer_params.json")
    if os.path.isfile(tuned_params_path):
        try:
            with open(tuned_params_path, "r") as f:
                tuned = json.load(f)
            if word_penalty is None:
                word_penalty = float(tuned.get("word_penalty", 0.0))
            if grammar_scale is None:
                grammar_scale = float(tuned.get("grammar_scale", 5.0))
            strict_cycle = bool(tuned.get("strict_cycle", strict_cycle))
            if pipeline_mode == "two-stage":
                pipeline_mode = str(tuned.get("pipeline_mode", pipeline_mode))
            if verbose:
                print(f"[hmm_infer] Using tuned decode params from: {tuned_params_path}")
        except Exception:
            if verbose:
                print(f"[hmm_infer] WARNING: failed to parse {tuned_params_path}; using defaults/CLI")

    # Optional feature masks auto-loaded from model dir.
    def _load_mask(path: str) -> Optional[list[int]]:
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as f:
                mask_json = json.load(f)
            return [int(x) for x in mask_json.get("selected_indices", [])]
        except Exception:
            return None

    if feature_mask is None:
        feature_mask = _load_mask(os.path.join(final_dir, "feature_mask.json"))
    if coarse_feature_mask is None:
        coarse_feature_mask = _load_mask(os.path.join(final_dir, "feature_mask_coarse.json"))
    if interact_feature_mask is None:
        interact_feature_mask = _load_mask(os.path.join(final_dir, "feature_mask_interact.json"))
    if carry_feature_mask is None:
        carry_feature_mask = _load_mask(os.path.join(final_dir, "feature_mask_carry.json"))

    pipeline_cfg_path = os.path.join(final_dir, "pipeline_config.json")
    if os.path.isfile(pipeline_cfg_path):
        try:
            with open(pipeline_cfg_path, "r") as f:
                pipeline_cfg = json.load(f)
            if pipeline_mode == "two-stage":
                pipeline_mode = str(pipeline_cfg.get("pipeline_mode", pipeline_mode))
        except Exception:
            if verbose:
                print(f"[hmm_infer] WARNING: failed to parse {pipeline_cfg_path}; using CLI/default pipeline mode")

    if word_penalty is None:
        word_penalty = 0.0
    if grammar_scale is None:
        grammar_scale = 5.0

    # ------------------------------------------------------------------
    # 1. Extract features
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[hmm_infer] Input video   : {video_path}")
        print(f"[hmm_infer] Model dir     : {model_dir}")

    aruco_detector = ArucoDetector(
        aruco_dict_type=DEFAULT_HTK_CONFIG.aruco_dict_type,
        distance_decay=DEFAULT_HTK_CONFIG.aruco_distance_decay,
    )
    if aruco_config and os.path.isfile(aruco_config):
        aruco_detector.load_bin_config(aruco_config)
        if verbose:
            print(f"[hmm_infer] ARUCO config  : {aruco_config}")
    elif verbose:
        print("[hmm_infer] ARUCO config  : none (bin context will be 0)")

    extractor = FeatureExtractor(aruco_detector=aruco_detector, feature_mask=feature_mask)
    features, frame_numbers, fps = extractor.extract_video_features(
        video_path,
        frame_skip=frame_skip,
        blur_threshold=blur_threshold,
        verbose=verbose,
    )

    if features.shape[0] == 0:
        raise RuntimeError(
            "No features could be extracted from the video. "
            "Check that a hand is visible and the video is not blurry."
        )

    # ------------------------------------------------------------------
    # 2. Decode with HTK Viterbi
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[hmm_infer] Pipeline mode : {pipeline_mode}")
        print("[hmm_infer] Running Viterbi decoding ...")

    if pipeline_mode == "legacy":
        htk_detector = HTKStateDetector(resolved_model_dir)
        segments: pd.DataFrame = htk_detector.decode(
            features,
            fps,
            frame_numbers=frame_numbers,
            verbose=verbose,
            word_penalty=word_penalty,
            grammar_scale=grammar_scale,
            strict_cycle=strict_cycle,
        )
    else:
        coarse_model_dir = os.path.join(resolved_model_dir, "coarse_model")
        if not os.path.isdir(coarse_model_dir):
            raise RuntimeError(
                "two-stage pipeline selected but coarse model is missing. "
                "Retrain with `python -m symbiote.hmm_train` (default two-stage) "
                "or run inference with --legacy."
            )
        coarse_detector = HTKStateDetector(coarse_model_dir, state_labels=COARSE_STATES)
        interact_model_dir = os.path.join(resolved_model_dir, "interact_model")
        carry_model_dir = os.path.join(resolved_model_dir, "carry_model")
        interact_detector = (
            HTKStateDetector(interact_model_dir, state_labels=INTERACT_STATES)
            if os.path.isdir(interact_model_dir)
            else None
        )
        carry_detector = (
            HTKStateDetector(carry_model_dir, state_labels=CARRY_STATES)
            if os.path.isdir(carry_model_dir)
            else None
        )
        if verbose and interact_detector is None:
            print("[hmm_infer] WARNING: interact_model missing; using fallback for INTERACT runs.")
        if verbose and carry_detector is None:
            print("[hmm_infer] WARNING: carry_model missing; using fallback for CARRY runs.")
        coarse_segments = coarse_detector.decode(
            apply_feature_mask(features, coarse_feature_mask if coarse_feature_mask is not None else feature_mask),
            fps,
            frame_numbers=frame_numbers,
            verbose=verbose,
            word_penalty=word_penalty,
            grammar_scale=grammar_scale,
            strict_cycle=strict_cycle,
        )
        coarse_frame_labels = segments_to_frame_labels(coarse_segments, frame_numbers, fps)
        fine_frame_labels = decode_subtype_with_runs(
            coarse_frame_labels,
            features,
            fps,
            frame_numbers,
            interact_detector=interact_detector,
            carry_detector=carry_detector,
            strict_cycle=False,
            word_penalty=word_penalty,
            grammar_scale=grammar_scale,
            interact_mask=interact_feature_mask if interact_feature_mask is not None else feature_mask,
            carry_mask=carry_feature_mask if carry_feature_mask is not None else feature_mask,
        )
        segments = frame_labels_to_segments(fine_frame_labels, frame_numbers, fps)

    if segments.empty:
        raise RuntimeError(
            "HVite produced no output. "
            "Check that the model is trained and the HTK binaries are on PATH."
        )

    # ------------------------------------------------------------------
    # 3. Save CSV
    # ------------------------------------------------------------------
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    segments.to_csv(output_csv, index=False)
    if verbose:
        print(f"\n[hmm_infer] Predicted states ({len(segments)} segments):")
        print(segments.to_string(index=False))
        print(f"\n[hmm_infer] CSV saved to: {output_csv}")

    result = {"csv": output_csv, "video": None, "segments": segments}

    # ------------------------------------------------------------------
    # 4. Render annotated video
    # ------------------------------------------------------------------
    if output_video is None:
        return result

    Path(output_video).parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[hmm_infer] WARNING: Could not reopen video for annotation: {video_path}")
        return result

    vid_fps    = cap.get(cv2.CAP_PROP_FPS) or fps
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_fr   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, vid_fps, (width, height))

    frame_idx = 0
    if verbose:
        print(f"[hmm_infer] Rendering annotated video ({total_fr} frames) ...")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1

        timestamp = frame_idx / vid_fps if vid_fps > 0 else 0.0
        state = _state_at(timestamp, segments)
        annotated = _draw_state_banner(frame_bgr, state)
        writer.write(annotated)

        if verbose and frame_idx % 300 == 0:
            print(f"  ... {frame_idx}/{total_fr} frames rendered")

    cap.release()
    writer.release()

    result["video"] = output_video
    if verbose:
        print(f"[hmm_infer] Annotated video saved to: {output_video}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m symbiote.hmm_infer",
        description=(
            "Run the trained HTK HMM state detector on a new video and output "
            "predicted state timestamps (CSV) plus an annotated copy of the video."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--video",
        required=True,
        help="Input video file to run inference on",
    )
    p.add_argument(
        "--model-dir",
        default=str(_ROOT / "models" / "htk"),
        help="Trained HTK model directory (output of hmm_train.py). "
             "Default: models/htk",
    )
    p.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV file path. Default: <video_stem>_states.csv next to the input video.",
    )
    p.add_argument(
        "--output-video",
        default=None,
        help="Output annotated video path. Default: <video_stem>_annotated.mp4 "
             "next to the input video. Pass --no-video to suppress.",
    )
    p.add_argument(
        "--no-video",
        action="store_true",
        help="Skip annotated video output (CSV only).",
    )
    p.add_argument(
        "--aruco-config",
        default=str(_ROOT / "config" / "aruco_bins.json"),
        help="Path to aruco_bins.json. Default: config/aruco_bins.json",
    )
    p.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="Feature extraction frame stride (default: 4)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help="Laplacian blur detection threshold (default: 100.0)",
    )
    p.add_argument(
        "--word-penalty",
        type=float,
        default=None,
        help="HVite word insertion penalty -p (default: auto from infer_params.json or 0.0)",
    )
    p.add_argument(
        "--grammar-scale",
        type=float,
        default=None,
        help="HVite grammar scale -s (default: auto from infer_params.json or 5.0)",
    )
    p.add_argument(
        "--free-order",
        action="store_true",
        help="Disable strict cycle grammar (allow any state order).",
    )
    p.add_argument(
        "--feature-mask",
        default=None,
        help="Comma-separated feature indices to keep active; overrides model feature_mask.json.",
    )
    p.add_argument("--coarse-feature-mask", default=None, help="Override coarse stage mask indices.")
    p.add_argument("--interact-feature-mask", default=None, help="Override interact stage mask indices.")
    p.add_argument("--carry-feature-mask", default=None, help="Override carry stage mask indices.")
    p.add_argument(
        "--pipeline",
        choices=["two-stage", "legacy"],
        default="two-stage",
        help="Inference pipeline mode (default: two-stage).",
    )
    p.add_argument(
        "--legacy",
        action="store_true",
        help="Shorthand for --pipeline legacy.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    video_path = args.video
    if not os.path.isabs(video_path):
        video_path = os.path.normpath(os.path.join(os.getcwd(), video_path))

    # Default CSV: alongside input video
    if args.output_csv is None:
        stem = Path(video_path).stem
        output_csv = str(Path(video_path).parent / f"{stem}_states.csv")
    else:
        output_csv = args.output_csv

    # Default annotated video: alongside input video
    if args.no_video:
        output_video = None
    elif args.output_video is not None:
        output_video = args.output_video
    else:
        stem = Path(video_path).stem
        output_video = str(Path(video_path).parent / f"{stem}_annotated.mp4")

    aruco_config = args.aruco_config if os.path.isfile(args.aruco_config) else None

    try:
        pipeline_mode = "legacy" if args.legacy else args.pipeline
        run_inference(
            video_path=video_path,
            model_dir=args.model_dir,
            output_csv=output_csv,
            output_video=output_video,
            aruco_config=aruco_config,
            frame_skip=args.frame_skip,
            blur_threshold=args.threshold,
            word_penalty=args.word_penalty,
            grammar_scale=args.grammar_scale,
            strict_cycle=not args.free_order,
            feature_mask=None if not args.feature_mask else _parse_int_list(args.feature_mask),
            coarse_feature_mask=None if not args.coarse_feature_mask else _parse_int_list(args.coarse_feature_mask),
            interact_feature_mask=None if not args.interact_feature_mask else _parse_int_list(args.interact_feature_mask),
            carry_feature_mask=None if not args.carry_feature_mask else _parse_int_list(args.carry_feature_mask),
            pipeline_mode=pipeline_mode,
            verbose=not args.quiet,
        )
    except Exception as exc:
        print(f"\n[hmm_infer] ERROR: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
