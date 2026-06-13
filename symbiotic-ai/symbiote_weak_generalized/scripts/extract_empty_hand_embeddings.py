"""
Extract CLIP embeddings from CARRY_EMPTY (``m``) segments for HandNeutralizer training.

Mirrors the train pipeline: downscale to 1080p when needed, MediaPipe hand crop,
blur check on the segmented hand, then frozen CLIP ``get_image_features``.

For each video with a compact label CSV under ``--labels-dir``, finds every
``m`` interval (same 1-based frame convention as ``video_processor``), **skips
the first and last** CARRY_EMPTY segment of that video, and for each remaining
segment samples the **middle** frame; if that frame fails (no hand / blurry),
scans **forward** within the segment until one works.

Writes one ``.npy`` per sample (shape ``(D,)``), compatible with
``training.hand_neutralizer.HandNeutralizer``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor
import mediapipe as mp

from ..core.config import MODEL
from ..lib.hand_detection import segment_hand
from ..preprocessing.blur_detection import is_blurry
from ..state_detection.compact_timeline import carry_empty_pipeline_frame_intervals_1based


def _repo_root() -> Path:
    # symbiotic-ai/symbiote_weak_generalized/scripts/this_file.py -> symbiotic-ai
    return Path(__file__).resolve().parents[2]


def _list_videos(videos_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for pattern in ("*.mp4", "*.MP4", "*.m4v", "*.M4V"):
        paths.extend(videos_dir.glob(pattern))
    return sorted(set(paths), key=lambda p: p.name.lower())


def _downscale_rgb_if_needed(image_rgb: np.ndarray) -> np.ndarray:
    h_orig, w_orig = image_rgb.shape[:2]
    if w_orig > 1920 or h_orig > 1080:
        scale = min(1920 / w_orig, 1080 / h_orig)
        return cv2.resize(
            image_rgb,
            (int(w_orig * scale), int(h_orig * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return image_rgb


def _try_embed_frame(
    cap: cv2.VideoCapture,
    frame_1based: int,
    *,
    clip_model: torch.nn.Module,
    processor: AutoProcessor,
    hands_detector,
    blur_threshold: float,
    device: str,
) -> Optional[np.ndarray]:
    """
    Seek to 1-based frame index (same as ``video_processor``), return CLIP vector
    ``(D,)`` or None if hand missing / empty crop / blurry.
    """
    if frame_1based < 1:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_1based - 1)
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb = _downscale_rgb_if_needed(image_rgb)
    segmented = segment_hand(image_rgb, hands_detector)
    if segmented is None or segmented.size == 0:
        return None
    if is_blurry(segmented, blur_threshold):
        return None
    try:
        inputs = processor(images=[segmented], return_tensors="pt").to(device)
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs)
        return emb.cpu().numpy()[0].astype(np.float64)
    except Exception:
        return None


def _embed_carry_empty_segment(
    cap: cv2.VideoCapture,
    start: int,
    end: int,
    *,
    clip_model: torch.nn.Module,
    processor: AutoProcessor,
    hands_detector,
    blur_threshold: float,
    device: str,
) -> Optional[np.ndarray]:
    """Middle frame first, then forward until ``end`` inclusive."""
    mid = (start + end) // 2
    for f in range(mid, end + 1):
        emb = _try_embed_frame(
            cap,
            f,
            clip_model=clip_model,
            processor=processor,
            hands_detector=hands_detector,
            blur_threshold=blur_threshold,
            device=device,
        )
        if emb is not None:
            return emb
    return None


def run_extraction(
    videos_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    *,
    compact_frame_indexing: str,
    blur_threshold: float,
    verbose: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = AutoModel.from_pretrained(MODEL)
    clip_model.eval()
    if device == "cuda":
        clip_model = clip_model.to(device)
    processor = AutoProcessor.from_pretrained(MODEL)

    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.3,
        max_num_hands=2,
    )

    n_written = 0
    try:
        for video_path in _list_videos(videos_dir):
            stem = video_path.stem
            csv_path = labels_dir / f"{stem}.csv"
            if not csv_path.is_file():
                if verbose:
                    print(f"[skip] no label CSV for {stem}: {csv_path}")
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                if verbose:
                    print(f"[skip] cannot open video: {video_path}")
                continue
            n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            try:
                intervals = carry_empty_pipeline_frame_intervals_1based(
                    csv_path,
                    n_total,
                    frame_indexing=compact_frame_indexing,
                )
            except Exception as e:
                if verbose:
                    print(f"[skip] {stem}: failed to parse labels: {e}")
                cap.release()
                continue

            if len(intervals) < 3:
                if verbose:
                    print(
                        f"[skip] {stem}: need >=3 CARRY_EMPTY (m) segments to skip first/last; "
                        f"found {len(intervals)}"
                    )
                cap.release()
                continue

            # Skip first and last CARRY_EMPTY segment of this video (indices 0 and L-1).
            for seg_i in range(1, len(intervals) - 1):
                start, end = intervals[seg_i]
                emb = _embed_carry_empty_segment(
                    cap,
                    start,
                    end,
                    clip_model=clip_model,
                    processor=processor,
                    hands_detector=hands_detector,
                    blur_threshold=blur_threshold,
                    device=device,
                )
                if emb is None:
                    if verbose:
                        print(f"[fail] {stem} m-seg#{seg_i} frames [{start},{end}] (no usable frame)")
                    continue
                out_path = output_dir / f"{stem}_carry_empty_{seg_i}.npy"
                np.save(out_path, emb)
                n_written += 1
                if verbose:
                    print(f"[ok]   {out_path.name}  dim={emb.shape[0]}")
            cap.release()
    finally:
        hands_detector.close()

    return n_written


def main(argv: Optional[List[str]] = None) -> int:
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Extract empty-hand CLIP embeddings from CARRY_EMPTY (m) segments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        default=str(root / "hmm-testing" / "picklist_videos"),
        help="Directory of training videos (.mp4)",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default=str(root / "hmm-testing" / "picklist_labels"),
        help="Directory of compact timeline CSVs ({stem}.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(root / "hmm-testing" / "hand_embeddings"),
        help="Where to write one .npy per extracted embedding",
    )
    parser.add_argument(
        "--compact-frame-indexing",
        type=str,
        default="opencv0",
        choices=["opencv0", "pipeline1"],
        help="Must match how label CSV frame column was authored",
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=50.0,
        help="Laplacian variance threshold on segmented hand (same as train)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less console output",
    )
    args = parser.parse_args(argv)

    videos_dir = Path(args.videos_dir).resolve()
    labels_dir = Path(args.labels_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    verbose = not bool(args.quiet)

    if not videos_dir.is_dir():
        print(f"Error: videos directory not found: {videos_dir}", file=sys.stderr)
        return 1
    if not labels_dir.is_dir():
        print(f"Error: labels directory not found: {labels_dir}", file=sys.stderr)
        return 1

    if verbose:
        print("=" * 60)
        print("EMPTY-HAND EMBEDDING EXTRACTION (CARRY_EMPTY / m)")
        print("=" * 60)
        print(f"Videos:  {videos_dir}")
        print(f"Labels: {labels_dir}")
        print(f"Out:    {output_dir}")
        print(f"CLIP:   {MODEL}")
        print(f"Frame indexing: {args.compact_frame_indexing}")
        print(f"Blur threshold:   {args.blur_threshold}")

    n = run_extraction(
        videos_dir,
        labels_dir,
        output_dir,
        compact_frame_indexing=args.compact_frame_indexing,
        blur_threshold=float(args.blur_threshold),
        verbose=verbose,
    )
    if verbose:
        print("=" * 60)
        print(f"Done. Wrote {n} embedding file(s) under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
