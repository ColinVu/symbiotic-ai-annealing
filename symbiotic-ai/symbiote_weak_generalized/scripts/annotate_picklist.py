"""
Produce an annotated MP4 from a picklist video with:

  - State banner (top, 65 px):  PICK / CARRY WITH / PLACE / CARRY EMPTY  (colour-coded)
  - Prediction bar (below, 55 px): shelf-constrained object label + confidence per
    CARRY_WITH segment; shows "Picking…" / "Placing: {label}" for other states.

Works from cached embeddings with no CLIP load when training cache is present.
Falls back to live CLIP embedding for uncached videos.

Usage (from symbiotic-ai/ directory)
-------------------------------------
python3 -m symbiote_weak_generalized.scripts.annotate_picklist \\
    --video ./hmm-testing/picklist_videos/picklist_312.MP4 \\
    --picklist-json ./hmm-testing/picklist_jsons/picklist_312.json \\
    --model-dir ../models/classifier \\
    --manual-labels ./hmm-testing/picklist_labels/picklist_312.csv \\
    --output ../outputs/picklist_312_annotated.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from ..embeddings.cache_manager import load_frame_from_cache
from ..persistence.model_io import load_model
from ..pipelines.video_training import (
    _flatten_candidate_assignments,
    _glob_cached_frame_indices_first_label,
    _load_picklists_nested_from_json,
    _process_single_video_from_cache,
)
from ..state_detection.compact_timeline import read_compact_state_table
from ..training.weak_supervision import spherical_mean

# ── shelf mapping (last digit of picklist stem → prefix) ──────────────────────
SHELF_DIGIT_MAP: Dict[str, str] = {"1": "c", "2": "d", "3": "e", "4": "f", "5": "g"}

# ── compact CSV code → state name ──────────────────────────────────────────────
_CODE_STATE: Dict[str, str] = {
    "a": "PICK",
    "e": "CARRY_WITH",
    "i": "PLACE",
    "m": "CARRY_EMPTY",
}

# ── overlay geometry ───────────────────────────────────────────────────────────
_BANNER_H = 65   # state name bar
_PRED_H   = 55   # prediction / info bar
_OVERLAY_H = _BANNER_H + _PRED_H

# ── state colours (BGR, matches hmm_gt_overlay.py) ────────────────────────────
_STATE_BGR: Dict[str, Tuple[int, int, int]] = {
    "PICK":        (30,  200,  30),
    "CARRY_WITH":  (200, 200,   0),
    "PLACE":       (0,    30, 220),
    "CARRY_EMPTY": (150, 150, 150),
}
_STATE_DISPLAY: Dict[str, str] = {
    "PICK":        "PICK",
    "CARRY_WITH":  "CARRY WITH",
    "PLACE":       "PLACE",
    "CARRY_EMPTY": "CARRY EMPTY",
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── per-frame state lookup ─────────────────────────────────────────────────────

def _build_frame_state_array(
    compact_csv_path: str,
    total_frames: int,
    frame_indexing: str = "opencv0",
) -> List[str]:
    """Return list[state_str] indexed by 0-based OpenCV frame index."""
    df = read_compact_state_table(compact_csv_path)
    if df is None or df.empty:
        return ["CARRY_EMPTY"] * total_frames

    rows: List[Tuple[int, str]] = []
    for fi, code in zip(df["frame_index"].tolist(), df["code"].tolist()):
        fi = int(fi)
        if frame_indexing != "opencv0":
            fi = max(0, fi - 1)  # pipeline1 → opencv0
        rows.append((fi, _CODE_STATE.get(str(code).strip().lower(), "CARRY_EMPTY")))
    rows.sort()

    out: List[str] = ["CARRY_EMPTY"] * total_frames
    boundary = 0
    cur = "CARRY_EMPTY"
    for i in range(total_frames):
        while boundary < len(rows) and rows[boundary][0] <= i:
            cur = rows[boundary][1]
            boundary += 1
        out[i] = cur
    return out


def _carry_intervals_1based(state_array: List[str]) -> List[Tuple[int, int]]:
    """Extract inclusive 1-based frame intervals for each CARRY_WITH run."""
    intervals: List[Tuple[int, int]] = []
    in_carry = False
    start = 0
    for i, s in enumerate(state_array):
        if s == "CARRY_WITH" and not in_carry:
            in_carry = True
            start = i + 1  # 1-based
        elif s != "CARRY_WITH" and in_carry:
            in_carry = False
            intervals.append((start, i))  # last 1-based = i
    if in_carry:
        intervals.append((start, len(state_array)))
    return intervals


def _interval_for_frame(frame_1based: int, intervals: List[Tuple[int, int]]) -> int:
    """Return CARRY_WITH interval index (0-based) for a 1-based frame, or -1."""
    for idx, (lo, hi) in enumerate(intervals):
        if lo <= frame_1based <= hi:
            return idx
    return -1


# ── model transforms (hand neutralizer + CLIP adapter) ────────────────────────

def _load_transforms(model_dir: str, metadata: dict, device: str):
    neutralizer = None
    adapter = None
    hn_path = os.path.join(model_dir, "hand_neutralizer.json")
    ap_path = os.path.join(model_dir, "clip_adapter.pt")
    if os.path.isfile(hn_path):
        from ..training.hand_neutralizer import HandNeutralizer
        with open(hn_path, "r", encoding="utf-8") as f:
            neutralizer = HandNeutralizer.from_state_dict(json.load(f), verbose=False)
    if os.path.isfile(ap_path):
        from ..training.clip_adapter import CLIPAdapter
        dim = int(metadata.get("embedding_dim", 512))
        m = CLIPAdapter(dim)
        m.load_state_dict(torch.load(ap_path, map_location=device))
        m.eval()
        if device == "cuda":
            m = m.to(device)
        adapter = m
    return neutralizer, adapter


def _postprocess(emb: np.ndarray, neutralizer, adapter, device: str) -> np.ndarray:
    x = np.asarray(emb, dtype=np.float64).reshape(-1)
    if neutralizer is not None and neutralizer.enabled:
        x = np.asarray(neutralizer.neutralize(x), dtype=np.float64).reshape(-1)
    if adapter is not None:
        with torch.no_grad():
            t = torch.from_numpy(x).float().unsqueeze(0).to(device)
            x = adapter(t).cpu().numpy().reshape(-1)
    return x.astype(np.float64)


# ── shelf-constrained prediction ───────────────────────────────────────────────

def _predict_constrained(
    centroid_model,
    processed_emb: np.ndarray,
    shelf_prefix: Optional[str],
) -> Tuple[str, float]:
    """Nearest-centroid; restricted to shelf_prefix labels when given."""
    if not shelf_prefix:
        return centroid_model.predict_with_confidence(processed_emb)
    candidates = {
        lab: c for lab, c in centroid_model.centroids.items()
        if lab.startswith(shelf_prefix)
    }
    if not candidates:
        return centroid_model.predict_with_confidence(processed_emb)
    x = centroid_model._l2_normalize(processed_emb.reshape(-1))
    best_label, best_dist = None, float("inf")
    for lab, centroid in candidates.items():
        dist = centroid_model.cosine_distance(x, centroid)
        if dist < best_dist:
            best_dist, best_label = dist, lab
    neg_dists = {lab: -centroid_model.cosine_distance(x, c) for lab, c in candidates.items()}
    max_neg = max(neg_dists.values())
    exp_scores = {lab: math.exp(v - max_neg) for lab, v in neg_dists.items()}
    total = sum(exp_scores.values())
    conf = exp_scores.get(best_label, 0.0) / max(total, 1e-12)
    return str(best_label), float(conf)


# ── segment prediction ─────────────────────────────────────────────────────────

def _predict_segments_from_cache(
    video_path: str,
    picklists_nested: List[List[str]],
    per_video_cache: str,
    manual_labels_dir: Optional[str],
    centroid_model,
    neutralizer,
    adapter,
    device: str,
    shelf_prefix: Optional[str],
    frame_skip: int,
    compact_frame_indexing: str,
    verbose: bool,
) -> List[Tuple[Optional[str], Optional[float]]]:
    """Load segments from disk cache and return shelf-constrained (label, conf) per segment."""
    segments, flat_picklist, _ = _process_single_video_from_cache(
        video_path,
        picklists_nested,
        per_video_cache,
        manual_labels_dir,
        require_manual_label_csv=(manual_labels_dir is not None),
        compact_frame_indexing=compact_frame_indexing,
        frame_skip=frame_skip,
        verbose=verbose,
    )
    results: List[Tuple[Optional[str], Optional[float]]] = []
    for seg in segments:
        if seg.is_placeholder or seg.embeddings.size == 0:
            results.append((None, None))
            continue
        em = np.asarray(seg.embeddings, dtype=np.float64)
        n = em.shape[0]
        processed = np.vstack([_postprocess(em[i], neutralizer, adapter, device) for i in range(n)])
        mean_emb = spherical_mean(processed)
        lab, conf = _predict_constrained(centroid_model, mean_emb, shelf_prefix)
        results.append((lab, conf))
    return results


def _predict_segments_via_clip(
    video_path: str,
    picklists_nested: List[List[str]],
    cache_dir: str,
    model_dir: str,
    manual_labels_dir: Optional[str],
    centroid_model,
    neutralizer,
    adapter,
    device: str,
    shelf_prefix: Optional[str],
    carry_intervals: List[Tuple[int, int]],
    frame_skip: int,
    blur_threshold: float,
    compact_frame_indexing: str,
    verbose: bool,
) -> List[Tuple[Optional[str], Optional[float]]]:
    """Embed with CLIP (populates cache), then delegate to cache path."""
    from ..inference.recognizer import ObjectRecognizer
    from ..preprocessing.video_processor import process_video_frames

    if verbose:
        print("  Loading CLIP model for embedding (first run only; results will be cached)…")
    recognizer = ObjectRecognizer(model_dir)

    flat = [item for sublist in picklists_nested for item in sublist]
    first_label = flat[0] if flat else "unknown"

    process_video_frames(
        video_path=video_path,
        label=first_label,
        model=recognizer.clip_model,
        processor=recognizer.processor,
        cache_dir=cache_dir,
        threshold=float(blur_threshold),
        frame_skip=int(frame_skip),
        state_detection_func=None,
        verbose=verbose,
        allowed_frame_intervals_1based=carry_intervals if carry_intervals else None,
    )
    return _predict_segments_from_cache(
        video_path,
        picklists_nested,
        cache_dir,
        manual_labels_dir,
        centroid_model,
        neutralizer,
        adapter,
        device,
        shelf_prefix,
        frame_skip,
        compact_frame_indexing,
        verbose,
    )


def _cache_has_embeddings(cache_dir: str, flat_picklist: List[str]) -> bool:
    if not os.path.isdir(cache_dir):
        return False
    first_label = flat_picklist[0] if flat_picklist else "unknown"
    return bool(_glob_cached_frame_indices_first_label(cache_dir, first_label))


# ── overlay drawing ────────────────────────────────────────────────────────────

def _text_centered(
    img: np.ndarray, text: str, y_center: int,
    scale: float, thickness: int,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    w = img.shape[1]
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    x = max(4, (w - tw) // 2)
    y = y_center + th // 2
    cv2.putText(img, text, (x + 2, y + 2), _FONT, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, (x, y), _FONT, scale, color, thickness)


def _build_annotated_frame(
    frame: np.ndarray,
    state: str,
    seg_idx: int,
    total_segs: int,
    pred_label: Optional[str],
    pred_conf: Optional[float],
    shelf_prefix: Optional[str],
    last_carry_label: Optional[str],
) -> np.ndarray:
    h, w = frame.shape[:2]
    canvas = np.zeros((_OVERLAY_H + h, w, 3), dtype=np.uint8)

    # ── state banner ──────────────────────────────────────────────────────────
    bg = _STATE_BGR.get(state, (120, 120, 120))
    cv2.rectangle(canvas, (0, 0), (w, _BANNER_H), bg, -1)
    _text_centered(canvas, _STATE_DISPLAY.get(state, state), _BANNER_H // 2, 1.4, 2)

    # ── prediction bar ────────────────────────────────────────────────────────
    cv2.rectangle(canvas, (0, _BANNER_H), (w, _OVERLAY_H), (28, 28, 28), -1)
    y_pred = _BANNER_H + _PRED_H // 2

    if state == "CARRY_WITH" and pred_label is not None:
        conf_str = f"{pred_conf * 100:.0f}%" if pred_conf is not None else "—"
        shelf_tag = f"  ·  shelf {shelf_prefix}" if shelf_prefix else ""
        info = f"Pick {seg_idx + 1} / {total_segs}   {pred_label}   {conf_str}{shelf_tag}"
        _text_centered(canvas, info, y_pred, 1.05, 2, color=(230, 230, 230))

    elif state == "PICK":
        _text_centered(canvas, "Picking…", y_pred, 0.9, 1, color=(160, 230, 160))

    elif state == "PLACE" and last_carry_label is not None:
        _text_centered(
            canvas, f"Placing:  {last_carry_label}", y_pred, 0.95, 1, color=(200, 200, 255)
        )

    # ── original frame ────────────────────────────────────────────────────────
    canvas[_OVERLAY_H:] = frame
    return canvas


# ── main pipeline ──────────────────────────────────────────────────────────────

def run_annotate(
    video_path: str,
    picklist_json_path: str,
    model_dir: str,
    *,
    manual_labels_path: Optional[str] = None,
    manual_labels_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    frame_skip: int = 5,
    compact_frame_indexing: str = "opencv0",
    blur_threshold: float = 100.0,
    verbose: bool = True,
) -> str:
    video_p = Path(video_path).resolve()
    model_p = Path(model_dir).resolve()
    stem = video_p.stem

    # ── resolve optional paths ─────────────────────────────────────────────────
    if manual_labels_path is None and manual_labels_dir is not None:
        candidate = Path(manual_labels_dir) / f"{stem}.csv"
        if candidate.is_file():
            manual_labels_path = str(candidate)

    if output_path is None:
        output_path = str(video_p.parent / f"{stem}_annotated.mp4")

    if cache_dir is None:
        cache_dir = str(model_p / ".cache" / stem)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # ── load centroid model ────────────────────────────────────────────────────
    if verbose:
        print(f"Loading centroid model from {model_p} …")
    centroid_model, metadata = load_model(str(model_p))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    neutralizer, adapter = _load_transforms(str(model_p), metadata, device)

    # ── picklist & shelf ───────────────────────────────────────────────────────
    picklists_nested = _load_picklists_nested_from_json(picklist_json_path)
    flat_picklist = [item for sublist in picklists_nested for item in sublist]
    total_segs = len(flat_picklist)
    shelf_prefix = SHELF_DIGIT_MAP.get(stem[-1]) if stem else None

    if verbose:
        print(f"Picklist: {total_segs} picks, shelf={shelf_prefix!r}")

    # ── video metadata ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_p))
    if not cap.isOpened():
        raise SystemExit(f"Error: cannot open video {video_p}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if verbose:
        print(f"Video: {total_frames} frames @ {fps:.2f} fps  ({w}×{h})")

    # ── state array ────────────────────────────────────────────────────────────
    if manual_labels_path and Path(manual_labels_path).is_file():
        if verbose:
            print(f"State labels: {manual_labels_path}")
        state_array = _build_frame_state_array(
            manual_labels_path, total_frames, compact_frame_indexing
        )
    else:
        if verbose:
            print(
                "  [warning] No manual labels found; treating all frames as CARRY_WITH.\n"
                "  Pass --manual-labels or --manual-labels-dir for accurate state segmentation."
            )
        state_array = ["CARRY_WITH"] * total_frames

    carry_intervals = _carry_intervals_1based(state_array)
    if verbose:
        print(f"CARRY_WITH intervals: {len(carry_intervals)}")

    # ── extract embeddings & predict per segment ───────────────────────────────
    _labels_dir_for_cache = str(Path(manual_labels_path).parent) if manual_labels_path else None

    if _cache_has_embeddings(cache_dir, flat_picklist):
        if verbose:
            print("Cache found — loading embeddings (no CLIP needed).")
        seg_predictions = _predict_segments_from_cache(
            str(video_p), picklists_nested, cache_dir,
            _labels_dir_for_cache,
            centroid_model, neutralizer, adapter, device,
            shelf_prefix, frame_skip, compact_frame_indexing, verbose,
        )
    else:
        if verbose:
            print("No cache found — embedding with CLIP (results will be saved to cache).")
        seg_predictions = _predict_segments_via_clip(
            str(video_p), picklists_nested, cache_dir, str(model_p),
            _labels_dir_for_cache,
            centroid_model, neutralizer, adapter, device,
            shelf_prefix, carry_intervals, frame_skip, blur_threshold,
            compact_frame_indexing, verbose,
        )

    if verbose:
        print(f"\nPer-segment predictions ({len(seg_predictions)}):")
        for i, (lab, conf) in enumerate(seg_predictions):
            conf_str = f"{conf*100:.0f}%" if conf is not None else "—"
            print(f"  seg {i+1:3d}:  {lab or '(no embed)':>8}  {conf_str}")

    # ── write annotated video ──────────────────────────────────────────────────
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_p), fourcc, fps, (w, h + _OVERLAY_H))
    if not writer.isOpened():
        raise SystemExit(f"Error: cannot open VideoWriter at {out_p}")

    cap = cv2.VideoCapture(str(video_p))
    opencv_idx = 0
    last_carry_label: Optional[str] = None

    if verbose:
        print(f"\nWriting annotated video → {out_p} …")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_1based = opencv_idx + 1
        state = state_array[opencv_idx] if opencv_idx < len(state_array) else "CARRY_EMPTY"
        seg_idx = _interval_for_frame(frame_1based, carry_intervals)

        pred_label: Optional[str] = None
        pred_conf: Optional[float] = None
        if seg_idx >= 0 and seg_idx < len(seg_predictions):
            pred_label, pred_conf = seg_predictions[seg_idx]
        if state == "CARRY_WITH" and pred_label:
            last_carry_label = pred_label

        annotated = _build_annotated_frame(
            frame, state, seg_idx, total_segs,
            pred_label, pred_conf, shelf_prefix, last_carry_label,
        )
        writer.write(annotated)
        opencv_idx += 1

    cap.release()
    writer.release()

    if verbose:
        print(f"Done. Output: {out_p}")
    return str(out_p)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Annotate a picklist video with state segmentation + shelf-constrained inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--picklist-json", required=True, help="Picklist JSON for this video")
    p.add_argument("--model-dir", required=True, help="Saved centroid model directory")
    p.add_argument(
        "--manual-labels",
        default=None,
        help="Compact state CSV for this video (e.g. picklist_labels/picklist_312.csv)",
    )
    p.add_argument(
        "--manual-labels-dir",
        default=None,
        help="Directory of compact state CSVs; script looks for {stem}.csv automatically",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Per-video cache dir (default: <model-dir>/.cache/<stem>)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output video path (default: {video_dir}/{stem}_annotated.mp4)",
    )
    p.add_argument("--frame-skip", type=int, default=5, help="Must match training frame-skip")
    p.add_argument(
        "--compact-frame-indexing",
        default="opencv0",
        choices=["opencv0", "pipeline1"],
    )
    p.add_argument("--blur-threshold", type=float, default=100.0)
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    script_dir = Path(__file__).resolve().parents[2]

    def _abs(s: Optional[str]) -> Optional[str]:
        return os.path.normpath(os.path.join(script_dir, s)) if s else None

    run_annotate(
        video_path=_abs(args.video),
        picklist_json_path=_abs(args.picklist_json),
        model_dir=_abs(args.model_dir),
        manual_labels_path=_abs(args.manual_labels),
        manual_labels_dir=_abs(args.manual_labels_dir),
        cache_dir=_abs(args.cache_dir),
        output_path=_abs(args.output),
        frame_skip=args.frame_skip,
        compact_frame_indexing=args.compact_frame_indexing,
        blur_threshold=args.blur_threshold,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
