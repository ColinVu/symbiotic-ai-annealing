"""
Synthetic 1D ILR sanity check: rebuild carry segments from disk cache (same as
``run_multi_video_training_from_cache``), replace real segment embeddings with
ground-truth-derived 1D constants, run ILR with strict constraints, then score
vs ``ground_truth.csv``.

Usage::

    python -m symbiote_weak.scripts.synthetic_ilr_sanity \\
        --videos ./videos \\
        --picklist-json-dir ./picklist_jsons \\
        --manual-labels-dir ./picklist_labels \\
        --ground-truth-csv ./ground_truth.csv \\
        --output-dir ./synthetic_ilr_out \\
        --ilr-epochs 500 \\
        --random-seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..experiments.assignment_score import score_training_assignments_vs_ground_truth
from ..experiments.evaluator import _load_ground_truth_labels
from ..pipelines.video_training import (
    _list_videos_in_folder,
    _load_picklists_nested_from_json,
    _process_single_video_from_cache,
)
from ..training.weak_supervision import Segment, WeakSupervisedTrainer


def _print_enforced_settings(args: argparse.Namespace, cache_root: str) -> None:
    print("=" * 60)
    print("SYNTHETIC ILR SANITY (oracle 1D embeddings)")
    print("=" * 60)
    print("Enforced ILR / supervision settings (must not drift):")
    print(f"  use_cluster_voting: False")
    print(f"  ilr_allow_cross_round_swaps: False")
    print(f"  skip_ilr: False (ILR enabled)")
    print(f"  PCA: not used (default fit() path)")
    print(f"  init: random bijection per (video_id, candidate_labels) multiset")
    print("Run parameters:")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v!r}")
    print(f"  cache_root (resolved): {cache_root}")
    print("=" * 60)


def _label_to_scalar(label: str) -> float:
    """Map ``cNN`` -> NN; otherwise a stable small float from digits/hash."""
    s = str(label).strip()
    m = re.match(r"^[cC](\d+)$", s)
    if m:
        return float(m.group(1))
    digits = re.sub(r"\D", "", s)
    if digits:
        return float(int(digits[-12:]))  # avoid huge floats
    return float(abs(hash(s)) % 10_000)


def _build_video_segments_from_cache(
    video_paths: List[str],
    picklist_json_dir: str,
    manual_labels_dir: str,
    cache_root: str,
    compact_frame_indexing: str,
    frame_skip: int,
    verbose: bool,
) -> Dict[str, Tuple[List[Segment], List[str]]]:
    video_segments: Dict[str, Tuple[List[Segment], List[str]]] = {}
    for video_path in video_paths:
        stem = Path(video_path).stem
        json_path = os.path.join(picklist_json_dir, f"{stem}.json")
        if not os.path.isfile(json_path):
            raise SystemExit(f"Error: Missing picklist JSON for {stem!r}: {json_path}")
        picklists_nested = _load_picklists_nested_from_json(json_path)
        per_video_cache = os.path.join(cache_root, stem)
        if not os.path.isdir(per_video_cache):
            raise SystemExit(
                f"Error: No cache directory for {stem!r}: expected {per_video_cache}"
            )
        segments, flat_picklist, video_name = _process_single_video_from_cache(
            video_path,
            picklists_nested,
            per_video_cache,
            manual_labels_dir,
            require_manual_label_csv=True,
            compact_frame_indexing=compact_frame_indexing,
            frame_skip=frame_skip,
            verbose=verbose,
        )
        video_segments[video_name] = (segments, flat_picklist)
    return video_segments


def _validate_gt_alignment(
    video_segments: Dict[str, Tuple[List[Segment], List[str]]],
    ground_truth_csv: str,
) -> Dict[str, Any]:
    """
    Preflight: GT column exists, length matches segment count, each expected label
    is feasible for that segment's candidate multiset. Logs placeholder warnings.
    """
    report: Dict[str, Any] = {"videos": {}, "errors": [], "warnings": []}
    gt_path = str(Path(ground_truth_csv).resolve())

    for video_name, (segments, _flat) in video_segments.items():
        entry: Dict[str, Any] = {"video_stem": video_name}
        try:
            expected = _load_ground_truth_labels(gt_path, video_name)
        except ValueError as e:
            msg = f"{video_name}: ground truth load failed: {e}"
            report["errors"].append(msg)
            entry["error"] = str(e)
            report["videos"][video_name] = entry
            continue

        n_seg = len(segments)
        n_gt = len(expected)
        entry["num_segments"] = n_seg
        entry["num_gt_rows"] = n_gt
        if n_gt != n_seg:
            msg = (
                f"{video_name}: GT length {n_gt} != segment count {n_seg} "
                "(assignment_score compares index-wise)."
            )
            report["errors"].append(msg)

        segs_sorted = sorted(segments, key=lambda s: s.segment_id)
        for i, seg in enumerate(segs_sorted):
            if seg.segment_id != i:
                report["warnings"].append(
                    f"{video_name}: segment_id {seg.segment_id} at sorted position {i} "
                    f"(expected contiguous 0..N-1 for GT alignment)"
                )
            if i >= len(expected):
                continue
            exp = expected[i]
            cand = seg.candidate_labels
            if cand is not None and exp not in cand:
                msg = (
                    f"{video_name}: segment {i} GT label {exp!r} not in "
                    f"candidate_labels {cand!r}"
                )
                report["errors"].append(msg)
            if seg.is_placeholder:
                report["warnings"].append(
                    f"{video_name}: segment {i} is a placeholder — ILR does not swap "
                    f"placeholders; label stays at random init (may hurt accuracy vs GT)."
                )
        report["videos"][video_name] = entry

    if report["errors"]:
        print("Preflight FAILED:", file=sys.stderr)
        for e in report["errors"]:
            print(f"  {e}", file=sys.stderr)
        raise SystemExit(2)
    return report


def _inject_synthetic_1d_embeddings(
    video_segments: Dict[str, Tuple[List[Segment], List[str]]],
    ground_truth_csv: str,
    noise_std: float,
    frames_mode: str,
    fixed_frames_per_segment: int,
    random_seed: int,
) -> None:
    gt_path = str(Path(ground_truth_csv).resolve())
    rng = np.random.default_rng(int(random_seed) + 911382)

    for video_name, (segments, _flat) in video_segments.items():
        expected = _load_ground_truth_labels(gt_path, video_name)
        segs_sorted = sorted(segments, key=lambda s: s.segment_id)
        if len(expected) != len(segs_sorted):
            raise RuntimeError(
                f"{video_name}: inject step expected preflight to catch GT/segment length mismatch"
            )
        for i, seg in enumerate(segs_sorted):
            if seg.is_placeholder:
                continue
            scalar = _label_to_scalar(expected[i])
            if frames_mode == "fixed":
                n_frames = max(1, int(fixed_frames_per_segment))
            else:
                n_frames = int(seg.embeddings.shape[0])
                if n_frames < 1:
                    n_frames = 1
            base = np.full((n_frames, 1), float(scalar), dtype=np.float64)
            if noise_std > 0:
                base = base + rng.normal(0.0, float(noise_std), size=base.shape)
            seg.embeddings = base


def _write_mismatches_csv(
    trainer: WeakSupervisedTrainer,
    ground_truth_csv: str,
    video_stems: List[str],
    out_path: str,
) -> int:
    refined = trainer.last_refined_labels or {}
    gt_path = str(Path(ground_truth_csv).resolve())
    n_written = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_stem", "segment_idx", "expected", "predicted"])
        for stem in video_stems:
            try:
                expected = _load_ground_truth_labels(gt_path, stem)
            except ValueError:
                continue
            pred_by_seg: Dict[int, str] = {}
            for (vid, seg_id), lab in refined.items():
                if vid == stem:
                    pred_by_seg[int(seg_id)] = str(lab)
            for i, exp in enumerate(expected):
                pred = pred_by_seg.get(i, "")
                if pred != exp:
                    w.writerow([stem, i, exp, pred])
                    n_written += 1
    return n_written


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--videos", required=True, help="Directory of .mp4 videos (non-recursive)")
    p.add_argument("--picklist-json-dir", required=True)
    p.add_argument("--manual-labels-dir", required=True)
    p.add_argument("--ground-truth-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--cache-dir", default=None, help="Embedding cache root (default: OUTPUT_DIR/.cache)")
    p.add_argument("--ilr-epochs", type=int, default=500)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--compact-frame-indexing", default="opencv0")
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--noise-std", type=float, default=0.0, help="Gaussian noise on 1D synthetic values")
    p.add_argument(
        "--frames-per-segment-mode",
        choices=("preserve", "fixed"),
        default="preserve",
        help="preserve: use each segment's frame count; fixed: use --fixed-frames-per-segment",
    )
    p.add_argument(
        "--fixed-frames-per-segment",
        type=int,
        default=8,
        help="Used when --frames-per-segment-mode=fixed",
    )
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    verbose = not args.quiet

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_root = args.cache_dir or os.path.join(str(out_dir), ".cache")
    cache_root = str(Path(cache_root).resolve())

    _print_enforced_settings(args, cache_root)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    video_paths = _list_videos_in_folder(args.videos)
    stem_order = [Path(vp).stem for vp in video_paths]

    video_segments = _build_video_segments_from_cache(
        video_paths,
        args.picklist_json_dir,
        args.manual_labels_dir,
        cache_root,
        args.compact_frame_indexing,
        args.frame_skip,
        verbose,
    )

    preflight = _validate_gt_alignment(video_segments, args.ground_truth_csv)
    if preflight["warnings"] and verbose:
        print("Preflight warnings:")
        for w in preflight["warnings"][:50]:
            print(f"  {w}")
        if len(preflight["warnings"]) > 50:
            print(f"  ... ({len(preflight['warnings']) - 50} more)")

    _inject_synthetic_1d_embeddings(
        video_segments,
        args.ground_truth_csv,
        noise_std=float(args.noise_std),
        frames_mode=str(args.frames_per_segment_mode),
        fixed_frames_per_segment=int(args.fixed_frames_per_segment),
        random_seed=int(args.random_seed),
    )

    trainer = WeakSupervisedTrainer(
        ilr_epochs=int(args.ilr_epochs),
        initial_temp=1.0,
        temp_decay="exponential",
        decay_rate=0.99,
        random_seed=int(args.random_seed),
        variance_eps=1e-6,
        bad_swap_cool_divisor=50.0,
        detect_empty=False,
        min_frames_per_cluster=3,
        ilr_allow_cross_round_swaps=False,
    )

    trainer.fit(
        video_segments,
        verbose=verbose,
        skip_ilr=False,
        initial_cluster_voting_csv=None,
        use_cluster_voting=False,
    )

    score = score_training_assignments_vs_ground_truth(
        trainer,
        args.ground_truth_csv,
        stem_order,
    )

    config_payload = {
        "enforced": {
            "use_cluster_voting": False,
            "ilr_allow_cross_round_swaps": False,
            "skip_ilr": False,
            "pca": False,
            "initialization": "random bijection per (video_id, candidate_labels) multiset",
        },
        "cli": vars(args),
        "cache_root": cache_root,
        "video_stems": stem_order,
        "preflight": preflight,
    }
    results_path = out_dir / "synthetic_eval_results.json"
    config_path = out_dir / "synthetic_run_config.json"
    mismatches_path = out_dir / "synthetic_mismatches.csv"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(score, f, indent=2)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)
    n_mm = _write_mismatches_csv(
        trainer,
        args.ground_truth_csv,
        stem_order,
        str(mismatches_path),
    )

    m = score["metrics"]
    if verbose:
        print("\n" + "=" * 60)
        print("SYNTHETIC ILR SUMMARY")
        print("=" * 60)
        print(f"Carry segments compared (micro denominator): {m['carry_segments_used']}")
        print(f"Top-1 hits: {m['segment_top1_hits']}")
        print(f"Top-1 accuracy: {m['segment_top1_accuracy']:.4f}")
        print(f"Mismatch rows written: {n_mm} -> {mismatches_path}")
        print(f"Artifacts: {results_path}, {config_path}")
        acc = float(m["segment_top1_accuracy"])
        if acc >= 0.99:
            print(
                "\nInterpretation: ~100% under oracle 1D embeddings suggests annealing + "
                "constraints behave; CLIP failures likely embedding/objective mismatch."
            )
        elif acc < 0.99:
            print(
                "\nInterpretation: sub-perfect accuracy with oracle embeddings suggests "
                "alignment/indexing/placeholder/init issues or swap mechanics — "
                "not (only) CLIP quality."
            )
        print("=" * 60)


if __name__ == "__main__":
    main()
