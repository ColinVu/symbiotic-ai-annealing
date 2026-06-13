"""
Synthetic 1D ILR sanity check using only picklist JSON + ground truth CSV.

No video decoding, no CLIP embedding pass, no cache dependency.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..experiments.assignment_score import score_training_assignments_vs_ground_truth
from ..experiments.evaluator import _load_ground_truth_labels
from ..pipelines.video_training import _flatten_candidate_assignments, _load_picklists_nested_from_json
from ..training.weak_supervision import LabelKey, Segment, WeakSupervisedTrainer


def _print_enforced_settings(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("SYNTHETIC ILR SANITY (json + ground truth only)")
    print("=" * 60)
    print("Enforced ILR / supervision settings:")
    print("  use_cluster_voting: False")
    print("  ilr_allow_cross_round_swaps: False")
    print("  skip_ilr: False")
    print("  pca: none (default fit path)")
    print("  segment source: picklist json + ground truth only")
    print("  distance metric: euclidean (sanity-script override)")
    print("  centroid type: arithmetic mean (sanity-script override)")
    print("Run parameters:")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v!r}")
    print("=" * 60)


def _label_to_scalar(label: str) -> float:
    s = str(label).strip()
    m = re.match(r"^[cC](\d+)$", s)
    if m:
        return float(m.group(1))
    digits = re.sub(r"\D", "", s)
    if digits:
        return float(int(digits[-12:]))
    return float(abs(hash(s)) % 10000)


def _discover_video_stems(picklist_json_dir: str) -> List[str]:
    stems: List[str] = []
    for p in sorted(Path(picklist_json_dir).glob("*.json")):
        stems.append(p.stem)
    if not stems:
        raise SystemExit(f"No json files found in {picklist_json_dir}")
    return stems


def _load_ground_truth_stems(ground_truth_csv: str) -> List[str]:
    path = Path(ground_truth_csv)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"No header row in ground truth CSV: {path}")
        stems = [h.strip() for h in reader.fieldnames if h and h.strip()]
    if not stems:
        raise SystemExit(f"No non-empty ground truth columns in: {path}")
    return stems


class EuclideanSanityTrainer(WeakSupervisedTrainer):
    """Script-local ILR variant for scalar synthetic embeddings."""

    def cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        # Keep method name expected by base ILR internals; metric is L2 here.
        av = np.asarray(a, dtype=np.float64).reshape(-1)
        bv = np.asarray(b, dtype=np.float64).reshape(-1)
        return float(np.linalg.norm(av - bv))

    def compute_centroids(
        self,
        segments: List[Segment],
        labels: Dict[LabelKey, str],
    ) -> Dict[str, np.ndarray]:
        """Per-label arithmetic mean of all frames (no spherical normalization)."""
        label_frames: Dict[str, List[np.ndarray]] = {}
        for seg in segments:
            em = np.asarray(seg.embeddings, dtype=np.float64)
            if em.size == 0:
                continue
            label_frames.setdefault(labels[seg.label_key], []).append(em)
        centroids: Dict[str, np.ndarray] = {}
        for label, blocks in label_frames.items():
            all_frames = np.vstack(blocks)
            centroids[label] = all_frames.mean(axis=0).astype(np.float64)
        return centroids

    def _loo_spherical_centroid(
        self,
        segments: List[Segment],
        labels: Dict[LabelKey, str],
        label: str,
        exclude: Segment,
    ) -> Optional[np.ndarray]:
        """
        Leave-one-out arithmetic centroid for this sanity metric.
        (Name kept for base-class call sites.)
        """
        mates = [
            s
            for s in segments
            if labels[s.label_key] == label and s.label_key != exclude.label_key
        ]
        if not mates:
            mates = [s for s in segments if labels[s.label_key] == label]
        if not mates:
            return None
        blocks: List[np.ndarray] = []
        for m in mates:
            em = np.asarray(m.embeddings, dtype=np.float64)
            if em.size:
                blocks.append(em)
        if not blocks:
            return None
        return np.vstack(blocks).mean(axis=0).astype(np.float64)


def _build_synthetic_video_segments(
    picklist_json_dir: str,
    ground_truth_csv: str,
    video_stems: List[str],
    noise_std: float,
    frames_mode: str,
    fixed_frames_per_segment: int,
    random_seed: int,
) -> Tuple[Dict[str, Tuple[List[Segment], List[str]]], Dict[str, Any]]:
    gt_path = str(Path(ground_truth_csv).resolve())
    rng = np.random.default_rng(int(random_seed) + 911382)

    video_segments: Dict[str, Tuple[List[Segment], List[str]]] = {}
    report: Dict[str, Any] = {"videos": {}, "errors": [], "warnings": []}

    for stem in video_stems:
        json_path = str(Path(picklist_json_dir) / f"{stem}.json")
        if not Path(json_path).is_file():
            report["errors"].append(f"{stem}: missing picklist json: {json_path}")
            continue
        picklists_nested = _load_picklists_nested_from_json(json_path)
        flat_picklist = [str(x) for block in picklists_nested for x in block]
        per_segment_candidates = _flatten_candidate_assignments(picklists_nested)

        try:
            expected = _load_ground_truth_labels(gt_path, stem)
        except ValueError as e:
            report["errors"].append(f"{stem}: ground truth load failed: {e}")
            continue

        if len(per_segment_candidates) != len(flat_picklist):
            report["errors"].append(
                f"{stem}: candidate flatten size {len(per_segment_candidates)} != "
                f"flat picklist size {len(flat_picklist)}"
            )
            continue
        if len(expected) != len(flat_picklist):
            report["errors"].append(
                f"{stem}: ground truth length {len(expected)} != picklist segment count {len(flat_picklist)}"
            )
            continue

        segments: List[Segment] = []
        for i, gt_label in enumerate(expected):
            cand = per_segment_candidates[i]
            if gt_label not in cand:
                report["errors"].append(
                    f"{stem}: segment {i} gt {gt_label!r} not in candidate multiset {cand!r}"
                )
                continue
            n_frames = max(1, int(fixed_frames_per_segment)) if frames_mode == "fixed" else 1
            base = np.full((n_frames, 1), _label_to_scalar(gt_label), dtype=np.float64)
            if noise_std > 0:
                base = base + rng.normal(0.0, float(noise_std), size=base.shape)
            segments.append(
                Segment(
                    segment_id=i,
                    embeddings=base,
                    video_id=stem,
                    candidate_labels=cand,
                    is_placeholder=False,
                )
            )

        report["videos"][stem] = {
            "video_stem": stem,
            "segments": len(segments),
            "gt_rows": len(expected),
            "flat_picklist_rows": len(flat_picklist),
        }
        video_segments[stem] = (segments, flat_picklist)

    if report["errors"]:
        print("Preflight FAILED:", file=sys.stderr)
        for e in report["errors"]:
            print(f"  {e}", file=sys.stderr)
        raise SystemExit(2)
    return video_segments, report


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
    p.add_argument("--picklist-json-dir", required=True)
    p.add_argument("--ground-truth-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--ilr-epochs", type=int, default=5000)
    p.add_argument("--bad-swap-cool-divisor", type=float, default=200.0)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--noise-std", type=float, default=0.0)
    p.add_argument(
        "--frames-per-segment-mode",
        choices=("preserve", "fixed"),
        default="preserve",
        help="preserve uses 1 synthetic row/segment in this script; fixed uses --fixed-frames-per-segment",
    )
    p.add_argument("--fixed-frames-per-segment", type=int, default=1)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    verbose = not args.quiet
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _print_enforced_settings(args)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    gt_stems = _load_ground_truth_stems(args.ground_truth_csv)
    json_stems = set(_discover_video_stems(args.picklist_json_dir))
    stem_order = [s for s in gt_stems if s in json_stems]
    missing_json = [s for s in gt_stems if s not in json_stems]
    if not stem_order:
        raise SystemExit(
            "No overlapping stems between ground_truth columns and picklist json files."
        )
    if missing_json and verbose:
        print(
            "Warning: ground truth stems missing picklist json and skipped: "
            f"{missing_json}"
        )

    video_segments, preflight = _build_synthetic_video_segments(
        picklist_json_dir=args.picklist_json_dir,
        ground_truth_csv=args.ground_truth_csv,
        video_stems=stem_order,
        noise_std=float(args.noise_std),
        frames_mode=str(args.frames_per_segment_mode),
        fixed_frames_per_segment=int(args.fixed_frames_per_segment),
        random_seed=int(args.random_seed),
    )

    trainer = EuclideanSanityTrainer(
        ilr_epochs=int(args.ilr_epochs),
        bad_swap_cool_divisor=float(args.bad_swap_cool_divisor),
        random_seed=int(args.random_seed),
        ilr_allow_cross_round_swaps=False,
    )
    trainer.fit(
        video_segments,
        verbose=verbose,
        skip_ilr=False,
        initial_cluster_voting_csv=None,
        use_cluster_voting=False,
    )

    score = score_training_assignments_vs_ground_truth(trainer, args.ground_truth_csv, stem_order)
    results_path = out_dir / "synthetic_eval_results.json"
    config_path = out_dir / "synthetic_run_config.json"
    mismatches_path = out_dir / "synthetic_mismatches.csv"

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(score, f, indent=2)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "enforced": {
                    "use_cluster_voting": False,
                    "ilr_allow_cross_round_swaps": False,
                    "skip_ilr": False,
                    "pca": False,
                },
                "segment_source": "picklist_json_plus_ground_truth",
                "video_stems": stem_order,
                "cli": vars(args),
                "preflight": preflight,
            },
            f,
            indent=2,
        )
    n_mm = _write_mismatches_csv(trainer, args.ground_truth_csv, stem_order, str(mismatches_path))

    if verbose:
        m = score["metrics"]
        print("\n" + "=" * 60)
        print("SYNTHETIC ILR SUMMARY")
        print("=" * 60)
        print(f"Carry segments compared: {m['carry_segments_used']}")
        print(f"Top-1 hits: {m['segment_top1_hits']}")
        print(f"Top-1 accuracy: {m['segment_top1_accuracy']:.4f}")
        print(f"Mismatch rows: {n_mm}")
        print(f"Artifacts: {results_path}, {config_path}, {mismatches_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
