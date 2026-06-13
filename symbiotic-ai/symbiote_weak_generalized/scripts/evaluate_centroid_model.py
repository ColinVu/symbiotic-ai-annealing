"""
Evaluate a saved centroid model (nearest-centroid on CLIP embeddings).

Single video::

    python -m symbiote_weak_generalized.scripts.evaluate_centroid_model \\
        --model-dir ../models/classifier \\
        --video ./hmm-testing/picklist_videos/picklist_061.MP4 \\
        --ground-truth-csv ./ground_truth.csv \\
        --manual-labels-dir ./hmm-testing/picklist_labels

All videos in a directory (non-recursive .mp4/.MP4/.m4v)::

    python -m symbiote_weak_generalized.scripts.evaluate_centroid_model \\
        --model-dir ../models/classifier \\
        --videos-dir ./hmm-testing/picklist_videos \\
        --ground-truth-csv ./ground_truth.csv \\
        --manual-labels-dir ./hmm-testing/picklist_labels

Videos without a matching column in ``ground_truth.csv`` or without
``{stem}.csv`` under ``--manual-labels-dir`` are skipped with a warning.

Optional: ``--output-json`` — single-video writes one eval dict; multi-video
writes ``{ "aggregate": ..., "per_video": [...], "skipped": [...] }``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..pipelines.video_training import _list_videos_in_folder


def _default_state_csv(video_path: str, manual_labels_dir: str) -> str:
    stem = Path(video_path).stem
    p = Path(manual_labels_dir) / f"{stem}.csv"
    return str(p.resolve())


def _ground_truth_columns(ground_truth_csv: str) -> Set[str]:
    with open(ground_truth_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {
            h.strip()
            for h in (reader.fieldnames or [])
            if h and str(h).strip()
        }


def _print_eval_block(ev: Dict[str, Any]) -> None:
    m = ev["metrics"]
    print("\n" + "=" * 60)
    print("CENTROID MODEL EVALUATION (saved centroids, nearest-centroid on embeddings)")
    print("=" * 60)
    print(f"  model_dir:     {ev['model_dir']}")
    print(f"  video:         {ev['video']}")
    print(f"  video_stem:    {ev['video_stem']}")
    print(f"  ground_truth:  {ev['ground_truth_csv']}")
    print(f"  state_csv:     {ev['state_label_csv']}")
    print(f"  frame_skip:    {ev['frame_skip']}")
    print(f"  threshold:     {ev['threshold']}")
    print()
    print(f"  carry_segments_used:          {m['carry_segments_used']}")
    print(f"  segments_with_predictions:    {m['segments_with_predictions']}")
    print(f"  segment_top1_hits:            {m['segment_top1_hits']}")
    print(f"  segment_top1_accuracy:        {m['segment_top1_accuracy']:.6f}")
    print(f"  segment_top3_hits:            {m['segment_top3_hits']}")
    print(f"  segment_top3_hit_rate:        {m['segment_top3_hit_rate']:.6f}")
    print(f"  mean_segment_top1_confidence: {m['mean_segment_top1_confidence']:.6f}")
    print("=" * 60 + "\n")


def _run_evaluate(
    *,
    model_dir: str,
    video_path: str,
    ground_truth_csv: str,
    state_label_csv_path: str,
    frame_skip: int,
    compact_frame_indexing: str,
    threshold: float,
    verbose: bool,
    eval_cache_dir: Optional[str],
) -> Dict[str, Any]:
    from ..experiments.evaluator import evaluate_model

    return evaluate_model(
        model_dir=model_dir,
        video_path=video_path,
        ground_truth_csv=ground_truth_csv,
        state_label_csv_path=state_label_csv_path,
        frame_skip=int(frame_skip),
        compact_frame_indexing=str(compact_frame_indexing),
        threshold=float(threshold),
        verbose=bool(verbose),
        eval_cache_dir=eval_cache_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment-level top-1 / top-3: saved centroids vs ground_truth.csv (carry segments).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory with centroids.npy, model_metadata.json (and optional clip_adapter.pt, hand_neutralizer.json)",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--video",
        type=str,
        default=None,
        help="Single picklist video path; stem must match a column in ground_truth.csv",
    )
    src.add_argument(
        "--videos-dir",
        type=str,
        default=None,
        help="Directory of .mp4/.MP4/.m4v videos (non-recursive); eval each with GT + manual_labels_dir/{stem}.csv",
    )
    parser.add_argument(
        "--ground-truth-csv",
        type=str,
        required=True,
        help="Wide CSV: one column per video stem, rows = ordered true labels per pick",
    )
    st = parser.add_mutually_exclusive_group(required=False)
    st.add_argument(
        "--state-label-csv",
        type=str,
        default=None,
        help="(Single --video only) Compact manual state CSV for that video",
    )
    st.add_argument(
        "--manual-labels-dir",
        type=str,
        default=None,
        help="Directory with {stem}.csv per video. Required with --videos-dir; optional with --video if not using --state-label-csv",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="Frame skip for embedding (should match training)",
    )
    parser.add_argument(
        "--compact-frame-indexing",
        type=str,
        default="opencv0",
        choices=["opencv0", "pipeline1"],
        help="Must match training / manual CSV frame numbering",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Blur / Laplacian threshold passed to frame pipeline (default 50, sweep alignment)",
    )
    parser.add_argument(
        "--eval-cache-dir",
        type=str,
        default=None,
        help="Optional CLIP frame cache root (single-video: one dir; multi-video: set per-video under a parent if omitted)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write results JSON (single video: one eval; multi: aggregate + per_video + skipped)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose embedding pipeline",
    )
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    ground_truth_csv = os.path.abspath(args.ground_truth_csv)
    eval_cache_root = os.path.abspath(args.eval_cache_dir) if args.eval_cache_dir else None

    if not Path(model_dir).is_dir():
        print(f"Error: --model-dir is not a directory: {model_dir}", file=sys.stderr)
        sys.exit(1)
    if not Path(ground_truth_csv).is_file():
        print(f"Error: --ground-truth-csv not found: {ground_truth_csv}", file=sys.stderr)
        sys.exit(1)

    if args.videos_dir:
        if args.state_label_csv:
            print("Error: --state-label-csv cannot be used with --videos-dir (use --manual-labels-dir).", file=sys.stderr)
            sys.exit(1)
        if not args.manual_labels_dir:
            print("Error: --videos-dir requires --manual-labels-dir.", file=sys.stderr)
            sys.exit(1)
        _run_multi(
            model_dir=model_dir,
            videos_dir=os.path.abspath(args.videos_dir),
            ground_truth_csv=ground_truth_csv,
            manual_labels_dir=os.path.abspath(args.manual_labels_dir),
            frame_skip=int(args.frame_skip),
            compact_frame_indexing=str(args.compact_frame_indexing),
            threshold=float(args.threshold),
            verbose=bool(args.verbose),
            eval_cache_root=eval_cache_root,
            output_json=args.output_json,
        )
        return

    # Single --video
    if not args.video:
        print("Error: internal: --video missing", file=sys.stderr)
        sys.exit(1)
    if not args.state_label_csv and not args.manual_labels_dir:
        print("Error: with --video, pass either --state-label-csv or --manual-labels-dir.", file=sys.stderr)
        sys.exit(1)

    video_path = os.path.abspath(args.video)
    if not Path(video_path).is_file():
        print(f"Error: --video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if args.state_label_csv:
        state_csv = os.path.abspath(args.state_label_csv)
    else:
        state_csv = _default_state_csv(video_path, os.path.abspath(args.manual_labels_dir))

    if not Path(state_csv).is_file():
        print(f"Error: state label CSV not found: {state_csv}", file=sys.stderr)
        sys.exit(1)

    ev = _run_evaluate(
        model_dir=model_dir,
        video_path=video_path,
        ground_truth_csv=ground_truth_csv,
        state_label_csv_path=state_csv,
        frame_skip=int(args.frame_skip),
        compact_frame_indexing=str(args.compact_frame_indexing),
        threshold=float(args.threshold),
        verbose=bool(args.verbose),
        eval_cache_dir=eval_cache_root,
    )
    _print_eval_block(ev)

    if args.output_json:
        out_path = Path(args.output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(ev, f, indent=2)
        print(f"Wrote full results to {out_path}")


def _run_multi(
    *,
    model_dir: str,
    videos_dir: str,
    ground_truth_csv: str,
    manual_labels_dir: str,
    frame_skip: int,
    compact_frame_indexing: str,
    threshold: float,
    verbose: bool,
    eval_cache_root: Optional[str],
    output_json: Optional[str],
) -> None:
    video_paths = _list_videos_in_folder(videos_dir)

    gt_cols = _ground_truth_columns(ground_truth_csv)
    per_video: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    total_top1 = 0
    total_top3 = 0
    total_segs = 0

    print("\n" + "=" * 60)
    print("CENTROID MODEL EVALUATION (multi-video)")
    print("=" * 60)
    print(f"  model_dir:          {model_dir}")
    print(f"  videos_dir:         {videos_dir}")
    print(f"  ground_truth_csv:   {ground_truth_csv}")
    print(f"  manual_labels_dir:  {manual_labels_dir}")
    print(f"  videos found:       {len(video_paths)}")
    print("=" * 60)

    for vp in video_paths:
        stem = Path(vp).stem
        if stem not in gt_cols:
            skipped.append({"video_stem": stem, "reason": "no_ground_truth_column"})
            print(f"  skip {stem}: no column in ground_truth.csv")
            continue
        state_csv = _default_state_csv(vp, manual_labels_dir)
        if not Path(state_csv).is_file():
            skipped.append({"video_stem": stem, "reason": "missing_state_csv", "path": state_csv})
            print(f"  skip {stem}: missing state CSV {state_csv}")
            continue

        per_cache: Optional[str] = None
        if eval_cache_root:
            per_cache = str(Path(eval_cache_root) / f"eval_{stem}")

        try:
            ev = _run_evaluate(
                model_dir=model_dir,
                video_path=os.path.abspath(vp),
                ground_truth_csv=ground_truth_csv,
                state_label_csv_path=state_csv,
                frame_skip=frame_skip,
                compact_frame_indexing=compact_frame_indexing,
                threshold=threshold,
                verbose=verbose,
                eval_cache_dir=per_cache,
            )
        except Exception as e:
            skipped.append({"video_stem": stem, "reason": "evaluate_failed", "detail": str(e)})
            print(f"  skip {stem}: {e}")
            continue

        per_video.append(ev)
        m = ev["metrics"]
        total_top1 += int(m["segment_top1_hits"])
        total_top3 += int(m["segment_top3_hits"])
        total_segs += int(m["carry_segments_used"])
        print(
            f"  {stem:32s}  top1={m['segment_top1_accuracy']:.4f}  "
            f"top3={m['segment_top3_hit_rate']:.4f}  segs={m['carry_segments_used']}"
        )

    micro_top1 = (total_top1 / total_segs) if total_segs else 0.0
    micro_top3 = (total_top3 / total_segs) if total_segs else 0.0

    print("\n" + "-" * 60)
    print(f"  videos_scored:     {len(per_video)}")
    print(f"  videos_skipped:    {len(skipped)}")
    print(f"  carry_segments:    {total_segs}")
    print(f"  micro_top1:        {micro_top1:.6f}  ({total_top1}/{total_segs} hits)")
    print(f"  micro_top3:        {micro_top3:.6f}  ({total_top3}/{total_segs} hits)")
    print("-" * 60 + "\n")

    if output_json:
        out_path = Path(output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "mode": "multi_video",
            "model_dir": model_dir,
            "videos_dir": videos_dir,
            "ground_truth_csv": ground_truth_csv,
            "manual_labels_dir": manual_labels_dir,
            "aggregate": {
                "videos_scored": len(per_video),
                "videos_skipped": len(skipped),
                "carry_segments_total": total_segs,
                "segment_top1_hits_total": total_top1,
                "segment_top3_hits_total": total_top3,
                "micro_segment_top1_accuracy": micro_top1,
                "micro_segment_top3_hit_rate": micro_top3,
            },
            "per_video": per_video,
            "skipped": skipped,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)
        print(f"Wrote multi-video results to {out_path}")


if __name__ == "__main__":
    main()
