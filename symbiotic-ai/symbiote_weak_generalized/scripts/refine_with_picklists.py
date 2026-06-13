"""
Joint global refinement: re-embed all videos listed in ``embedded_video_stems``,
run ILR once on the combined segment set (same constraints as training), then save
a new model and a report of label changes.

Usage::

    python -m symbiote_weak.scripts.refine_with_picklists \\
        --model-dir ../models/classifier \\
        --video-dir ./hmm-testing/videos \\
        --picklist-json-dir ./hmm-testing/picklist_jsons \\
        --manual-labels-dir ./hmm-testing/picklist_labels \\
        --output-dir ../models/classifier_refined \\
        --epochs 500 \\
        --random-seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from ..core.config import MODEL
from ..persistence.model_io import load_weak_trainer, save_model
from ..pipelines.video_training import (
    _load_picklists_nested_from_json,
    _process_single_video_to_segments,
)
from ..training.weak_supervision import LabelKey, Segment, WeakSupervisedTrainer


def _resolve_video_path(video_dir: str, stem: str) -> str:
    for ext in (".MP4", ".mp4", ".m4v", ".M4V"):
        p = os.path.join(video_dir, stem + ext)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"No video file for stem {stem!r} in {video_dir} (tried .MP4/.mp4/.m4v/.M4V)"
    )


def _initial_labels_random_within_candidates(
    segments: List[Segment], random_seed: int
) -> Dict[LabelKey, str]:
    """
    Random shuffle within each segment's ``candidate_labels`` (same logic as training).
    Groups segments by (video_id, candidate_labels), shuffles the multiset, assigns.
    """
    random.seed(random_seed)
    labels: Dict[LabelKey, str] = {}
    groups: Dict[Tuple[str, Tuple[str, ...]], List[Segment]] = defaultdict(list)
    for seg in segments:
        if seg.candidate_labels is None:
            raise ValueError(
                f"Segment {seg.label_key} missing candidate_labels; "
                "refinement requires them (set by _process_single_video_to_segments)."
            )
        groups[(seg.video_id, seg.candidate_labels)].append(seg)

    for (_vid_key, multiset), segs in groups.items():
        segs_sorted = sorted(segs, key=lambda s: s.segment_id)
        draw = list(multiset)
        if len(segs_sorted) != len(draw):
            raise ValueError(
                f"Group size mismatch: {len(segs_sorted)} segments vs multiset size {len(draw)} "
                f"for multiset {multiset}"
            )
        random.shuffle(draw)
        for seg, lab in zip(segs_sorted, draw):
            labels[seg.label_key] = lab
    return labels


def run_joint_refinement(
    model_dir: str,
    video_dir: str,
    picklist_json_dir: str,
    manual_labels_dir: str,
    output_dir: str,
    epochs: int,
    random_seed: int,
    threshold: float,
    frame_skip: int,
    htk_model_dir: Optional[str],
    aruco_config_path: Optional[str],
    compact_frame_indexing: str,
    verbose: bool = True,
) -> Tuple[WeakSupervisedTrainer, Dict[str, Any]]:
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    trainer = load_weak_trainer(
        model_dir,
        ilr_epochs_override=epochs,
        random_seed_override=random_seed,
    )
    stems = list(trainer.embedded_video_stems)
    if not stems:
        raise ValueError(
            "Model metadata has no embedded_video_stems; cannot discover training videos."
        )

    with open(os.path.join(model_dir, "model_metadata.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    config: Dict[str, Any] = dict(meta.get("config", {}))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = AutoModel.from_pretrained(MODEL)
    clip_model.eval()
    if device == "cuda":
        clip_model = clip_model.to(device)
    processor = AutoProcessor.from_pretrained(MODEL)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    base_cache = os.path.join(output_dir, ".cache_refine")

    video_segments: Dict[str, Tuple[List[Segment], List[str]]] = {}

    if verbose:
        print("=" * 60)
        print("JOINT GLOBAL REFINEMENT")
        print("=" * 60)
        print(f"Model: {model_dir}")
        print(f"Videos ({len(stems)}): {stems}")
        print(f"ILR epochs: {epochs}, seed: {random_seed}")

    for stem in stems:
        video_path = _resolve_video_path(video_dir, stem)
        json_path = os.path.join(picklist_json_dir, f"{stem}.json")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Missing picklist JSON: {json_path}")
        picklists_nested = _load_picklists_nested_from_json(json_path)
        per_video_cache = os.path.join(base_cache, stem)
        segments, flat_picklist, video_name = _process_single_video_to_segments(
            video_path,
            picklists_nested,
            clip_model,
            processor,
            per_video_cache,
            manual_labels_dir,
            require_manual_label_csv=True,
            compact_frame_indexing=compact_frame_indexing,
            threshold=threshold,
            frame_skip=frame_skip,
            htk_model_dir=htk_model_dir,
            aruco_config_path=aruco_config_path,
            verbose=verbose,
        )
        video_segments[video_name] = (segments, flat_picklist)

    all_segments: List[Segment] = []
    for _vid, (segs, _pl) in video_segments.items():
        all_segments.extend(segs)

    if len(all_segments) == 0:
        raise ValueError("No segments collected for refinement.")

    initial_labels = _initial_labels_random_within_candidates(all_segments, random_seed)

    init_centroids = trainer.compute_centroids(all_segments, initial_labels)
    initial_cost = trainer.compute_total_cosine_cost(
        all_segments, initial_labels, init_centroids
    )

    if verbose:
        print(f"\nInitial cosine cost (before joint ILR): {initial_cost:.4f}")

    refined_labels = trainer.refine_labels(all_segments, initial_labels, verbose=verbose)

    trainer.last_refined_labels = dict(refined_labels)
    trainer._rebuild_label_video_means_from_segments(all_segments, refined_labels)
    trainer.centroid_stds = trainer.compute_centroid_stds(all_segments, refined_labels)

    final_cost = trainer.compute_total_cosine_cost(
        all_segments,
        refined_labels,
        trainer.centroids,
    )

    reassignments: List[Dict[str, Any]] = []
    for seg in all_segments:
        lk = seg.label_key
        old_l = initial_labels.get(lk)
        new_l = refined_labels.get(lk)
        if old_l != new_l:
            reassignments.append(
                {
                    "video": seg.video_id,
                    "segment_id": seg.segment_id,
                    "old_label": old_l,
                    "new_label": new_l,
                }
            )

    report: Dict[str, Any] = {
        "refinement_metadata": {
            "source_model": os.path.abspath(model_dir),
            "output_dir": os.path.abspath(output_dir),
            "epochs": epochs,
            "random_seed": random_seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "reassignments": reassignments,
        "summary": {
            "total_segments": len(all_segments),
            "reassignments_count": len(reassignments),
            "videos_affected": sorted(
                {r["video"] for r in reassignments}
            ),
            "initial_cost": float(initial_cost),
            "final_cost": float(final_cost),
            "cost_delta": float(final_cost - initial_cost),
            "cost_reduction": float(initial_cost - final_cost),
        },
    }

    save_model(
        trainer,
        config,
        output_dir,
        embedded_video_stems_override=stems,
    )

    report_path = os.path.join(output_dir, "refinement_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    txt_path = os.path.join(output_dir, "refinement_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Joint global refinement report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Source model: {report['refinement_metadata']['source_model']}\n")
        f.write(f"Epochs: {epochs}, seed: {random_seed}\n")
        f.write(
            f"Initial cost: {initial_cost:.4f}  Final cost: {final_cost:.4f}  "
            f"Delta: {final_cost - initial_cost:.4f}\n"
        )
        f.write(f"Segments: {len(all_segments)}  Reassignments: {len(reassignments)}\n\n")
        by_vid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in reassignments:
            by_vid[r["video"]].append(r)
        for vid in sorted(by_vid.keys()):
            f.write(f"\n[{vid}]\n")
            for r in by_vid[vid]:
                f.write(
                    f"  segment {r['segment_id']}: {r['old_label']} -> {r['new_label']}\n"
                )

    if verbose:
        print(f"\nReport written to {report_path}")
        print(f"           and {txt_path}")
        print(f"\nRefined model saved to: {output_dir}")

    return trainer, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Joint ILR refinement over all embedded_video_stems; saves new model + report."
    )
    parser.add_argument("--model-dir", required=True, help="Existing trained model directory")
    parser.add_argument("--video-dir", required=True, help="Directory containing video files")
    parser.add_argument(
        "--picklist-json-dir",
        required=True,
        help="Directory with {stem}.json picklists",
    )
    parser.add_argument(
        "--manual-labels-dir",
        required=True,
        help="Directory with {stem}.csv manual state labels",
    )
    parser.add_argument("--output-dir", required=True, help="Where to save refined model")
    parser.add_argument("--epochs", type=int, default=500, help="ILR epochs (default 500)")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=30.0, help="Blur threshold for embedding")
    parser.add_argument("--frame-skip", type=int, default=3)
    parser.add_argument("--htk-model-dir", type=str, default=None)
    parser.add_argument("--aruco-config", type=str, default=None)
    parser.add_argument(
        "--compact-frame-indexing",
        type=str,
        default="opencv0",
        help="Frame indexing for manual CSV (default opencv0)",
    )
    parser.add_argument("--quiet", action="store_true", help="Less console output")
    args = parser.parse_args()

    htk = args.htk_model_dir
    aruco = args.aruco_config

    run_joint_refinement(
        model_dir=os.path.abspath(args.model_dir),
        video_dir=os.path.abspath(args.video_dir),
        picklist_json_dir=os.path.abspath(args.picklist_json_dir),
        manual_labels_dir=os.path.abspath(args.manual_labels_dir),
        output_dir=os.path.abspath(args.output_dir),
        epochs=args.epochs,
        random_seed=args.random_seed,
        threshold=args.threshold,
        frame_skip=args.frame_skip,
        htk_model_dir=os.path.abspath(htk) if htk else None,
        aruco_config_path=os.path.abspath(aruco) if aruco else None,
        compact_frame_indexing=args.compact_frame_indexing,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
