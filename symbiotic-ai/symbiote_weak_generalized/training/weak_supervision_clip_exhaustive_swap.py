"""
Reference implementation: exhaustive pairwise ILR + Metropolis-style acceptance.

This was the default refinement in `weak_supervision.py` before switching to the
HSV (IterativeClustering.py) swap schedule. Not used by the training pipeline;
kept for experiments or future re-enable.

Expected `labels` keys: ``Segment.label_key`` -> ``(video_id, segment_id)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import numpy as np

if TYPE_CHECKING:
    from .weak_supervision import Segment, WeakSupervisedTrainer


def refine_labels_exhaustive_pairwise(
    trainer: "WeakSupervisedTrainer",
    segments: List["Segment"],
    labels: Dict,
    verbose: bool = True,
) -> Dict:
    """
    For each epoch, for each video, try all segment pairs with different labels
    and accept swaps using ``trainer._accept_swap(delta_cost, temperature)``.
    """
    labels = labels.copy()

    videos: Dict[str, List] = {}
    for seg in segments:
        if seg.video_id not in videos:
            videos[seg.video_id] = []
        videos[seg.video_id].append(seg)

    if verbose:
        print("\n" + "=" * 60)
        print("ITERATIVE LABEL REFINEMENT (CLIP exhaustive pairwise)")
        print("=" * 60)
        print(f"Segments: {len(segments)}")
        print(f"Videos: {len(videos)}")
        print(f"Epochs: {trainer.ilr_epochs}")
        print(f"Initial temperature: {trainer.initial_temp}")
        print(f"Decay: {trainer.temp_decay} (rate={trainer.decay_rate})")

    centroids = trainer.compute_centroids(segments, labels)
    initial_cost = trainer.compute_total_cost(segments, labels, centroids)

    if verbose:
        print(f"Initial total cost (cosine): {initial_cost:.4f}")

    best_labels = labels.copy()
    best_cost = initial_cost

    for epoch in range(trainer.ilr_epochs):
        temperature = trainer._get_temperature(epoch)

        swaps_made = 0
        swaps_accepted_bad = 0

        for _video_id, video_segments in videos.items():
            if len(video_segments) < 2:
                continue

            for i in range(len(video_segments)):
                for j in range(i + 1, len(video_segments)):
                    seg_a = video_segments[i]
                    seg_b = video_segments[j]

                    if labels[seg_a.label_key] == labels[seg_b.label_key]:
                        continue
                    la = labels[seg_a.label_key]
                    lb = labels[seg_b.label_key]
                    if seg_a.candidate_labels is None or seg_b.candidate_labels is None:
                        continue
                    if la not in seg_b.candidate_labels or lb not in seg_a.candidate_labels:
                        continue
                    if (
                        not getattr(trainer, "ilr_allow_cross_round_swaps", False)
                        and seg_a.candidate_labels != seg_b.candidate_labels
                    ):
                        continue

                    delta_cost = trainer.evaluate_swap(
                        segments, seg_a, seg_b, labels
                    )

                    if trainer._accept_swap(delta_cost, temperature):
                        labels[seg_a.label_key], labels[seg_b.label_key] = (
                            labels[seg_b.label_key],
                            labels[seg_a.label_key],
                        )

                        swaps_made += 1
                        if delta_cost >= 0:
                            swaps_accepted_bad += 1

        centroids = trainer.compute_centroids(segments, labels)
        current_cost = trainer.compute_total_cost(segments, labels, centroids)

        if current_cost < best_cost:
            best_cost = current_cost
            best_labels = labels.copy()

        if verbose and (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch+1:4d}: cost={current_cost:.4f}, "
                f"swaps={swaps_made}, bad_accepted={swaps_accepted_bad}, "
                f"temp={temperature:.4f}"
            )

    if verbose:
        print(f"\nFinal cost: {best_cost:.4f} (improved from {initial_cost:.4f})")
        improvement = (
            (initial_cost - best_cost) / initial_cost * 100 if initial_cost > 0 else 0
        )
        print(f"Improvement: {improvement:.4f}%")

    return best_labels


__all__ = ["refine_labels_exhaustive_pairwise"]
