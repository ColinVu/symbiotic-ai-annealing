"""
6 known items + item_X ablation for weakly supervised ILR (real CLIP cache).

- Routes segments whose candidate_labels do not intersect KNOWN_ITEMS to
  singleton candidate_labels ("item_X",).
- Runs WeakSupervisedTrainer ILR (imported; trainer module is not modified).
- Seeds item_X's *first* refine-step centroid from the spherical mean of
  per-segment spherical means for item_X-labeled segments; subsequent epochs
  use the trainer's normal frame-level spherical centroids.
- Records total cosine cost at every refine step (initial + after each epoch).
- Evaluates accuracy only on segments whose ground-truth label is in KNOWN_ITEMS.

Run from the ``symbiotic-ai`` package root (no CLI args; edit CONFIG below)::

    cd symbiotic-ai
    python -m symbiote_weak_generalized.scripts.synthetic_ilr_itemx_ablation
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from ..experiments.evaluator import _load_ground_truth_labels
from ..pipelines.video_training import (
    _list_videos_in_folder,
    _load_picklists_nested_from_json,
    _process_single_video_from_cache,
)
from ..training.weak_supervision import (
    LabelKey,
    Segment,
    WeakSupervisedTrainer,
    spherical_mean,
)

# ---------------------------------------------------------------------------
# CONFIG — edit paths and hyperparameters here (no command-line arguments).
# ---------------------------------------------------------------------------

KNOWN_ITEMS: List[str] = ["c11", "c12", "c13", "c14", "c15", "c16"]

# Paths (absolute, or relative to the ``symbiotic-ai`` directory).
GROUND_TRUTH_PATH = "./ground_truth.csv"
PICKLIST_PATH = "./hmm-testing/picklist_labels"  # manual timeline CSVs per video stem
PICKLIST_JSON_DIR = "./hmm-testing/picklist_jsons"
VIDEOS_DIR = "./hmm-testing/picklist_videos"
EMBEDDING_CACHE_PATH = "../models/classifier/.cache"  # per-video subdirs: {stem}/

ILR_EPOCHS = 5000
INITIAL_TEMP = 1.0
DECAY_RATE = 0.98
RANDOM_SEED = 42

OUTPUT_DIR = "experiments/itemx_ablation_out"

COMPACT_FRAME_INDEXING = "opencv0"
FRAME_SKIP = 4

ITEM_X_LABEL = "item_X"
VERBOSE = True

# Repository root: ``.../symbiotic-ai`` (parent of ``symbiote_weak_generalized``).
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(p: str | Path) -> Path:
    path = Path(p)
    return path.resolve() if path.is_absolute() else (_REPO_ROOT / path).resolve()


def _load_ground_truth_stems(ground_truth_csv: Path) -> List[str]:
    with ground_truth_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"No header row in ground truth CSV: {ground_truth_csv}")
        stems = [h.strip() for h in reader.fieldnames if h and h.strip()]
    if not stems:
        raise SystemExit(f"No non-empty ground truth columns in: {ground_truth_csv}")
    return stems


def _collect_gt_label_vocab(ground_truth_csv: Path) -> Set[str]:
    vocab: Set[str] = set()
    with ground_truth_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for v in row.values():
                s = (v or "").strip()
                if s:
                    vocab.add(s)
    return vocab


def _discover_runnable_stems(
    gt_stems: List[str],
    json_dir: Path,
    videos_dir: Path,
    cache_root: Path,
) -> List[str]:
    video_by_stem = {Path(p).stem: p for p in _list_videos_in_folder(str(videos_dir))}
    out: List[str] = []
    for stem in gt_stems:
        if stem not in video_by_stem:
            continue
        jp = json_dir / f"{stem}.json"
        if not jp.is_file():
            continue
        cd = cache_root / stem
        if not cd.is_dir():
            continue
        out.append(stem)
    return out


def _route_candidate_labels_to_item_x(segments: List[Segment], known: Set[str]) -> None:
    """Mutate segments: non-overlapping picklists -> singleton (item_X,)."""
    for seg in segments:
        cand = seg.candidate_labels
        if cand is None:
            continue
        if set(cand) & known:
            continue
        seg.candidate_labels = (ITEM_X_LABEL,)


def _item_x_segment_mean_directions(
    segments: List[Segment],
    labels: Dict[LabelKey, str],
) -> np.ndarray:
    """Stack per-segment spherical means for segments currently labeled item_X."""
    rows: List[np.ndarray] = []
    for seg in segments:
        if seg.is_placeholder:
            continue
        if labels.get(seg.label_key) != ITEM_X_LABEL:
            continue
        em = np.asarray(seg.embeddings, dtype=np.float64)
        if em.size == 0:
            continue
        rows.append(spherical_mean(em))
    if not rows:
        return np.zeros((0,), dtype=np.float64)
    return np.stack(rows, axis=0)


class ItemXAblationTrainer(WeakSupervisedTrainer):
    """
    Tracks per-refine-step cosine cost; seeds item_X centroid on first
    compute_centroids call inside refine_labels only.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cost_trajectory: List[float] = []
        self._in_refine: bool = False
        self._item_x_centroid_seeded: bool = False

    def _initialize_labels_for_segments(
        self,
        segments: List[Segment],
        flat_picklist: Optional[List[str]],
    ) -> Dict[LabelKey, str]:
        """
        Relax base group-size invariant only for singleton ``('item_X',)`` groups:
        assign item_X to every segment in that group.
        """
        from collections import defaultdict

        labels: Dict[LabelKey, str] = {}
        groups: Dict[Tuple[str, Tuple[str, ...]], List[Segment]] = defaultdict(list)
        for seg in segments:
            tup = self._candidate_multiset_tuple(seg, flat_picklist)
            groups[(seg.video_id, tup)].append(seg)

        for (_vid, multiset), segs in groups.items():
            segs_sorted = sorted(segs, key=lambda s: s.segment_id)
            if multiset == (ITEM_X_LABEL,):
                for seg in segs_sorted:
                    labels[seg.label_key] = ITEM_X_LABEL
                continue

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

    def refine_labels(
        self,
        segments: List[Segment],
        labels: Dict[LabelKey, str],
        verbose: bool = True,
    ) -> Dict[LabelKey, str]:
        self.cost_trajectory = []
        self._item_x_centroid_seeded = False
        self._in_refine = True
        try:
            return super().refine_labels(segments, labels, verbose=verbose)
        finally:
            self._in_refine = False

    def compute_centroids(
        self,
        segments: List[Segment],
        labels: Dict[LabelKey, str],
    ) -> Dict[str, np.ndarray]:
        out = super().compute_centroids(segments, labels)
        if (
            self._in_refine
            and not self._item_x_centroid_seeded
            and ITEM_X_LABEL in out
        ):
            mats = _item_x_segment_mean_directions(segments, labels)
            if mats.size > 0 and mats.ndim == 2:
                seeded = spherical_mean(mats)
                if float(np.linalg.norm(seeded)) > 1e-12:
                    out[ITEM_X_LABEL] = seeded
            self._item_x_centroid_seeded = True
        return out

    def compute_total_cosine_cost(
        self,
        segments: List[Segment],
        labels: Dict[LabelKey, str],
        centroids: Dict[str, np.ndarray],
    ) -> float:
        total = super().compute_total_cosine_cost(segments, labels, centroids)
        if self._in_refine:
            self.cost_trajectory.append(float(total))
        return total


def _evaluate_known_items_only(
    refined: Dict[LabelKey, str],
    ground_truth_csv: Path,
    video_stems: List[str],
    known_items: Set[str],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    """
    Returns:
        per_item: label -> {gt_count, correct}
        totals: gt_count, correct across all known items
    """
    per_item: Dict[str, Dict[str, int]] = {
        k: {"gt_count": 0, "correct": 0} for k in sorted(known_items)
    }
    totals = {"gt_count": 0, "correct": 0}
    gt_path = str(ground_truth_csv.resolve())

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
            if exp not in known_items:
                continue
            pred = pred_by_seg.get(i, "")
            per_item[exp]["gt_count"] += 1
            totals["gt_count"] += 1
            if pred == exp:
                per_item[exp]["correct"] += 1
                totals["correct"] += 1
    return per_item, totals


def _print_results_table(
    per_item: Dict[str, Dict[str, int]],
    totals: Dict[str, int],
    known_order: List[str],
) -> None:
    print("\n" + "=" * 72)
    print("KNOWN-ITEM ACCURACY (item_X segments excluded from evaluation)")
    print("=" * 72)
    print(f"{'item':<10} {'gt_count':>10} {'correct':>10} {'accuracy':>12}")
    print("-" * 72)
    for lab in known_order:
        row = per_item[lab]
        n = row["gt_count"]
        c = row["correct"]
        acc = (c / n) if n else float("nan")
        print(f"{lab:<10} {n:>10} {c:>10} {acc:>12.4f}")
    print("-" * 72)
    n_tot = totals["gt_count"]
    c_tot = totals["correct"]
    overall = (c_tot / n_tot) if n_tot else float("nan")
    print(f"{'OVERALL':<10} {n_tot:>10} {c_tot:>10} {overall:>12.4f}")
    print("=" * 72)


def main() -> None:
    if len(KNOWN_ITEMS) != 6:
        raise SystemExit(f"KNOWN_ITEMS must have length 6, got {len(KNOWN_ITEMS)}")

    known_set = set(str(x).strip() for x in KNOWN_ITEMS)
    if ITEM_X_LABEL in known_set:
        raise SystemExit(f"{ITEM_X_LABEL!r} must not appear in KNOWN_ITEMS")

    gt_path = _resolve_path(GROUND_TRUTH_PATH)
    picklist_labels_dir = _resolve_path(PICKLIST_PATH)
    json_dir = _resolve_path(PICKLIST_JSON_DIR)
    videos_dir = _resolve_path(VIDEOS_DIR)
    cache_root = _resolve_path(EMBEDDING_CACHE_PATH)
    out_dir = _resolve_path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    gt_stems = _load_ground_truth_stems(gt_path)
    vocab = _collect_gt_label_vocab(gt_path)
    missing_vocab = [k for k in KNOWN_ITEMS if k not in vocab]
    if missing_vocab and VERBOSE:
        print(
            "Warning: these KNOWN_ITEMS do not appear in any ground-truth cell — "
            f"evaluation counts may be zero: {missing_vocab}",
            file=sys.stderr,
        )

    stem_order = _discover_runnable_stems(gt_stems, json_dir, videos_dir, cache_root)
    if not stem_order:
        raise SystemExit(
            "No runnable videos: need overlapping ground-truth column, "
            f".json under {json_dir}, .mp4 under {videos_dir}, cache dir under {cache_root}/<stem>/"
        )

    video_by_stem = {Path(p).stem: p for p in _list_videos_in_folder(str(videos_dir))}
    video_segments: Dict[str, Tuple[List[Segment], List[str]]] = {}

    for stem in stem_order:
        json_path = str(json_dir / f"{stem}.json")
        picklists_nested = _load_picklists_nested_from_json(json_path)
        per_video_cache = str(cache_root / stem)
        video_path = video_by_stem[stem]
        segments, flat_picklist, video_name = _process_single_video_from_cache(
            video_path,
            picklists_nested,
            per_video_cache,
            str(picklist_labels_dir),
            require_manual_label_csv=True,
            compact_frame_indexing=COMPACT_FRAME_INDEXING,
            frame_skip=FRAME_SKIP,
            verbose=VERBOSE,
        )
        _route_candidate_labels_to_item_x(segments, known_set)
        video_segments[video_name] = (segments, flat_picklist)

    n_item_x_segments = sum(
        1
        for _vid, (segs, _fp) in video_segments.items()
        for s in segs
        if s.candidate_labels == (ITEM_X_LABEL,)
    )
    if n_item_x_segments == 0 and VERBOSE:
        print(
            "Warning: no segments routed to item_X (every segment intersects KNOWN_ITEMS).",
            file=sys.stderr,
        )

    trainer = ItemXAblationTrainer(
        ilr_epochs=int(ILR_EPOCHS),
        initial_temp=float(INITIAL_TEMP),
        temp_decay="exponential",
        decay_rate=float(DECAY_RATE),
        random_seed=int(RANDOM_SEED),
        ilr_allow_cross_round_swaps=False,
    )

    trainer.fit(
        video_segments,
        verbose=VERBOSE,
        skip_ilr=False,
        initial_cluster_voting_csv=None,
        use_cluster_voting=False,
    )

    refined = trainer.last_refined_labels or {}
    per_item, totals = _evaluate_known_items_only(
        refined,
        gt_path,
        stem_order,
        known_set,
    )
    _print_results_table(per_item, totals, list(KNOWN_ITEMS))

    # Cost plot (primary diagnostic)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required to save cost_trajectory.png. "
            f"Install matplotlib or inspect cost_trajectory.json. ({e})"
        ) from e

    costs = list(trainer.cost_trajectory)
    steps = np.arange(len(costs), dtype=np.int32)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, costs, color="C0", linewidth=1.0)
    ax.set_xlabel("refine step (0 = initial cost, k = after epoch k-1)")
    ax.set_ylabel("total cosine cost")
    ax.set_title("ILR cosine cost trajectory (6 known + item_X)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = out_dir / "cost_trajectory.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    metrics_path = out_dir / "itemx_ablation_metrics.json"
    payload: Dict[str, Any] = {
        "known_items": list(KNOWN_ITEMS),
        "video_stems": stem_order,
        "ilr_epochs": ILR_EPOCHS,
        "initial_temp": INITIAL_TEMP,
        "decay_rate": DECAY_RATE,
        "random_seed": RANDOM_SEED,
        "n_item_x_segments": n_item_x_segments,
        "per_item": {
            k: {
                "ground_truth_count": per_item[k]["gt_count"],
                "correct": per_item[k]["correct"],
                "accuracy": (
                    (per_item[k]["correct"] / per_item[k]["gt_count"])
                    if per_item[k]["gt_count"]
                    else None
                ),
            }
            for k in KNOWN_ITEMS
        },
        "overall": {
            "ground_truth_count": totals["gt_count"],
            "correct": totals["correct"],
            "accuracy": (
                (totals["correct"] / totals["gt_count"]) if totals["gt_count"] else None
            ),
        },
        "cost_trajectory": costs,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if VERBOSE:
        print(f"\nWrote plot: {plot_path}")
        print(f"Wrote metrics: {metrics_path}")
        if costs:
            print(
                f"Cost trajectory length: {len(costs)} "
                f"(expected {ILR_EPOCHS + 1} = 1 initial + {ILR_EPOCHS} epoch ends)"
            )


if __name__ == "__main__":
    main()
