"""Deterministic, balanced K-fold assignment shared by every CV stage."""

from __future__ import annotations

import random
from typing import Iterable, List, Sequence, Tuple


def balanced_fold_assignments(
    annealing_ids: Sequence[str],
    segmentation_ids: Sequence[str],
    k_fold: int,
    seed: int,
) -> List[Tuple[List[str], List[str]]]:
    """
    Return ``k_fold`` ``(train_ids, test_ids)`` pairs over *annealing_ids*.

    Segmentation-labelled IDs are round-robin assigned to held-out folds first
    so each fold gets a balanced share; remaining IDs fill the rest. Every
    annealing ID appears in exactly one test fold.
    """
    annealing = list(dict.fromkeys(annealing_ids))
    if k_fold < 2:
        raise ValueError(f"k_fold must be >= 2, got {k_fold}")
    if len(annealing) < k_fold:
        raise ValueError(
            f"Need at least {k_fold} annealing IDs for {k_fold}-fold CV, got {len(annealing)}"
        )

    seg_set = set(segmentation_ids)
    seg = [s for s in annealing if s in seg_set]
    non = [s for s in annealing if s not in seg_set]

    rng = random.Random(int(seed))
    rng.shuffle(seg)
    rng.shuffle(non)

    test_buckets: List[List[str]] = [[] for _ in range(k_fold)]
    for i, stem in enumerate(seg):
        test_buckets[i % k_fold].append(stem)
    for i, stem in enumerate(non):
        test_buckets[i % k_fold].append(stem)

    annealing_set = set(annealing)
    folds: List[Tuple[List[str], List[str]]] = []
    seen_test = []
    for test in test_buckets:
        test_sorted = sorted(test)
        train_sorted = sorted(annealing_set - set(test_sorted))
        if not test_sorted:
            raise ValueError("Fold assignment produced an empty test set")
        if set(train_sorted) & set(test_sorted):
            raise ValueError("Fold assignment leaked train/test IDs")
        seen_test.extend(test_sorted)
        folds.append((train_sorted, test_sorted))

    if sorted(seen_test) != sorted(annealing):
        raise ValueError("Fold assignment does not partition annealing IDs")
    if len(seen_test) != len(set(seen_test)):
        raise ValueError("An annealing ID appears in more than one test fold")
    return folds


def assert_disjoint(train_ids: Iterable[str], test_ids: Iterable[str], *, what: str) -> None:
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise ValueError(f"{what} train/test overlap: {overlap}")
