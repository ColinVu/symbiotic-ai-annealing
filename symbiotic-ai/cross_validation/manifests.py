"""Immutable dataset / fold manifests shared by every CV stage."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

from .layout import CvLayout
from .splits import assert_disjoint, balanced_fold_assignments
from .stems import id_from_stem, normalize_stem

_PICKLIST_FILE = re.compile(r"picklist_(\d+)$", re.I)
_VIDEO_EXTS = {".mp4", ".mov", ".m4v"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_stems(directory: Optional[Union[str, Path]], suffixes: Sequence[str]) -> List[str]:
    if directory is None:
        return []
    root = Path(directory)
    if not root.is_dir():
        return []
    found: List[str] = []
    wanted = {s.lower() for s in suffixes}
    for path in root.iterdir():
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in wanted:
            continue
        if not _PICKLIST_FILE.match(path.stem):
            continue
        found.append(normalize_stem(path.stem))
    return sorted(set(found), key=lambda s: int(id_from_stem(s)))


def _discover_feature_stems(features_dir: Optional[Union[str, Path]]) -> List[str]:
    if features_dir is None:
        return []
    root = Path(features_dir)
    if not root.is_dir():
        return []
    found: List[str] = []
    for path in root.iterdir():
        if not path.is_file() or path.name.startswith("."):
            continue
        if not _PICKLIST_FILE.match(path.name):
            continue
        found.append(normalize_stem(path.name))
    return sorted(set(found), key=lambda s: int(id_from_stem(s)))


def _discover_label_stems(labels_dir: Optional[Union[str, Path]]) -> List[str]:
    stems = _discover_stems(labels_dir, [".csv"])
    if labels_dir is None:
        return stems
    root = Path(labels_dir)
    kept: List[str] = []
    for stem in stems:
        text = (root / f"{stem}.csv").read_text(encoding="utf-8", errors="replace")
        if any(line.strip() for line in text.splitlines()):
            kept.append(stem)
    return kept


def _first_sku(json_path: Path) -> Optional[str]:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    picklists = payload.get("picklists") or []
    if not picklists:
        return None
    first = picklists[0]
    if isinstance(first, str):
        return first
    if isinstance(first, list) and first:
        return str(first[0])
    return None


def _fingerprint(
    *,
    k_fold: int,
    seed: int,
    annealing_ids: Sequence[str],
    segmentation_ids: Sequence[str],
    paths: Dict[str, Optional[str]],
) -> str:
    payload = {
        "k_fold": int(k_fold),
        "seed": int(seed),
        "annealing_ids": list(annealing_ids),
        "segmentation_ids": list(segmentation_ids),
        "paths": paths,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(blob).hexdigest()


def discover_dataset_ids(
    *,
    segmentation_labels_dir: Union[str, Path],
    videos_dir: Union[str, Path],
    json_dir: Union[str, Path],
    features_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, List[str]]:
    """IDs used to build folds. ``ground_truth.csv`` is not consulted."""
    json_stems = _discover_stems(json_dir, [".json"])
    video_stems = _discover_stems(videos_dir, list(_VIDEO_EXTS))
    annealing = set(json_stems) & set(video_stems)
    if not annealing:
        raise ValueError(
            "No shared annealing IDs found. Need matching picklist JSONs and videos."
        )

    annealing_ids = sorted(annealing, key=lambda s: int(id_from_stem(s)))
    seg = set(_discover_label_stems(segmentation_labels_dir)) & set(annealing_ids)
    feat = _discover_feature_stems(features_dir)
    if feat:
        seg &= set(feat)
    segmentation_ids = sorted(seg, key=lambda s: int(id_from_stem(s)))
    return {
        "annealing_ids": annealing_ids,
        "segmentation_ids": segmentation_ids,
        "video_ids": sorted(video_stems, key=lambda s: int(id_from_stem(s))) if video_stems else [],
        "json_ids": sorted(json_stems, key=lambda s: int(id_from_stem(s))) if json_stems else [],
    }


def _resolved(path: Optional[Union[str, Path]]) -> Optional[str]:
    if path is None:
        return None
    return str(Path(path).resolve())


def create_or_load_dataset(
    *,
    cv_root: Union[str, Path],
    k_fold: int,
    seed: int,
    segmentation_labels_dir: Union[str, Path],
    videos_dir: Union[str, Path],
    json_dir: Union[str, Path],
    features_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Create immutable dataset + fold manifests, or reuse them after validation.

    Later stages must pass the same ``k_fold`` and dataset fingerprint; they
    never resplit.
    """
    layout = CvLayout(cv_root)
    layout.root.mkdir(parents=True, exist_ok=True)

    discovered = discover_dataset_ids(
        segmentation_labels_dir=segmentation_labels_dir,
        videos_dir=videos_dir,
        json_dir=json_dir,
        features_dir=features_dir,
    )
    annealing_ids = discovered["annealing_ids"]
    segmentation_ids = discovered["segmentation_ids"]
    paths = {
        "segmentation_labels_dir": _resolved(segmentation_labels_dir),
        "videos_dir": _resolved(videos_dir),
        "json_dir": _resolved(json_dir),
        "features_dir": _resolved(features_dir),
    }
    fingerprint = _fingerprint(
        k_fold=k_fold,
        seed=seed,
        annealing_ids=annealing_ids,
        segmentation_ids=segmentation_ids,
        paths=paths,
    )

    if layout.dataset_manifest.is_file() and not overwrite:
        existing = _load(layout.dataset_manifest)
        _validate_existing(
            existing,
            k_fold=k_fold,
            fingerprint=fingerprint,
            annealing_ids=annealing_ids,
            segmentation_ids=segmentation_ids,
        )
        for fold_manifest in iter_fold_manifests(layout.root):
            _validate_fold_manifest(fold_manifest, annealing_ids, segmentation_ids, k_fold)
        return existing

    assignments = balanced_fold_assignments(annealing_ids, segmentation_ids, k_fold, seed)
    first_skus: Dict[str, str] = {}
    if json_dir is not None:
        json_root = Path(json_dir)
        for stem in annealing_ids:
            sku = _first_sku(json_root / f"{stem}.json")
            if sku:
                first_skus[stem] = sku

    dataset = {
        "k_fold": int(k_fold),
        "seed": int(seed),
        "fingerprint": fingerprint,
        "annealing_ids": annealing_ids,
        "segmentation_ids": segmentation_ids,
        "paths": paths,
        "first_skus": first_skus,
        "created_at": _now(),
    }
    _dump(layout.dataset_manifest, dataset)

    seg_set = set(segmentation_ids)
    for i, (train_ids, test_ids) in enumerate(assignments, start=1):
        assert_disjoint(train_ids, test_ids, what="annealing")
        fold_payload = {
            "fold": i,
            "k_fold": int(k_fold),
            "seed": int(seed),
            "fingerprint": fingerprint,
            "annealing_train_ids": train_ids,
            "annealing_test_ids": test_ids,
            "segmentation_train_ids": [s for s in train_ids if s in seg_set],
            "segmentation_test_ids": [s for s in test_ids if s in seg_set],
            "first_skus": {s: first_skus[s] for s in train_ids + test_ids if s in first_skus},
        }
        assert_disjoint(
            fold_payload["segmentation_train_ids"],
            fold_payload["segmentation_test_ids"],
            what="segmentation",
        )
        _dump(layout.fold(i).fold_manifest, fold_payload)

    return dataset


def _validate_existing(
    existing: Dict[str, Any],
    *,
    k_fold: int,
    fingerprint: str,
    annealing_ids: Sequence[str],
    segmentation_ids: Sequence[str],
) -> None:
    if int(existing.get("k_fold", -1)) != int(k_fold):
        raise ValueError(
            f"Existing dataset_manifest k_fold={existing.get('k_fold')!r} "
            f"does not match requested k_fold={k_fold}. Pass overwrite=True to resplit."
        )
    if existing.get("fingerprint") != fingerprint:
        raise ValueError(
            "Existing dataset_manifest fingerprint does not match the current "
            "dataset (IDs, paths, K, or seed changed). Pass overwrite=True to resplit."
        )
    if list(existing.get("annealing_ids") or []) != list(annealing_ids):
        raise ValueError("Existing annealing_ids do not match the current dataset.")
    if list(existing.get("segmentation_ids") or []) != list(segmentation_ids):
        raise ValueError("Existing segmentation_ids do not match the current dataset.")


def _validate_fold_manifest(
    fold: Dict[str, Any],
    annealing_ids: Sequence[str],
    segmentation_ids: Sequence[str],
    k_fold: int,
) -> None:
    if int(fold.get("k_fold", -1)) != int(k_fold):
        raise ValueError(f"Fold {fold.get('fold')} k_fold mismatch")
    train = list(fold.get("annealing_train_ids") or [])
    test = list(fold.get("annealing_test_ids") or [])
    assert_disjoint(train, test, what=f"fold {fold.get('fold')} annealing")
    if sorted(set(train) | set(test)) != sorted(annealing_ids):
        raise ValueError(f"Fold {fold.get('fold')} annealing IDs do not cover the dataset")
    seg_set = set(segmentation_ids)
    expected_seg_train = sorted(s for s in train if s in seg_set)
    expected_seg_test = sorted(s for s in test if s in seg_set)
    if sorted(fold.get("segmentation_train_ids") or []) != expected_seg_train:
        raise ValueError(f"Fold {fold.get('fold')} segmentation_train_ids mismatch")
    if sorted(fold.get("segmentation_test_ids") or []) != expected_seg_test:
        raise ValueError(f"Fold {fold.get('fold')} segmentation_test_ids mismatch")


def load_dataset_manifest(cv_root: Union[str, Path]) -> Dict[str, Any]:
    path = CvLayout(cv_root).dataset_manifest
    if not path.is_file():
        raise FileNotFoundError(f"dataset_manifest.json not found: {path}")
    return _load(path)


def require_existing_dataset(
    cv_root: Union[str, Path],
    k_fold: int,
    expected_paths: Optional[Dict[str, Optional[Union[str, Path]]]] = None,
) -> Dict[str, Any]:
    """Load an already-created CV dataset; never resplit."""
    dataset = load_dataset_manifest(cv_root)
    if int(dataset.get("k_fold", -1)) != int(k_fold):
        raise SystemExit(
            f"Existing dataset_manifest k_fold={dataset.get('k_fold')!r} "
            f"does not match requested --k-fold={k_fold}."
        )
    if expected_paths:
        recorded = dataset.get("paths") or {}
        for key, value in expected_paths.items():
            if value is None:
                continue
            requested = _resolved(value)
            if recorded.get(key) != requested:
                raise ValueError(
                    f"Existing dataset_manifest path {key}={recorded.get(key)!r} "
                    f"does not match requested {requested!r}."
                )
    for fold_manifest in iter_fold_manifests(cv_root):
        _validate_fold_manifest(
            fold_manifest,
            list(dataset.get("annealing_ids") or []),
            list(dataset.get("segmentation_ids") or []),
            int(k_fold),
        )
    return dataset


def load_fold_manifest(cv_root: Union[str, Path], fold: int) -> Dict[str, Any]:
    path = CvLayout(cv_root).fold(int(fold)).fold_manifest
    if not path.is_file():
        raise FileNotFoundError(f"fold_manifest.json not found: {path}")
    return _load(path)


def iter_fold_manifests(cv_root: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    layout = CvLayout(cv_root)
    dataset = load_dataset_manifest(layout.root)
    k = int(dataset["k_fold"])
    for i in range(1, k + 1):
        yield load_fold_manifest(layout.root, i)


def require_cv_pair(k_fold: Optional[int], cv_results_dir: Optional[Union[str, Path]]) -> bool:
    """Return True when CV mode is on. Require both flags together."""
    if (k_fold is None) ^ (cv_results_dir is None):
        raise SystemExit("--k-fold and --cv-results-dir must be passed together")
    if k_fold is not None and int(k_fold) < 2:
        raise SystemExit(f"--k-fold must be >= 2, got {k_fold}")
    return k_fold is not None
