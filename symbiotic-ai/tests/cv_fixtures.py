"""Synthetic picklists, features, labels, videos, and CLIP-cache files for CV tests."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import cv2
import numpy as np

from cross_validation.videos import resolve_video_path

FRAME_SKIP = 4
N_FRAMES = 40
FPS = 10.0
EMBED_DIM = 8

# Compact opencv0 labels: CARRY_WITH is 1-based [20, 29] -> skip-4 candidates 20, 24, 28.
COMPACT_LABEL_ROWS = [(0, "m"), (6, "a"), (19, "e"), (29, "i")]


def resolve_video_path_safe(video_dir: Path, stem: str) -> Path:
    return resolve_video_path(video_dir, stem)


def write_dummy_video(path: Path, n_frames: int = N_FRAMES, fps: float = FPS, size: int = 32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (size, size),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter for {path}")
    rng = np.random.default_rng(abs(hash(path.name)) % (2**32))
    for _ in range(int(n_frames)):
        frame = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def write_compact_labels(path: Path, rows: Sequence[tuple] = COMPACT_LABEL_ROWS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "code"])
        for frame_index, code in rows:
            w.writerow([int(frame_index), code])


def write_htk_labels(path: Path, rows: Sequence[tuple] = COMPACT_LABEL_ROWS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'"{int(fi)}\t{code}"' for fi, code in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_picklist_json(path: Path, skus: Union[str, Sequence[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(skus, str):
        picklists = [[skus]]
        vid = path.stem.replace("picklist_", "")
    else:
        picklists = [list(skus)]
        vid = path.stem.replace("picklist_", "")
    payload = {"id": vid, "picklists": picklists}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _cache_filename(label: str, frame_number: int) -> str:
    cache_key = hashlib.md5(f"{label}_frame_{frame_number}".encode()).hexdigest()
    return f"{label}_frame_{frame_number}_{cache_key}.npy"


def write_cache_frame(cache_dir: Path, label: str, frame_number: int, embedding: Optional[np.ndarray] = None) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if embedding is None:
        rng = np.random.default_rng(frame_number)
        embedding = rng.normal(size=EMBED_DIM).astype(np.float64)
    dest = cache_dir / _cache_filename(label, int(frame_number))
    np.save(dest, np.asarray(embedding, dtype=np.float64))
    return dest


def write_cache_for_video(
    cache_dir: Path,
    label: str,
    in_range: Sequence[int],
    extra_out_of_range: Optional[Sequence[int]] = None,
    skip_frames: Optional[Sequence[int]] = None,
    embedding_fn=None,
) -> List[Path]:
    """Write first-SKU cache files. ``skip_frames`` are candidate indices left uncached."""
    skip = {int(t) for t in (skip_frames or [])}
    written: List[Path] = []
    for t in list(in_range) + list(extra_out_of_range or []):
        if int(t) in skip:
            continue
        emb = embedding_fn(int(t)) if embedding_fn else None
        written.append(write_cache_frame(cache_dir, label, int(t), emb))
    return written


def write_features_3d(path: Path, n_frames: int = N_FRAMES) -> None:
    """State-patterned 3-D features matching COMPACT_LABEL_ROWS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in range(int(n_frames)):
        if t < 6:
            p_hold, direction, net = 0.05, -1.0, 0.0
        elif t < 19:
            p_hold, direction, net = 0.4, 0.2, 0.3
        elif t < 29:
            p_hold, direction, net = 0.95, 1.0, 1.0
        else:
            p_hold, direction, net = 0.2, -0.2, 0.1
        rows.append(f"{p_hold:.4f} {direction:.4f} {net:.4f}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_ground_truth(path: Path, stem_to_skus: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stems = list(stem_to_skus.keys())
    max_len = max(len(v) for v in stem_to_skus.values())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(stems)
        for i in range(max_len):
            row = []
            for stem in stems:
                skus = stem_to_skus[stem]
                row.append(skus[i] if i < len(skus) else "")
            w.writerow(row)


def sku_embedding(sku: str, frame_number: int, dim: int = EMBED_DIM) -> np.ndarray:
    seed = abs(hash((sku, frame_number))) % (2**32)
    rng = np.random.default_rng(seed)
    base = np.zeros(dim, dtype=np.float64)
    idx = abs(hash(sku)) % dim
    base[idx] = 3.0
    return base + 0.01 * rng.normal(size=dim)


def build_synthetic_dataset(
    root: Path,
    n: int = 4,
    n_segmentation: int = 2,
) -> dict:
    """
    Build a tiny combined-pipeline dataset under *root*.

    Returns a dict of directory paths used by the CV smoke tests.
    """
    root = Path(root)
    videos = root / "videos"
    jsons = root / "jsons"
    csv_labels = root / "csv_labels"
    picklist_labels = root / "picklist_labels"
    features = root / "features"
    cache = root / "cache"
    cv_root = root / "cv_results"
    ground_truth = root / "ground_truth.csv"

    for d in (videos, jsons, csv_labels, picklist_labels, features, cache, cv_root):
        d.mkdir(parents=True, exist_ok=True)

    stems = [f"picklist_{i:03d}" for i in range(1, n + 1)]
    stem_to_skus = {}
    for i, stem in enumerate(stems, start=1):
        sku = f"sku_{i:03d}"
        stem_to_skus[stem] = [sku]
        write_dummy_video(videos / f"{stem}.mp4")
        write_picklist_json(jsons / f"{stem}.json", sku)
        write_compact_labels(picklist_labels / f"{stem}.csv")
        write_features_3d(features / stem)
        if i <= n_segmentation:
            write_htk_labels(csv_labels / f"{stem}.csv")
        write_cache_for_video(
            cache / stem,
            sku,
            in_range=[20, 24, 28],
            extra_out_of_range=[4, 8, 12, 16, 32, 36, 40],
            embedding_fn=lambda t, sku=sku: sku_embedding(sku, t),
        )

    write_ground_truth(ground_truth, stem_to_skus)
    return {
        "root": root,
        "videos": videos,
        "jsons": jsons,
        "csv_labels": csv_labels,
        "picklist_labels": picklist_labels,
        "features": features,
        "cache": cache,
        "cv_root": cv_root,
        "ground_truth": ground_truth,
        "stems": stems,
    }
