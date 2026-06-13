"""Per-feature reliability report for HTK state detection features.

Generates a ranked table of how informative each individual feature is for
predicting the 4-state label sequence on a labeled dataset.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from symbiote.state_detection.aruco_detection import ArucoDetector  # noqa: E402
from symbiote.state_detection.config import DEFAULT_HTK_CONFIG, STATE_CYCLE  # noqa: E402
from symbiote.state_detection.feature_extraction import FeatureExtractor  # noqa: E402

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
_TASK_MODES = {"full4", "coarse", "interact", "carry"}
_FEATURE_NAMES = [
    "hand_center_x",
    "hand_center_y",
    "velocity_x",
    "velocity_y",
    "accel_x",
    "accel_y",
    "bbox_width",
    "bbox_height",
    "bbox_dwidth",
    "bbox_dheight",
    "orientation_x",
    "orientation_y",
    "orientation_z",
    "object_confidence",
    "aruco_bin_context",
]


def _discover_pairs(video_dir: str, label_dir: str) -> List[Tuple[str, str]]:
    vd = Path(video_dir)
    ld = Path(label_dir)
    videos = {p.stem: p for p in vd.iterdir() if p.suffix.lower() in _VIDEO_EXTS}
    pairs: List[Tuple[str, str]] = []
    for stem, vp in sorted(videos.items()):
        lp = ld / f"{stem}.csv"
        if lp.is_file():
            pairs.append((str(vp), str(lp)))
    if not pairs:
        raise RuntimeError("No matched video/label pairs found.")
    return pairs


def _load_label_segments(label_csv: str) -> List[Tuple[float, float, str]]:
    out: List[Tuple[float, float, str]] = []
    with open(label_csv, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            out.append((float(row["timestamp_start"]), float(row["timestamp_end"]), row["state"].strip()))
    return out


def _state_at(t: float, segs: List[Tuple[float, float, str]]) -> Optional[str]:
    for s, e, st in segs:
        if s <= t <= e:
            return st
    return None


def _map_state_for_task(state: str, task_mode: str) -> Optional[str]:
    if task_mode == "full4":
        return state if state in {"PICK", "CARRY_WITH", "PLACE", "CARRY_EMPTY"} else None
    if task_mode == "coarse":
        if state in {"PICK", "PLACE"}:
            return "INTERACT"
        if state in {"CARRY_WITH", "CARRY_EMPTY"}:
            return "CARRY"
        return None
    if task_mode == "interact":
        return state if state in {"PICK", "PLACE"} else None
    if task_mode == "carry":
        return state if state in {"CARRY_WITH", "CARRY_EMPTY"} else None
    raise ValueError(f"Unknown task_mode: {task_mode}")


def _cache_key(video_path: str, frame_skip: int, blur_threshold: float) -> dict:
    return {
        "mtime": os.path.getmtime(video_path),
        "frame_skip": frame_skip,
        "blur_threshold": blur_threshold,
        # Keep cache key compatible with training cache manifest entries.
        "feature_mask": None,
    }


def _load_manifest(manifest_path: str) -> dict:
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r") as f:
            return json.load(f)
    return {}


def _save_manifest(manifest_path: str, manifest: dict) -> None:
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def _load_cached_features(
    cache_dir: str,
    manifest: dict,
    video_path: str,
    frame_skip: int,
    blur_threshold: float,
) -> Optional[np.ndarray]:
    stem = Path(video_path).stem
    if stem not in manifest:
        return None
    key = _cache_key(video_path, frame_skip, blur_threshold)
    current = manifest[stem]
    # Backward compatibility for old entries without feature_mask.
    if isinstance(current, dict) and "feature_mask" not in current:
        current = dict(current)
        current["feature_mask"] = None
    if current != key:
        return None
    npy = os.path.join(cache_dir, f"{stem}.npy")
    if not os.path.isfile(npy):
        return None
    return np.load(npy)


def _save_cached_features(
    cache_dir: str,
    manifest: dict,
    video_path: str,
    features: np.ndarray,
    frame_skip: int,
    blur_threshold: float,
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    stem = Path(video_path).stem
    npy = os.path.join(cache_dir, f"{stem}.npy")
    np.save(npy, features)
    manifest[stem] = _cache_key(video_path, frame_skip, blur_threshold)


def _get_frame_numbers_and_fps(video_path: str, frame_skip: int, n_rows: int) -> Tuple[List[int], float]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0:
        fps = 30.0
    frames = [i for i in range(1, total + 1) if i % frame_skip == 0]
    if len(frames) > n_rows:
        frames = frames[:n_rows]
    elif len(frames) < n_rows:
        if len(frames) == 0:
            frames = list(range(1, n_rows + 1))
        else:
            last = frames[-1]
            while len(frames) < n_rows:
                last += frame_skip
                frames.append(last)
    return frames, fps


def _cv_scores_single_feature(x: np.ndarray, y: np.ndarray, seed: int = 7) -> Tuple[float, float]:
    # Constant feature: classifier has no signal; return majority-baseline metrics.
    if np.allclose(np.std(x), 0.0):
        majority = int(np.bincount(y).argmax())
        pred = np.full_like(y, majority)
        return float(accuracy_score(y, pred)), float(f1_score(y, pred, average="macro"))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs: List[float] = []
    f1s: List[float] = []
    for tr, te in skf.split(x, y):
        # Keep constructor minimal for broad sklearn compatibility.
        clf = LogisticRegression(max_iter=1000)
        clf.fit(x[tr], y[tr])
        pred = clf.predict(x[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))
    return float(np.mean(accs)), float(np.mean(f1s))


def build_report(
    video_dir: str,
    label_dir: str,
    output_dir: str,
    aruco_config: Optional[str],
    frame_skip: int,
    blur_threshold: float,
    task_mode: str = "full4",
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    if task_mode not in _TASK_MODES:
        raise ValueError(f"task_mode must be one of {sorted(_TASK_MODES)}")
    pairs = _discover_pairs(video_dir, label_dir)
    if verbose:
        print(f"[hmm_feature_report] pairs: {len(pairs)}")

    aruco = ArucoDetector(
        aruco_dict_type=DEFAULT_HTK_CONFIG.aruco_dict_type,
        distance_decay=DEFAULT_HTK_CONFIG.aruco_distance_decay,
    )
    if aruco_config and os.path.isfile(aruco_config):
        aruco.load_bin_config(aruco_config)
    extractor = FeatureExtractor(aruco_detector=aruco)

    cache_dir = os.path.join(output_dir, "feature_cache")
    manifest_path = os.path.join(cache_dir, "manifest.json")
    manifest = _load_manifest(manifest_path)

    X_all: List[np.ndarray] = []
    y_all: List[int] = []

    mapped_labels_order = []
    for s in STATE_CYCLE:
        mapped = _map_state_for_task(s, task_mode)
        if mapped is not None and mapped not in mapped_labels_order:
            mapped_labels_order.append(mapped)
    label_to_idx = {s: i for i, s in enumerate(mapped_labels_order)}
    for vp, lp in pairs:
        feats = None
        if use_cache:
            feats = _load_cached_features(cache_dir, manifest, vp, frame_skip, blur_threshold)
            if feats is not None and verbose:
                print(f"[hmm_feature_report] cache hit: {Path(vp).name} ({feats.shape[0]})")
        if feats is None:
            feats, frame_numbers, fps = extractor.extract_video_features(
                vp, frame_skip=frame_skip, blur_threshold=blur_threshold, verbose=verbose
            )
            if use_cache and feats.shape[0] > 0:
                _save_cached_features(
                    cache_dir=cache_dir,
                    manifest=manifest,
                    video_path=vp,
                    features=feats,
                    frame_skip=frame_skip,
                    blur_threshold=blur_threshold,
                )
        else:
            frame_numbers, fps = _get_frame_numbers_and_fps(vp, frame_skip, feats.shape[0])

        if feats.shape[0] == 0:
            continue
        labels = _load_label_segments(lp)
        y_rows: List[int] = []
        x_rows: List[np.ndarray] = []
        for idx, fn in enumerate(frame_numbers):
            t = fn / fps if fps > 0 else 0.0
            st = _state_at(t, labels)
            mapped = None if st is None else _map_state_for_task(st, task_mode)
            if mapped is None or mapped not in label_to_idx:
                continue
            y_rows.append(label_to_idx[mapped])
            x_rows.append(feats[idx])
        if x_rows:
            X_all.append(np.vstack(x_rows))
            y_all.extend(y_rows)
    if use_cache:
        _save_manifest(manifest_path, manifest)

    if not X_all:
        raise RuntimeError("No aligned feature/label rows found.")
    X = np.vstack(X_all)
    y = np.array(y_all, dtype=np.int32)
    if len(np.unique(y)) < 2:
        raise RuntimeError(
            f"Need at least 2 classes after filtering for task_mode='{task_mode}'. "
            "Try a different task mode or larger dataset."
        )

    # Filter constant columns to avoid degenerate stats.
    f_vals, _ = f_classif(X, y)
    # Replace invalid scores from constant/degenerate features.
    f_vals = np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0)
    mi_vals = mutual_info_classif(X, y, random_state=7)
    mi_vals = np.nan_to_num(mi_vals, nan=0.0, posinf=0.0, neginf=0.0)

    rows: List[Dict] = []
    for i, name in enumerate(_FEATURE_NAMES):
        x1 = X[:, [i]]
        acc, f1m = _cv_scores_single_feature(x1, y)
        rows.append(
            {
                "feature_idx": i,
                "feature": name,
                "cv_accuracy": acc,
                "cv_macro_f1": f1m,
                "anova_f": float(f_vals[i]) if np.isfinite(f_vals[i]) else 0.0,
                "mutual_info": float(mi_vals[i]),
                "mean": float(np.mean(X[:, i])),
                "std": float(np.std(X[:, i])),
                "task_mode": task_mode,
            }
        )

    df = pd.DataFrame(rows).sort_values(
        by=["cv_macro_f1", "mutual_info", "anova_f"], ascending=False
    ).reset_index(drop=True)

    final_dir = os.path.join(output_dir, "models", "hmm_final")
    if not os.path.isdir(final_dir):
        final_dir = output_dir
    suffix = "" if task_mode == "full4" else f"_{task_mode}"
    out_csv = os.path.join(final_dir, f"feature_reliability{suffix}.csv")
    df.to_csv(out_csv, index=False)
    if verbose:
        print(f"[hmm_feature_report] wrote: {out_csv}")
        print(df.head(15).to_string(index=False))
    return df


def suggest_feature_mask(df: pd.DataFrame, top_k: int) -> Dict:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    selected = [int(v) for v in df["feature_idx"].head(top_k).tolist()]
    selected = sorted(set(selected))
    return {
        "top_k": top_k,
        "selected_indices": selected,
        "selected_features": [
            _FEATURE_NAMES[i] if 0 <= i < len(_FEATURE_NAMES) else f"feature_{i}"
            for i in selected
        ],
        "mask_mode": "zero_out_unselected",
        "ranking_metric": "cv_macro_f1",
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m symbiote.hmm_feature_report",
        description="Rank each feature's reliability for state prediction.",
    )
    p.add_argument("--video-dir", default=str(_ROOT / "hmm-testing" / "picklist_videos"))
    p.add_argument("--label-dir", default=str(_ROOT / "hmm-testing" / "picklist_labels"))
    p.add_argument("--output-dir", default=str(_ROOT / "models" / "htk"))
    p.add_argument("--aruco-config", default=str(_ROOT / "config" / "aruco_bins.json"))
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--threshold", type=float, default=100.0)
    p.add_argument(
        "--task-mode",
        choices=["full4", "coarse", "interact", "carry"],
        default="full4",
        help="Label space for feature reliability scoring.",
    )
    p.add_argument("--suggest-top-k", type=int, default=0, help="Write top-K mask suggestion JSON.")
    p.add_argument("--no-cache", action="store_true", help="Ignore feature cache and re-extract")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    aruco_config = args.aruco_config if os.path.isfile(args.aruco_config) else None
    df = build_report(
        video_dir=args.video_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        aruco_config=aruco_config,
        frame_skip=args.frame_skip,
        blur_threshold=args.threshold,
        task_mode=args.task_mode,
        use_cache=not args.no_cache,
        verbose=not args.quiet,
    )
    if args.suggest_top_k > 0:
        suggestion = suggest_feature_mask(df, args.suggest_top_k)
        final_dir = os.path.join(args.output_dir, "models", "hmm_final")
        if not os.path.isdir(final_dir):
            final_dir = args.output_dir
        out_json = os.path.join(final_dir, f"feature_mask_top{args.suggest_top_k}.json")
        import json
        with open(out_json, "w") as f:
            json.dump(suggestion, f, indent=2)
        if not args.quiet:
            print(f"[hmm_feature_report] wrote mask suggestion: {out_json}")
            print(f"[hmm_feature_report] selected idx: {suggestion['selected_indices']}")


if __name__ == "__main__":
    main()

