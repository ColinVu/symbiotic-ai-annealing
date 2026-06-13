"""Tune HVite decode parameters for HTK state inference.

This script sweeps `(word_penalty, grammar_scale)` on a labeled dev set and
writes the best params to:

    <model_dir>/models/hmm_final/infer_params.json

`hmm_infer.py` auto-loads this file when present.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from symbiote.state_detection.aruco_detection import ArucoDetector  # noqa: E402
from symbiote.state_detection.config import DEFAULT_HTK_CONFIG  # noqa: E402
from symbiote.state_detection.feature_extraction import FeatureExtractor  # noqa: E402
from symbiote.state_detection.htk_interface import HTKStateDetector  # noqa: E402
from symbiote.state_detection.two_stage import (  # noqa: E402
    CARRY_STATES,
    COARSE_STATES,
    INTERACT_STATES,
    apply_feature_mask,
    decode_subtype_with_runs,
    frame_labels_to_segments,
    map_fine_to_coarse,
    segments_to_frame_labels,
)


_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def _discover_pairs(video_dir: str, label_dir: str) -> List[Tuple[str, str]]:
    vd = Path(video_dir)
    ld = Path(label_dir)
    if not vd.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    if not ld.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    videos = {p.stem: p for p in vd.iterdir() if p.suffix.lower() in _VIDEO_EXTS}
    pairs: List[Tuple[str, str]] = []
    for stem, vp in sorted(videos.items()):
        lp = ld / f"{stem}.csv"
        if lp.is_file():
            pairs.append((str(vp), str(lp)))
    if not pairs:
        raise RuntimeError("No matched video/label pairs found for tuning.")
    return pairs


def _load_label_segments(label_csv: str) -> List[Tuple[float, float, str]]:
    out: List[Tuple[float, float, str]] = []
    with open(label_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
                (
                    float(row["timestamp_start"]),
                    float(row["timestamp_end"]),
                    row["state"].strip(),
                )
            )
    return out


def _state_at(t: float, segs: List[Tuple[float, float, str]]) -> Optional[str]:
    for s, e, st in segs:
        if s <= t <= e:
            return st
    return None


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _cache_key(
    video_path: str,
    frame_skip: int,
    blur_threshold: float,
    feature_mask: Optional[List[int]] = None,
) -> dict:
    return {
        "mtime": os.path.getmtime(video_path),
        "frame_skip": frame_skip,
        "blur_threshold": blur_threshold,
        "feature_mask": list(feature_mask) if feature_mask is not None else None,
    }


def _load_manifest(manifest_path: str) -> dict:
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r") as f:
            return json.load(f)
    return {}


def _load_cached_features(
    cache_dir: str,
    manifest: dict,
    video_path: str,
    frame_skip: int,
    blur_threshold: float,
    feature_mask: Optional[List[int]],
) -> Optional[np.ndarray]:
    stem = Path(video_path).stem
    if stem not in manifest:
        return None
    if manifest[stem] != _cache_key(video_path, frame_skip, blur_threshold, feature_mask):
        return None
    npy_path = os.path.join(cache_dir, f"{stem}.npy")
    if not os.path.isfile(npy_path):
        return None
    return np.load(npy_path)


def tune(
    video_dir: str,
    label_dir: str,
    model_dir: str,
    aruco_config: Optional[str],
    frame_skip: int,
    blur_threshold: float,
    penalties: List[float],
    scales: List[float],
    strict_cycle: bool,
    feature_mask: Optional[List[int]] = None,
    coarse_feature_mask: Optional[List[int]] = None,
    interact_feature_mask: Optional[List[int]] = None,
    carry_feature_mask: Optional[List[int]] = None,
    pipeline_mode: str = "two-stage",
    verbose: bool = True,
) -> Dict:
    final_dir = os.path.join(model_dir, "models", "hmm_final")
    if not os.path.isdir(final_dir):
        final_dir = model_dir
    if pipeline_mode == "two-stage":
        cfg_path = os.path.join(final_dir, "pipeline_config.json")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
                pipeline_mode = str(cfg.get("pipeline_mode", pipeline_mode))
            except Exception:
                pass

    def _load_mask(path: str) -> Optional[List[int]]:
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return [int(x) for x in data.get("selected_indices", [])]
        except Exception:
            return None

    if feature_mask is None:
        feature_mask = _load_mask(os.path.join(final_dir, "feature_mask.json"))
    if coarse_feature_mask is None:
        coarse_feature_mask = _load_mask(os.path.join(final_dir, "feature_mask_coarse.json"))
    if interact_feature_mask is None:
        interact_feature_mask = _load_mask(os.path.join(final_dir, "feature_mask_interact.json"))
    if carry_feature_mask is None:
        carry_feature_mask = _load_mask(os.path.join(final_dir, "feature_mask_carry.json"))

    pairs = _discover_pairs(video_dir, label_dir)
    if verbose:
        print(f"[hmm_tune] Matched dev pairs: {len(pairs)}")

    aruco_detector = ArucoDetector(
        aruco_dict_type=DEFAULT_HTK_CONFIG.aruco_dict_type,
        distance_decay=DEFAULT_HTK_CONFIG.aruco_distance_decay,
    )
    if aruco_config and os.path.isfile(aruco_config):
        aruco_detector.load_bin_config(aruco_config)

    extractor = FeatureExtractor(aruco_detector=aruco_detector, feature_mask=feature_mask)
    htk = HTKStateDetector(model_dir)
    coarse_htk = HTKStateDetector(os.path.join(model_dir, "coarse_model"), state_labels=COARSE_STATES)
    interact_dir = os.path.join(model_dir, "interact_model")
    carry_dir = os.path.join(model_dir, "carry_model")
    interact_htk = HTKStateDetector(interact_dir, state_labels=INTERACT_STATES) if os.path.isdir(interact_dir) else None
    carry_htk = HTKStateDetector(carry_dir, state_labels=CARRY_STATES) if os.path.isdir(carry_dir) else None
    if verbose and pipeline_mode == "two-stage":
        if interact_htk is None:
            print("[hmm_tune] WARNING: interact_model missing; using fallback during interact runs.")
        if carry_htk is None:
            print("[hmm_tune] WARNING: carry_model missing; using fallback during carry runs.")
    coarse_mask = coarse_feature_mask if coarse_feature_mask is not None else feature_mask
    interact_mask = interact_feature_mask if interact_feature_mask is not None else feature_mask
    carry_mask = carry_feature_mask if carry_feature_mask is not None else feature_mask

    # Reuse training feature cache for speed when available.
    cache_dir = os.path.join(model_dir, "feature_cache")
    manifest = _load_manifest(os.path.join(cache_dir, "manifest.json"))

    # Cache extracted features once so sweep is fast.
    cached: List[Dict] = []
    for video_path, label_csv in pairs:
        feats = _load_cached_features(
            cache_dir=cache_dir,
            manifest=manifest,
            video_path=video_path,
            frame_skip=frame_skip,
            blur_threshold=blur_threshold,
            feature_mask=feature_mask,
        )
        if feats is not None:
            # Frame numbers/fps still needed for timestamp mapping and scoring.
            # These are cheap to compute from video metadata + stride.
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if fps <= 0:
                fps = 30.0
            frame_numbers = [i for i in range(1, total_frames + 1) if i % frame_skip == 0]
            if len(frame_numbers) > feats.shape[0]:
                frame_numbers = frame_numbers[:feats.shape[0]]
            elif len(frame_numbers) < feats.shape[0]:
                if len(frame_numbers) == 0:
                    frame_numbers = list(range(1, feats.shape[0] + 1))
                else:
                    last = frame_numbers[-1]
                    while len(frame_numbers) < feats.shape[0]:
                        last += frame_skip
                        frame_numbers.append(last)
            if verbose:
                print(f"[hmm_tune] cache hit: {Path(video_path).name} ({feats.shape[0]} frames)")
        else:
            feats, frame_numbers, fps = extractor.extract_video_features(
                video_path,
                frame_skip=frame_skip,
                blur_threshold=blur_threshold,
                verbose=verbose,
            )
        if feats.shape[0] == 0:
            continue
        labels = _load_label_segments(label_csv)
        cached.append(
            {
                "video": video_path,
                "label": label_csv,
                "features": feats,
                "frame_numbers": frame_numbers,
                "fps": fps,
                "labels": labels,
            }
        )

    if not cached:
        raise RuntimeError("No usable feature sequences in dev set.")

    rows: List[Dict] = []
    best = {"macro_f1": -1.0}
    for p in penalties:
        for s in scales:
            y_true: List[str] = []
            y_pred: List[str] = []
            y_true_coarse: List[str] = []
            y_pred_coarse: List[str] = []
            y_true_interact: List[str] = []
            y_pred_interact: List[str] = []
            y_true_carry: List[str] = []
            y_pred_carry: List[str] = []
            for item in cached:
                if pipeline_mode == "legacy":
                    pred_df: pd.DataFrame = htk.decode(
                        item["features"],
                        item["fps"],
                        frame_numbers=item["frame_numbers"],
                        verbose=False,
                        word_penalty=p,
                        grammar_scale=s,
                        strict_cycle=strict_cycle,
                    )
                else:
                    pred_coarse_df: pd.DataFrame = coarse_htk.decode(
                        apply_feature_mask(item["features"], coarse_mask),
                        item["fps"],
                        frame_numbers=item["frame_numbers"],
                        verbose=False,
                        word_penalty=p,
                        grammar_scale=s,
                        strict_cycle=strict_cycle,
                    )
                    pred_coarse_frame = segments_to_frame_labels(
                        pred_coarse_df, item["frame_numbers"], item["fps"]
                    )
                    pred_fine_frame = decode_subtype_with_runs(
                        pred_coarse_frame,
                        item["features"],
                        item["fps"],
                        item["frame_numbers"],
                        interact_detector=interact_htk,
                        carry_detector=carry_htk,
                        strict_cycle=False,
                        word_penalty=p,
                        grammar_scale=s,
                        interact_mask=interact_mask,
                        carry_mask=carry_mask,
                    )
                    pred_df = frame_labels_to_segments(pred_fine_frame, item["frame_numbers"], item["fps"])
                pred_segs = [
                    (float(r.timestamp_start), float(r.timestamp_end), str(r.state))
                    for r in pred_df.itertuples()
                ]
                pred_coarse_segs = [(a, b, map_fine_to_coarse(c)) for (a, b, c) in pred_segs]
                for fn in item["frame_numbers"]:
                    t = fn / item["fps"] if item["fps"] > 0 else 0.0
                    gt = _state_at(t, item["labels"])
                    if gt is None:
                        continue
                    gt_coarse = map_fine_to_coarse(gt)
                    pr = _state_at(t, pred_segs)
                    if pr is None:
                        continue
                    pr_coarse = _state_at(t, pred_coarse_segs)
                    y_true.append(gt)
                    y_pred.append(pr)
                    if pr_coarse is not None:
                        y_true_coarse.append(gt_coarse)
                        y_pred_coarse.append(pr_coarse)
                    if gt in ("PICK", "PLACE") and pr in ("PICK", "PLACE"):
                        y_true_interact.append(gt)
                        y_pred_interact.append(pr)
                    if gt in ("CARRY_WITH", "CARRY_EMPTY") and pr in ("CARRY_WITH", "CARRY_EMPTY"):
                        y_true_carry.append(gt)
                        y_pred_carry.append(pr)

            labels = ["PICK", "CARRY_WITH", "PLACE", "CARRY_EMPTY"]
            if y_true:
                macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
                recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
                per_class_recall = {label: float(recalls[i]) for i, label in enumerate(labels)}
            else:
                macro_f1 = 0.0
                per_class_recall = {label: 0.0 for label in labels}
            coarse_macro_f1 = (
                float(f1_score(y_true_coarse, y_pred_coarse, labels=COARSE_STATES, average="macro", zero_division=0))
                if y_true_coarse
                else 0.0
            )
            interact_macro_f1 = (
                float(f1_score(y_true_interact, y_pred_interact, labels=["PICK", "PLACE"], average="macro", zero_division=0))
                if y_true_interact
                else 0.0
            )
            carry_macro_f1 = (
                float(
                    f1_score(
                        y_true_carry,
                        y_pred_carry,
                        labels=["CARRY_WITH", "CARRY_EMPTY"],
                        average="macro",
                        zero_division=0,
                    )
                )
                if y_true_carry
                else 0.0
            )

            row = {
                "word_penalty": p,
                "grammar_scale": s,
                "macro_f1": macro_f1,
                "samples": len(y_true),
                "coarse_macro_f1": coarse_macro_f1,
                "interact_macro_f1": interact_macro_f1,
                "carry_macro_f1": carry_macro_f1,
                "recall_pick": per_class_recall["PICK"],
                "recall_carry_with": per_class_recall["CARRY_WITH"],
                "recall_place": per_class_recall["PLACE"],
                "recall_carry_empty": per_class_recall["CARRY_EMPTY"],
            }
            rows.append(row)
            if verbose:
                print(
                    f"[hmm_tune] p={p:>5.2f} s={s:>5.2f} "
                    f"macro_f1={macro_f1:.4f} n={len(y_true)} "
                    f"coarse/interact/carry="
                    f"{coarse_macro_f1:.3f}/{interact_macro_f1:.3f}/{carry_macro_f1:.3f} "
                    f"R(P/CW/PL/CE)="
                    f"{row['recall_pick']:.3f}/{row['recall_carry_with']:.3f}/"
                    f"{row['recall_place']:.3f}/{row['recall_carry_empty']:.3f}"
                )
            tiebreak = coarse_macro_f1 + interact_macro_f1 + carry_macro_f1
            best_tiebreak = (
                best.get("coarse_macro_f1", 0.0)
                + best.get("interact_macro_f1", 0.0)
                + best.get("carry_macro_f1", 0.0)
            )
            if (macro_f1 > best["macro_f1"]) or (abs(macro_f1 - best["macro_f1"]) < 1e-12 and tiebreak > best_tiebreak):
                best = {
                    "macro_f1": macro_f1,
                    "word_penalty": p,
                    "grammar_scale": s,
                    "samples": len(y_true),
                    "coarse_macro_f1": coarse_macro_f1,
                    "interact_macro_f1": interact_macro_f1,
                    "carry_macro_f1": carry_macro_f1,
                    "per_class_recall": per_class_recall,
                }

    rows_sorted = sorted(rows, key=lambda r: r["macro_f1"], reverse=True)
    result = {
        "strict_cycle": strict_cycle,
        "pipeline_mode": pipeline_mode,
        "best": best,
        "grid": rows_sorted,
    }

    out_json = os.path.join(final_dir, "infer_params.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "word_penalty": best["word_penalty"],
                "grammar_scale": best["grammar_scale"],
                "strict_cycle": strict_cycle,
                "pipeline_mode": pipeline_mode,
                "dev_macro_f1": best["macro_f1"],
                "dev_per_class_recall": best["per_class_recall"],
                "dev_coarse_macro_f1": best["coarse_macro_f1"],
                "dev_interact_macro_f1": best["interact_macro_f1"],
                "dev_carry_macro_f1": best["carry_macro_f1"],
                "dev_samples": best["samples"],
            },
            f,
            indent=2,
        )
    out_csv = os.path.join(final_dir, "tuning_grid.csv")
    pd.DataFrame(rows_sorted).to_csv(out_csv, index=False)

    if verbose:
        print(
            f"\n[hmm_tune] Best: p={best['word_penalty']}, "
            f"s={best['grammar_scale']}, macro_f1={best['macro_f1']:.4f}"
        )
        print(f"[hmm_tune] Wrote params: {out_json}")
        print(f"[hmm_tune] Wrote grid  : {out_csv}")

    return result


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m symbiote.hmm_tune",
        description="Tune HVite decode params (word penalty / grammar scale) on a labeled dev set.",
    )
    p.add_argument(
        "--video-dir",
        default=str(_ROOT / "hmm-testing" / "picklist_videos"),
        help="Directory containing dev videos",
    )
    p.add_argument(
        "--label-dir",
        default=str(_ROOT / "hmm-testing" / "picklist_labels"),
        help="Directory containing dev CSV labels",
    )
    p.add_argument(
        "--model-dir",
        default=str(_ROOT / "models" / "htk"),
        help="Trained HTK model directory",
    )
    p.add_argument(
        "--aruco-config",
        default=str(_ROOT / "config" / "aruco_bins.json"),
        help="Path to aruco_bins.json",
    )
    p.add_argument("--frame-skip", type=int, default=4, help="Feature extraction frame stride")
    p.add_argument("--threshold", type=float, default=100.0, help="Blur threshold")
    p.add_argument(
        "--penalties",
        default="-3,-1,0,1,2,3",
        help="Comma-separated list for HVite -p sweep",
    )
    p.add_argument(
        "--scales",
        default="0.5,1,2,5,8",
        help="Comma-separated list for HVite -s sweep",
    )
    p.add_argument(
        "--free-order",
        action="store_true",
        help="Use unconstrained order instead of strict cycle grammar.",
    )
    p.add_argument(
        "--feature-mask",
        default=None,
        help="Comma-separated feature indices to keep active (others zeroed).",
    )
    p.add_argument("--coarse-feature-mask", default=None, help="Comma-separated mask for coarse stage.")
    p.add_argument("--interact-feature-mask", default=None, help="Comma-separated mask for interact stage.")
    p.add_argument("--carry-feature-mask", default=None, help="Comma-separated mask for carry stage.")
    p.add_argument(
        "--pipeline",
        choices=["two-stage", "legacy"],
        default="two-stage",
        help="Tuning/eval pipeline mode (default: two-stage).",
    )
    p.add_argument(
        "--legacy",
        action="store_true",
        help="Shorthand for --pipeline legacy.",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return p


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    aruco_config = args.aruco_config if os.path.isfile(args.aruco_config) else None
    pipeline_mode = "legacy" if args.legacy else args.pipeline
    tune(
        video_dir=args.video_dir,
        label_dir=args.label_dir,
        model_dir=args.model_dir,
        aruco_config=aruco_config,
        frame_skip=args.frame_skip,
        blur_threshold=args.threshold,
        penalties=_parse_float_list(args.penalties),
        scales=_parse_float_list(args.scales),
        strict_cycle=not args.free_order,
        feature_mask=None if not args.feature_mask else [int(x.strip()) for x in args.feature_mask.split(",") if x.strip()],
        coarse_feature_mask=None if not args.coarse_feature_mask else [int(x.strip()) for x in args.coarse_feature_mask.split(",") if x.strip()],
        interact_feature_mask=None if not args.interact_feature_mask else [int(x.strip()) for x in args.interact_feature_mask.split(",") if x.strip()],
        carry_feature_mask=None if not args.carry_feature_mask else [int(x.strip()) for x in args.carry_feature_mask.split(",") if x.strip()],
        pipeline_mode=pipeline_mode,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
