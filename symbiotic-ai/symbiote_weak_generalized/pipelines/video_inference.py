"""Video inference pipeline for frame-by-frame object classification with CSV output.

This pipeline processes videos and outputs inference results without adding data to training cache.
"""

import json
import os
import csv
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import mediapipe as mp
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..lib.hand_detection import segment_hand

from ..preprocessing.blur_detection import is_blurry
from ..inference.recognizer import ObjectRecognizer
from ..models.classifier import CentroidModel
from ..state_detection.compact_timeline import carry_with_pipeline_frame_intervals_1based
from ..training.cluster_voting import _cosine_distance_matrix
from ..training.weak_supervision import spherical_mean


def _load_picklist_rounds_ordered(picklist_json_path: str) -> List[List[str]]:
    """Load ``picklists`` as ordered rounds (each inner list is one picklist round)."""
    with open(picklist_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rounds: List[List[str]] = []
    for block in data.get("picklists", []):
        if isinstance(block, list):
            rounds.append([str(x) for x in block])
    return rounds


def _union_labels_from_rounds(rounds: List[List[str]]) -> Set[str]:
    s: Set[str] = set()
    for r in rounds:
        s.update(r)
    return s


def _interval_index_for_frame(frame_count: int, intervals: List[Tuple[int, int]]) -> int:
    """Inclusive 1-based frame index -> interval index, or -1 if none."""
    for i, (lo, hi) in enumerate(intervals):
        if lo <= frame_count <= hi:
            return i
    return -1


def _l2_normalize_row(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n


def _top_labels_by_cosine(
    vec: np.ndarray,
    labels: List[str],
    centroids: Dict[str, np.ndarray],
    k: int = 3,
) -> Tuple[float, List[Tuple[str, float]]]:
    """Return (best_cosine_sim, top-k list of (label, cosine_sim)) over unique labels."""
    uniq: List[str] = []
    for lab in labels:
        if lab not in uniq:
            uniq.append(lab)
    vn = _l2_normalize_row(vec)
    scored: List[Tuple[str, float]] = []
    for lab in uniq:
        c = centroids.get(lab)
        if c is None:
            continue
        sim = float(np.dot(vn, _l2_normalize_row(np.asarray(c, dtype=np.float64))))
        scored.append((lab, sim))
    scored.sort(key=lambda x: -x[1])
    top = scored[:k]
    best_conf = top[0][1] if top else 0.0
    return best_conf, top


def _hungarian_interval_labels(
    interval_means: Dict[int, np.ndarray],
    group_interval_indices: List[int],
    multiset: List[str],
    centroids: Dict[str, np.ndarray],
) -> Dict[int, str]:
    """Assign one label per interval in the group via Hungarian (min cosine cost)."""
    m = len(group_interval_indices)
    if m != len(multiset):
        raise ValueError("Hungarian group size mismatch")
    Xs = np.stack([interval_means[i] for i in group_interval_indices], axis=0)
    R = np.stack([np.asarray(centroids[lab], dtype=np.float64) for lab in multiset], axis=0)
    for i in range(R.shape[0]):
        ni = float(np.linalg.norm(R[i]))
        if ni < 1e-12:
            v = np.zeros(R.shape[1], dtype=np.float64)
            v[0] = 1.0
            R[i] = v
        else:
            R[i] = R[i] / ni
    cost = _cosine_distance_matrix(Xs, R)
    row_norms = np.linalg.norm(Xs, axis=1)
    cost[row_norms < 1e-12, :] = 1.0
    ri, ci = linear_sum_assignment(cost)
    out: Dict[int, str] = {}
    for r, c in zip(ri, ci):
        out[group_interval_indices[int(r)]] = multiset[int(c)]
    return out


def _constrained_top_k(
    model: CentroidModel,
    embedding: np.ndarray,
    valid_labels: Optional[Set[str]],
    k: int = 3,
    verbose: bool = False,
    frame_number: int = 0,
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Top-k by softmax probability over centroids; if ``valid_labels`` is set,
    restrict to those labels (re-normalize over the subset).
    """
    probs = model.predict_proba(embedding)
    if not valid_labels:
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top = sorted_probs[:k]
        return top[0][0], top[0][1], top

    constrained = {lab: p for lab, p in probs.items() if lab in valid_labels}
    if constrained:
        total = sum(constrained.values())
        if total > 0:
            constrained = {lab: p / total for lab, p in constrained.items()}
        sorted_probs = sorted(constrained.items(), key=lambda x: x[1], reverse=True)
        top = sorted_probs[:k]
        return top[0][0], top[0][1], top

    if verbose:
        print(
            f"  Frame {frame_number}: WARNING picklist has no overlap with model labels; "
            "using unconstrained prediction"
        )
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top = sorted_probs[:k]
    return top[0][0], top[0][1], top


def _frame_in_carry_with_intervals(
    frame_count_1based: int,
    intervals: List[Tuple[int, int]],
) -> bool:
    """Same convention as ``video_processor``: inclusive 1-based ``[lo, hi]`` ranges."""
    return any(lo <= frame_count_1based <= hi for (lo, hi) in intervals)


def run_video_inference(
    video_path: str,
    model_dir: str,
    output_csv: str,
    threshold: float = 50.0,
    frame_skip: int = 4,
    verbose: bool = True,
    picklist_json: Optional[str] = None,
    manual_labels_dir: Optional[str] = None,
    compact_frame_indexing: str = "opencv0",
    apply_iterated_postprocess: bool = False,
) -> str:
    """
    Run inference on video frames and output results to CSV.

    Frame handling matches ``process_video_frames`` (training): RGB, downscale above
    1080p to fit in 1920x1080 before MediaPipe, same Hands confidence settings, then
    blur on the segmented hand patch.

    With ``manual_labels_dir`` and CARRY_WITH intervals:
    - Aggregates normalized frame embeddings per interval (spherical mean).
    - If ``picklist_json`` is also set: strict per-round Hungarian over each round's
      multiset (same geometry as training); CSV rows inherit the interval label.
    - If only manual labels: per-interval cosine top-1 over all centroids.

    Without manual labels: per-frame inference (optionally constrained to the union
    of labels in ``picklist_json`` if provided).

    IMPORTANT: Does NOT add data to training cache/dataset.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("VIDEO INFERENCE PIPELINE")
        print("=" * 60)
        print(f"Video: {video_path}")
        print(f"Model: {model_dir}")
        print(f"Output: {output_csv}")
        print(f"Blur threshold: {threshold}")
        print(f"Frame skip: {frame_skip}")
        print(f"Apply iterated postprocess: {bool(apply_iterated_postprocess)}")
        if picklist_json:
            print(f"Picklist JSON: {picklist_json}")
        if manual_labels_dir:
            print(f"Manual labels dir (CARRY_WITH windows): {manual_labels_dir}")
            print(f"  compact_frame_indexing: {compact_frame_indexing}")

    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    annotation_path: Optional[str] = None
    if manual_labels_dir is not None:
        if not os.path.isdir(manual_labels_dir):
            raise FileNotFoundError(
                f"manual_labels_dir is not a directory: {manual_labels_dir}"
            )
        annotation_path = os.path.join(manual_labels_dir, f"{video_stem}.csv")
        if not os.path.isfile(annotation_path):
            raise FileNotFoundError(
                f"No manual label CSV for video stem {video_stem!r}: expected {annotation_path} "
                f"(same naming as train --manual-labels-dir)"
            )

    carry_intervals: Optional[List[Tuple[int, int]]] = None

    rounds_ordered: Optional[List[List[str]]] = None
    valid_labels: Optional[Set[str]] = None
    if picklist_json:
        if not os.path.isfile(picklist_json):
            raise FileNotFoundError(f"Picklist JSON not found: {picklist_json}")
        rounds_ordered = _load_picklist_rounds_ordered(picklist_json)
        valid_labels = _union_labels_from_rounds(rounds_ordered)
        if verbose:
            print(f"  Picklist union ({len(valid_labels)}): {sorted(valid_labels)}")

    if verbose:
        print("\nLoading model...")
    recognizer = ObjectRecognizer(model_dir)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if annotation_path is not None:
        carry_intervals = carry_with_pipeline_frame_intervals_1based(
            annotation_path,
            total_frames,
            frame_indexing=compact_frame_indexing,
        )
        if verbose:
            print(f"  Using manual labels: {annotation_path}")
            print(f"  CARRY_WITH intervals (1-based, inclusive): {len(carry_intervals)} span(s)")
            for i, (lo, hi) in enumerate(carry_intervals):
                print(f"    [{i}] frames {lo}–{hi}")

    if verbose:
        print(f"\nVideo info:")
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Processing every {frame_skip} frames...")
        if carry_intervals is not None:
            print(f"  Restricted to CARRY_WITH: yes ({len(carry_intervals)} interval(s))")
        print("\nProcessing frames...")

    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.3,
        max_num_hands=2,
    )

    frame_buffer: List[Dict[str, Any]] = []
    frame_count = 0
    processed_count = 0
    inference_count = 0

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % frame_skip != 0:
                continue

            processed_count += 1

            if carry_intervals is not None:
                if not _frame_in_carry_with_intervals(frame_count, carry_intervals):
                    continue

            timestamp = frame_count / fps if fps > 0 else 0.0

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h_orig, w_orig = image_rgb.shape[:2]
            if w_orig > 1920 or h_orig > 1080:
                scale = min(1920 / w_orig, 1080 / h_orig)
                image_rgb = cv2.resize(
                    image_rgb,
                    (int(w_orig * scale), int(h_orig * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            segmented = segment_hand(image_rgb, hands_detector)

            if segmented is None:
                if verbose:
                    print(f"  Frame {frame_count} (t={timestamp:.2f}s): NO HAND (skipped)")
                continue

            if segmented.size == 0:
                if verbose:
                    print(f"  Frame {frame_count} (t={timestamp:.2f}s): EMPTY (skipped)")
                continue

            if is_blurry(segmented, threshold):
                if verbose:
                    print(f"  Frame {frame_count} (t={timestamp:.2f}s): BLURRY (skipped)")
                continue

            try:
                import torch

                inputs = recognizer.processor(
                    images=[segmented], return_tensors="pt"
                ).to(recognizer.clip_model.device)
                with torch.no_grad():
                    embedding = recognizer.clip_model.get_image_features(**inputs)
                embedding_np = embedding.cpu().numpy()[0].astype(np.float64)
                if apply_iterated_postprocess:
                    embedding_np = recognizer._postprocess_embedding(embedding_np)

                int_idx = -1
                if carry_intervals is not None:
                    int_idx = _interval_index_for_frame(frame_count, carry_intervals)
                    if int_idx < 0:
                        continue

                frame_buffer.append(
                    {
                        "frame_number": frame_count,
                        "timestamp": timestamp,
                        "embedding": embedding_np,
                        "interval_idx": int_idx,
                    }
                )
                inference_count += 1

                if verbose:
                    print(f"  Frame {frame_count} (t={timestamp:.2f}s): embedded (buffered)")

            except Exception as e:
                if verbose:
                    print(f"  Frame {frame_count} (t={timestamp:.2f}s): ERROR ({e})")
    finally:
        hands_detector.close()
        capture.release()

    if verbose:
        print(f"\n" + "=" * 60)
        print("EMBEDDING PASS COMPLETE")
        print("=" * 60)
        print(f"Total frames: {frame_count}")
        print(f"Frames checked: {processed_count}")
        print(f"Embeddings buffered: {inference_count}")

    results: List[Dict[str, Any]] = []
    centroids = recognizer.model.centroids

    if not frame_buffer:
        if verbose:
            print("\nWarning: No frames could be processed. Creating empty CSV.")
    elif carry_intervals is not None:
        n_iv = len(carry_intervals)
        by_iv: Dict[int, List[np.ndarray]] = defaultdict(list)
        for fb in frame_buffer:
            by_iv[int(fb["interval_idx"])].append(fb["embedding"])

        emb_dim = int(frame_buffer[0]["embedding"].shape[0])
        interval_mean: Dict[int, np.ndarray] = {}
        for i in range(n_iv):
            vecs = by_iv.get(i, [])
            if vecs:
                interval_mean[i] = spherical_mean(np.stack(vecs, axis=0))
            else:
                interval_mean[i] = np.zeros(emb_dim, dtype=np.float64)

        interval_label: Dict[int, str] = {}
        interval_best_sim: Dict[int, float] = {}
        interval_top3: Dict[int, List[Tuple[str, float]]] = {}

        model_labs = list(centroids.keys())

        use_hungarian = bool(picklist_json and rounds_ordered)
        if use_hungarian and rounds_ordered is not None:
            need = sum(len(r) for r in rounds_ordered)
            if need != n_iv:
                if verbose:
                    print(
                        f"\nWARNING: picklist interval count ({need}) != CARRY_WITH intervals ({n_iv}); "
                        "falling back to per-interval top-1 over picklist union."
                    )
                use_hungarian = False

        if use_hungarian and rounds_ordered is not None:
            pos = 0
            for rlabs in rounds_ordered:
                m = len(rlabs)
                idxs = list(range(pos, pos + m))
                pos += m
                sub = _hungarian_interval_labels(interval_mean, idxs, rlabs, centroids)
                interval_label.update(sub)
                for ii in idxs:
                    lab = interval_label[ii]
                    c = np.asarray(centroids[lab], dtype=np.float64)
                    sim = float(
                        np.dot(_l2_normalize_row(interval_mean[ii]), _l2_normalize_row(c))
                    )
                    interval_best_sim[ii] = sim
                    _, top = _top_labels_by_cosine(interval_mean[ii], rlabs, centroids, k=3)
                    interval_top3[ii] = top
        else:
            cand_union = sorted(valid_labels & set(centroids.keys())) if valid_labels else model_labs
            if not cand_union:
                cand_union = model_labs
            for i in range(n_iv):
                sim, top = _top_labels_by_cosine(interval_mean[i], cand_union, centroids, k=3)
                interval_label[i] = top[0][0] if top else (cand_union[0] if cand_union else "unknown")
                interval_best_sim[i] = sim
                interval_top3[i] = top

        for fb in frame_buffer:
            iid = int(fb["interval_idx"])
            pred = interval_label.get(iid, "unknown")
            conf = float(interval_best_sim.get(iid, 0.0))
            pred_cos_sim = conf
            top = interval_top3.get(iid, [])
            top_3_labels = ";".join([lab for lab, _ in top])
            top_3_confidences = ";".join([f"{s:.4f}" for _, s in top])
            results.append(
                {
                    "frame_number": fb["frame_number"],
                    "timestamp": fb["timestamp"],
                    "predicted_label": pred,
                    "confidence": conf,
                    "predicted_label_cosine_similarity": pred_cos_sim,
                    "top_3_labels": top_3_labels,
                    "top_3_confidences": top_3_confidences,
                }
            )
            if verbose:
                print(
                    f"  Frame {fb['frame_number']}: {pred} (interval {iid}, cos={conf:.4f})"
                )
    else:
        for fb in frame_buffer:
            predicted_label, confidence, top_k_results = _constrained_top_k(
                recognizer.model,
                fb["embedding"],
                valid_labels,
                k=3,
                verbose=verbose,
                frame_number=int(fb["frame_number"]),
            )
            if not top_k_results:
                continue
            centroid_vec = centroids.get(predicted_label)
            pred_cos_sim = 0.0
            if centroid_vec is not None:
                pred_cos_sim = float(
                    np.dot(
                        _l2_normalize_row(np.asarray(fb["embedding"], dtype=np.float64)),
                        _l2_normalize_row(np.asarray(centroid_vec, dtype=np.float64)),
                    )
                )
            top_3_labels = ";".join([label for label, _ in top_k_results])
            top_3_confidences = ";".join([f"{conf:.4f}" for _, conf in top_k_results])
            results.append(
                {
                    "frame_number": fb["frame_number"],
                    "timestamp": fb["timestamp"],
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "predicted_label_cosine_similarity": pred_cos_sim,
                    "top_3_labels": top_3_labels,
                    "top_3_confidences": top_3_confidences,
                }
            )
            if verbose:
                print(
                    f"  Frame {fb['frame_number']} (t={fb['timestamp']:.2f}s): "
                    f"{predicted_label} ({confidence:.2f})"
                )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "frame_number",
            "timestamp",
            "predicted_label",
            "confidence",
            "predicted_label_cosine_similarity",
            "top_3_labels",
            "top_3_confidences",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    if verbose:
        print(f"\nResults saved to: {output_csv}")
        print(f"Total CSV rows: {len(results)}")

    return output_csv


__all__ = ["run_video_inference"]
