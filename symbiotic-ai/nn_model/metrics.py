"""Evaluation metrics for state prediction."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import parse_boundary_labels
from .states import IDX_TO_STATE, STATE_TO_IDX, STATES


def frame_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true >= 0
    if not mask.any():
        return 0.0
    return float((y_true[mask] == y_pred[mask]).mean())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def per_class_prf(cm: np.ndarray) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for i, state in enumerate(STATES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[state] = {"precision": prec, "recall": rec, "f1": f1}
    return results


def macro_f1(cm: np.ndarray) -> float:
    prf = per_class_prf(cm)
    return float(np.mean([v["f1"] for v in prf.values()]))


def carry_count_accuracy(
    pred_paths: Sequence[np.ndarray],
    expected_counts: Sequence[Optional[int]],
) -> Tuple[int, int]:
    correct = 0
    total = 0
    from .decode import count_carry_segments

    for path, expected in zip(pred_paths, expected_counts):
        if expected is None:
            continue
        total += 1
        if count_carry_segments(path) == expected:
            correct += 1
    return correct, total


def labels_to_segment_starts(labels: np.ndarray, fps: float) -> List[Tuple[float, str]]:
    """Convert frame labels to (start_time, state_name) segment boundaries."""
    if len(labels) == 0:
        return []
    segments: List[Tuple[float, str]] = []
    prev = int(labels[0])
    start = 0
    for t in range(1, len(labels)):
        cur = int(labels[t])
        if cur != prev:
            segments.append((start / fps, IDX_TO_STATE[prev]))
            start = t
            prev = cur
    segments.append((start / fps, IDX_TO_STATE[prev]))
    return segments


def boundary_starts_from_labels_file(label_path, fps: float) -> List[Tuple[float, str]]:
    boundaries, _ = parse_boundary_labels(label_path)
    return [(frame / fps, _token_to_name(tok)) for frame, tok in boundaries if _token_to_name(tok)]


def _token_to_name(token: str) -> Optional[str]:
    from .states import TOKEN_TO_STATE

    if token in TOKEN_TO_STATE and token != "sil":
        return TOKEN_TO_STATE[token]
    return None


def boundary_rmse(
    gt_starts: List[Tuple[float, str]],
    pred_starts: List[Tuple[float, str]],
) -> Tuple[float, int]:
    """RMSE of segment start times aligned by index."""
    n = min(len(gt_starts), len(pred_starts))
    if n == 0:
        return 0.0, 0
    err = sum((gt_starts[i][0] - pred_starts[i][0]) ** 2 for i in range(n))
    return float(np.sqrt(err / n)), n


def evaluate_predictions(
    y_true_list: Sequence[np.ndarray],
    y_pred_list: Sequence[np.ndarray],
    expected_carry_counts: Optional[Sequence[Optional[int]]] = None,
) -> Dict[str, object]:
    all_true = np.concatenate(y_true_list)
    all_pred = np.concatenate(y_pred_list)
    cm = confusion_matrix(all_true, all_pred)
    prf = per_class_prf(cm)
    result: Dict[str, object] = {
        "frame_accuracy": frame_accuracy(all_true, all_pred),
        "macro_f1": macro_f1(cm),
        "per_class": prf,
        "confusion_matrix": cm.tolist(),
    }
    if expected_carry_counts is not None:
        correct, total = carry_count_accuracy(y_pred_list, expected_carry_counts)
        result["carry_count_accuracy"] = correct / total if total > 0 else 0.0
        result["carry_count_correct"] = correct
        result["carry_count_total"] = total
    return result


def format_report(metrics: Dict[str, object]) -> str:
    lines = [
        f"Frame accuracy: {metrics['frame_accuracy']:.4f}",
        f"Macro F1:       {metrics['macro_f1']:.4f}",
    ]
    if "carry_count_accuracy" in metrics:
        lines.append(
            f"Carry count accuracy: {metrics['carry_count_accuracy']:.4f} "
            f"({metrics['carry_count_correct']}/{metrics['carry_count_total']})"
        )
    lines.append("")
    lines.append("Per-class metrics:")
    per_class = metrics["per_class"]
    for state in STATES:
        m = per_class[state]
        lines.append(
            f"  {state:12s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}"
        )
    lines.append("")
    lines.append("Confusion matrix (rows=GT, cols=PRED):")
    cm = np.array(metrics["confusion_matrix"])
    header = " " * 14 + "".join(f"{s:14s}" for s in STATES)
    lines.append(header)
    for i, state in enumerate(STATES):
        row = f"{state:12s}  " + "".join(f"{cm[i, j]:14d}" for j in range(len(STATES)))
        lines.append(row)
    return "\n".join(lines)
