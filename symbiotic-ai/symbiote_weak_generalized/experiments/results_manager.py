"""Aggregate sweep results, export CSV, and write a short markdown summary."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from .sweep_config import SWEEP_FIXED_THRESHOLD, SWEEP_PARAM_KEYS


def _flatten_run_row(
    run: Dict[str, Any],
    eval_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = run.get("config") or {}
    row: Dict[str, Any] = {
        "run_index": run.get("run_index"),
        "run_dir": run.get("run_dir"),
        "success": bool(run.get("success")),
        "error": run.get("error"),
    }
    for k in SWEEP_PARAM_KEYS:
        row[f"param_{k}"] = cfg.get(k)

    m = (eval_metrics or {}).get("metrics") if eval_metrics else None
    if m:
        row["assignment_hit_rate"] = m.get("assignment_hit_rate")
        row["assignment_hits"] = m.get("assignment_hits")
        row["assignment_compared"] = m.get("assignment_compared")
        row["segment_top1_accuracy"] = m.get("segment_top1_accuracy")
        row["segment_top3_hit_rate"] = m.get("segment_top3_hit_rate")
        row["carry_segments_used"] = m.get("carry_segments_used")
        row["segments_with_predictions"] = m.get("segments_with_predictions")
        row["mean_segment_top1_confidence"] = m.get("mean_segment_top1_confidence")
    else:
        row["assignment_hit_rate"] = None
        row["assignment_hits"] = None
        row["assignment_compared"] = None
        row["segment_top1_accuracy"] = None
        row["segment_top3_hit_rate"] = None
        row["carry_segments_used"] = None
        row["segments_with_predictions"] = None
        row["mean_segment_top1_confidence"] = None

    return row


def aggregate_results(
    runs: Sequence[Dict[str, Any]],
    eval_by_run_dir: Optional[Dict[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Build a DataFrame of swept params + success + optional eval metrics.

    *eval_by_run_dir* maps ``run_dir`` -> scoring dict (e.g. from
    :func:`score_final_assignments_hit_rate`).
    """
    rows: List[Dict[str, Any]] = []
    eval_by_run_dir = eval_by_run_dir or {}
    for run in runs:
        rd = str(run.get("run_dir") or "")
        ev = eval_by_run_dir.get(rd)
        rows.append(_flatten_run_row(run, ev))
    return pd.DataFrame(rows)


def find_best_config(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Row with highest ``assignment_hit_rate`` (or legacy ``segment_top1_accuracy``)."""
    if df.empty:
        return None
    ok = df[df["success"] == True].copy()  # noqa: E712
    metric_col = "assignment_hit_rate"
    if metric_col not in ok.columns or ok[metric_col].isna().all():
        metric_col = "segment_top1_accuracy"
    if metric_col not in ok.columns:
        return None
    ok = ok[ok[metric_col].notna()]
    if ok.empty:
        return None
    idx = ok[metric_col].astype(float).idxmax()
    return ok.loc[idx].to_dict()


def save_results_csv(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_summary_report(
    df: pd.DataFrame,
    path: str,
    *,
    best: Optional[Dict[str, Any]] = None,
) -> None:
    """Write markdown: best run, top 5 by assignment hit rate, numeric correlations."""
    lines: List[str] = []
    lines.append("# Hyperparameter sweep summary\n")
    lines.append(f"Fixed blur threshold (not swept): **{SWEEP_FIXED_THRESHOLD}**\n")
    n = len(df)
    n_ok = int((df["success"] == True).sum()) if "success" in df.columns else 0  # noqa: E712
    lines.append(f"Total runs: {n}, successful: {n_ok}\n")

    metric_col = "assignment_hit_rate"
    if metric_col not in df.columns or df[metric_col].isna().all():
        metric_col = "segment_top1_accuracy"

    if best:
        lines.append(f"## Best configuration (by {metric_col})\n")
        for k, v in sorted(best.items()):
            if str(k).startswith("param_") or k in (
                metric_col,
                "assignment_hits",
                "assignment_compared",
                "segment_top1_accuracy",
                "segment_top3_hit_rate",
                "run_dir",
                "run_index",
            ):
                lines.append(f"- **{k}**: `{v}`")
        lines.append("")

    if metric_col in df.columns and not df[metric_col].isna().all():
        top = df[df[metric_col].notna()].sort_values(metric_col, ascending=False).head(5)
        lines.append(f"## Top 5 runs ({metric_col})\n")
        lines.append("```")
        lines.append(top.to_string(index=False))
        lines.append("```\n")

    param_cols = [c for c in df.columns if c.startswith("param_")]
    numeric_params: List[str] = []
    for c in param_cols:
        ser = pd.to_numeric(df[c], errors="coerce")
        valid = ser.dropna()
        if len(valid) >= 2 and valid.std(ddof=0) > 0:
            numeric_params.append(c)

    if metric_col in df.columns and numeric_params:
        sub = df[df["success"] == True].copy()  # noqa: E712
        sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
        for c in numeric_params:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        sub = sub.dropna(subset=[metric_col])
        if len(sub) >= 2:
            try:
                corr = (
                    sub[numeric_params + [metric_col]]
                    .corr(numeric_only=True)[metric_col]
                    .drop(metric_col, errors="ignore")
                )
                if not corr.empty:
                    lines.append(f"## Pearson correlation with {metric_col}\n")
                    lines.append("```")
                    lines.append(corr.sort_values(ascending=False).to_string())
                    lines.append("```\n")
            except Exception:
                pass

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
