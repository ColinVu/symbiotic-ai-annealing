"""Build JSON + Markdown reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import run_full_report
from .segments import SegmentEmbeddings


def build_markdown(
    video_stem: str,
    report: Dict[str, Any],
    coverage: Dict[str, Any],
    notes: List[str],
) -> str:
    lines: List[str] = [
        f"# Cached embedding analysis: `{video_stem}`",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Coverage",
        f"- {coverage.get('summary', '')}",
        "",
        "## Notes",
    ]
    for n in notes:
        lines.append(f"- {n}")
    lines.append("")
    lines.append("## Within-segment (cosine)")
    lines.append(
        "For each CARRY segment: **mean_pairwise_cos** = mean cos sim between all frame pairs; "
        "**cos_middle_to_centroid** = cos sim between the middle frame (by time) and the segment spherical mean; "
        "**mean_cos_*_third** = mean cos to centroid for frames in first/middle/last third of the segment."
    )
    lines.append("")
    w = report.get("within_segments", [])
    for row in w:
        lines.append(
            f"- seg {row['segment_idx']} **{row['true_label']}**  "
            f"n={row['num_frames']}  "
            f"pairwise_mean={_fmt(row.get('mean_pairwise_cos'))}  "
            f"mid↔centroid={_fmt(row.get('cos_middle_to_centroid'))}  "
            f"begin/mid/end→centroid={_fmt(row.get('mean_cos_begin_third'))}/"
            f"{_fmt(row.get('mean_cos_mid_third'))}/"
            f"{_fmt(row.get('mean_cos_end_third'))}"
        )
    lines.append("")
    lines.append(
        "## Global segment–segment (same item, all videos in run)\n\n"
        "For each item with **≥2** segments in the full run, cosine sim between **spherical means** of "
        "each segment (cached frames) is reported in the **summary JSON** and written as heatmaps: "
        "`matrix_seg_global_<item>.png` and `.html` in the same output directory — not in this per-video file."
    )
    lines.append("")

    lines.append("## Cross-segment, same ground-truth item (middle frame only)")
    lines.append(
        "For each SKU that appears in **≥2** segments: pairwise cosine between **middle-of-segment** "
        "unit vectors, and mean Pearson **r** between raw embedding components."
    )
    lines.append("")
    for row in report.get("cross_segment_same_ground_item", []):
        lines.append(
            f"- **{row['true_label']}** (n={row['num_segments']} seg): "
            f"cos_mean={row['middle_frame_pairwise_cos_mean']:.4f} "
            f"min={row['middle_frame_pairwise_cos_min']:.4f} max={row['middle_frame_pairwise_cos_max']:.4f} "
            f"pearson_mean={row['middle_embedding_pearson_mean']:.4f}"
        )
    if not report.get("cross_segment_same_ground_item"):
        lines.append("- (no item with 2+ segments in this video)")
    lines.append("")
    return "\n".join(lines)


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x:.4f}"


def write_reports(
    out_dir: str | Path,
    video_stem: str,
    segments: List[SegmentEmbeddings],
    coverage: Dict[str, Any],
    notes: List[str],
) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    full = run_full_report(segments)
    full["meta"] = {
        "video_stem": video_stem,
        "coverage": coverage,
        "notes": notes,
    }
    jpath = out / f"embedding_analysis_{video_stem}.json"
    mpath = out / f"embedding_analysis_{video_stem}.md"
    with jpath.open("w", encoding="utf-8") as f:
        json.dump(full, f, indent=2, default=str)
    with mpath.open("w", encoding="utf-8") as f:
        f.write(build_markdown(video_stem, full, coverage, notes))
    return jpath, mpath
