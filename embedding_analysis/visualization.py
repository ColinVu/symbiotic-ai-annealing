"""Color-coded similarity matrices (PNG + HTML) for cosine sim in [-1,1] (typically [0,1])."""

from __future__ import annotations

import html
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


def _cosine_heatmap_rgba(
    t: float,
) -> Tuple[int, int, int]:
    """Map cosine similarity t in [0, 1] to a green-ish colormap (dark red -> bright green)."""
    t = max(0.0, min(1.0, float(t)))
    # RdYlGn: red(0) -> yellow(0.5) -> green(1)
    if t < 0.5:
        u = t * 2.0
        r = 255
        g = int(round(40 + u * (200 - 40)))
        b = int(round(40 * (1.0 - u)))
    else:
        u = (t - 0.5) * 2.0
        r = int(round(200 * (1.0 - 0.85 * u)))
        g = 220
        b = int(round(40 * (1.0 - u)))
    return (r, g, b)


def _rgb_css(r: int, g: int, b: int) -> str:
    return f"rgb({r},{g},{b})"


def write_similarity_heatmap_png(
    matrix: np.ndarray,
    labels: Sequence[str],
    output_path: str | Path,
    title: str,
    *,
    dpi: int = 300,
    figsize: Tuple[float, float] | None = None,
) -> Path:
    """
    Save a matplotlib heatmap with colorbar (cosine similarity, vmin=0, vmax=1).
    """
    from matplotlib import pyplot as plt

    output_path = Path(output_path)
    a = np.asarray(matrix, dtype=np.float64)
    n = a.shape[0]
    if figsize is None:
        s = min(20.0, max(4.0, 0.4 * n + 2.0))
        figsize = (s, s)

    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    im = ax.imshow(
        a,
        vmin=0.0,
        vmax=1.0,
        cmap="RdYlGn",
        aspect="equal",
        origin="upper",
    )
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    max_label = max((len(str(x)) for x in labels), default=0)
    rot = 90 if n > 8 or max_label > 4 else 0
    ax.set_xticklabels([str(x) for x in labels], rotation=rot, ha="right" if rot else "center", fontsize=7)
    ax.set_yticklabels([str(x) for x in labels], fontsize=7)
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("cosine similarity", fontsize=9)

    show_text = n <= 48
    if show_text:
        for i in range(n):
            for j in range(n):
                v = a[i, j]
                if not np.isfinite(v):
                    t = "—"
                else:
                    t = f"{v:.2f}" if n > 24 else f"{v:.3f}"
                ax.text(
                    j,
                    i,
                    t,
                    ha="center",
                    va="center",
                    color="black" if 0.25 < v < 0.75 else "white",
                    fontsize=5.5 if n > 20 else 7.0,
                )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_similarity_heatmap_html(
    matrix: np.ndarray,
    labels: Sequence[str],
    output_path: str | Path,
    title: str,
) -> Path:
    """
    HTML table with background color per cell and a small legend.
    Assumes values in [0, 1] for display (truncated to [0,1] for color).
    """
    output_path = Path(output_path)
    a = np.asarray(matrix, dtype=np.float64)
    n = a.shape[0]
    rows: List[str] = []
    esc_labels = [html.escape(str(x)) for x in labels]
    for i in range(n):
        tds: List[str] = []
        for j in range(n):
            v = a[i, j]
            v_disp = 0.0 if not np.isfinite(v) else max(0.0, min(1.0, float(v)))
            r, g, b_ = _cosine_heatmap_rgba(v_disp)
            bg = _rgb_css(r, g, b_)
            lum = 0.299 * r + 0.587 * g + 0.114 * b_
            tc = "#000" if lum > 140 else "#fff"
            txt = "—" if not np.isfinite(a[i, j]) else f"{a[i, j]:.3f}"
            tds.append(
                f'<td style="background:{bg};color:{tc};'
                f'text-align:center;padding:4px 6px;'
                f'font-size:0.8em;white-space:nowrap">{html.escape(txt)}</td>'
            )
        rows.append(f"<tr><th>{esc_labels[i]}</th>{''.join(tds)}</tr>")

    header = "".join(
        f"<th>{h}</th>" for h in esc_labels
    )
    legend_steps = 11
    leg_cells = []
    for k in range(legend_steps + 1):
        t = k / legend_steps
        r, g, b_ = _cosine_heatmap_rgba(t)
        leg_cells.append(
            f'<td style="width:40px;height:20px;background:{_rgb_css(r, g, b_)};'
            f'border:1px solid #888"></td>'
        )
    legend = (
        f'<p style="margin:8px 0"><strong>Legend (cosine similarity):</strong> 0.0 (red) → 1.0 (green)</p>'
        f'<table style="border-collapse:collapse;margin:8px 0"><tr>{"".join(leg_cells)}</tr>'
        f'<tr><td colspan="{legend_steps + 1}" style="text-align:left;font-size:0.9em">0.0'
        f'<span style="float:right">1.0</span></td></tr></table>'
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{html.escape(title)}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 1rem; }}
table.bordered {{ border-collapse: collapse; }}
table.bordered th, table.bordered td {{ border: 1px solid #999; }}
</style>
</head>
<body>
<h2>{html.escape(title)}</h2>
{legend}
<table class="bordered" style="margin-top:12px">
<thead>
<tr><th></th>{header}</tr>
</thead>
<tbody>
{chr(10).join(rows)}
</tbody>
</table>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc, encoding="utf-8")
    return output_path
