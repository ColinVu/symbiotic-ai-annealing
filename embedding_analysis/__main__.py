"""
Standalone cached-embedding analysis (cosine geometry vs ground truth).

Run from repo root, with symbiotic-ai on PYTHONPATH, e.g.:

  cd 022026  # project root
  python -m embedding_analysis \\
    --models-root models/classifier \\
    --manual-labels symbiotic-ai/hmm-testing/picklist_labels \\
    --videos picklist_121 picklist_061

  # Global item–item matrix: one binary logistic (one-vs-rest) per class, then cos sim of coef rows:
  python -m embedding_analysis ... --log-reg
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from .pipeline import analyze_video


def _default_symbiotic_root() -> Path:
    return Path(__file__).resolve().parent.parent / "symbiotic-ai"


def _resolve(p: str, base: Path) -> Path:
    return Path(p) if os.path.isabs(p) else (base / p).resolve()


def _safe_matrix_stem(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s.strip())
    return s or "item"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cosine-similarity report for cached CLIP frame embeddings")
    ap.add_argument(
        "--models-root",
        type=str,
        default="../models/classifier",
        help="Directory with ground_truth.csv and .cache/{stem}/",
    )
    ap.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="ground_truth.csv path (default: <models-root>/ground_truth.csv)",
    )
    ap.add_argument(
        "--manual-labels",
        type=str,
        required=True,
        help="Directory of {stem}.csv compact state labels (same as training)",
    )
    ap.add_argument(
        "--videos",
        type=str,
        nargs="*",
        help="Video stems to analyze (default: all columns in ground truth)",
    )
    ap.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Directory containing {stem}.MP4 / .mp4 (default: symbiotic-ai/hmm-testing/picklist_videos)",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="embedding_analysis_out",
        help="Where to write JSON/MD per video",
    )
    ap.add_argument(
        "--symbiotic-ai",
        type=str,
        default=None,
        dest="symbiotic_ai",
        help="Path to symbiotic-ai/ (for CARRY interval parsing; default: sibling of embedding_analysis/)",
    )
    ap.add_argument(
        "--frame-skip", type=int, default=4, help="Must match the run that built the cache"
    )
    ap.add_argument(
        "--total-frames",
        type=int,
        default=None,
        help="Override video frame count (if OpenCV cannot read the file)",
    )
    ap.add_argument(
        "--compact-frame-indexing",
        type=str,
        default="opencv0",
        choices=["opencv0", "pipeline1"],
        dest="compact_frame_indexing",
        help="Must match training (compact manual state CSVs)",
    )
    ap.add_argument(
        "--log-reg",
        action="store_true",
        help="Use multinomial logistic regression (class weight rows) for global item–item matrix; "
        "default is spherical-mean centroids per item",
    )
    ap.add_argument(
        "--hand-neutralize",
        type=int,
        default=0,
        dest="hand_neutralize_components",
        help="Apply hand PCA neutralization with n_components (0=disabled, >0=remove top N hand directions)",
    )
    ap.add_argument(
        "--hand-embeddings-dir",
        type=str,
        default=None,
        help="Directory with empty-hand .npy files for PCA fitting (required if --hand-neutralize > 0)",
    )
    args = ap.parse_args(argv)

    here = Path.cwd()
    root = _resolve(args.models_root, here)
    gt = Path(args.ground_truth) if args.ground_truth else (root / "ground_truth.csv")
    manual = _resolve(args.manual_labels, here)
    out = _resolve(args.output_dir, here)
    out.mkdir(parents=True, exist_ok=True)

    sroot = Path(getattr(args, "symbiotic_ai", None) or _default_symbiotic_root()).resolve()
    if not sroot.is_dir():
        ap.error(f"symbiotic-ai not found at {sroot}")

    vid_dir = _resolve(args.video_dir, here) if args.video_dir else (sroot / "hmm-testing" / "picklist_videos")
    cache_base = root / ".cache"
    if not cache_base.is_dir():
        ap.error(f"Cache root not found: {cache_base} (expected per-video .cache/<stem>/)")

    hand_neutralize_n = int(args.hand_neutralize_components)
    hand_emb_dir = None
    if hand_neutralize_n > 0:
        if not args.hand_embeddings_dir:
            hand_emb_dir = sroot / "hmm-testing" / "hand_embeddings"
        else:
            hand_emb_dir = _resolve(args.hand_embeddings_dir, here)
        if not hand_emb_dir.is_dir():
            ap.error(
                f"--hand-neutralize={hand_neutralize_n} requires --hand-embeddings-dir; "
                f"expected at {hand_emb_dir}"
            )

    if args.videos:
        stems = list(args.videos)
    else:
        from .io_ground_truth import list_stems_in_ground_truth

        stems = sorted(x for x in list_stems_in_ground_truth(gt) if x)

    summary: list[dict] = []
    per_video_for_global: list[tuple[str, list]] = []
    for stem in stems:
        cdir = cache_base / stem
        if not cdir.is_dir():
            print(f"[skip] {stem}: no cache at {cdir}", file=sys.stderr)
            continue
        mp4s = [vid_dir / f"{stem}.MP4", vid_dir / f"{stem}.mp4"]
        vpath = next((p for p in mp4s if p.is_file()), None)
        if vpath is None:
            print(f"[skip] {stem}: no video in {vid_dir}", file=sys.stderr)
            continue
        try:
            res = analyze_video(
                video_stem=stem,
                video_path=vpath,
                ground_truth_csv=gt,
                manual_labels_dir=manual,
                cache_dir=cdir,
                output_dir=out,
                symbiotic_ai_root=sroot,
                frame_skip=args.frame_skip,
                frame_indexing=args.compact_frame_indexing,
                total_frames_override=args.total_frames,
                hand_neutralize_components=hand_neutralize_n,
                hand_embeddings_dir=str(hand_emb_dir) if hand_emb_dir else None,
            )
        except Exception as e:
            print(f"[error] {stem}: {e}", file=sys.stderr)
            continue
        per_video_for_global.append((stem, res["segments"]))
        print(json.dumps({"stem": stem, **res["coverage"]}, default=str, indent=2))
        print(f"  wrote: {res['json']}, {res['markdown']}")
        summary.append(
            {
                "stem": stem,
                "json": res["json"],
                "markdown": res["markdown"],
                "coverage": res["coverage"],
            }
        )
    sjson = out / "embedding_analysis_summary.json"
    global_item: dict = {}
    same_item_seg_global: list[dict] = []
    if per_video_for_global:
        from .aggregation import (
            aggregate_all_frames_by_item,
            build_item_to_item_matrix,
            build_item_to_item_matrix_logreg,
            item_centroids_from_aggregated_frames,
        )
        from .metrics import build_global_segment_similarity_by_item
        from .visualization import write_similarity_heatmap_html, write_similarity_heatmap_png

        gseg = build_global_segment_similarity_by_item(per_video_for_global)
        for sm in gseg:
            safe = _safe_matrix_stem(sm.true_label)
            base = out / f"matrix_seg_global_{safe}"
            t = (
                f"Global, all videos in run: same-item segment–segment cos sim "
                f"({sm.true_label}); one row/col per (video#segment) spherical mean"
            )
            row: dict = {
                "item": sm.true_label,
                "n_segment_instances": sm.matrix.shape[0],
                "row_labels": sm.row_labels,
                "video_stem_per_row": sm.video_stem_per_row,
                "segment_idx_per_row": sm.segment_idx_per_row,
                "matrix": sm.matrix.tolist(),
                "png": "",
                "html": "",
            }
            try:
                p_file = write_similarity_heatmap_png(
                    sm.matrix, sm.row_labels, base.with_suffix(".png"), t
                )
                row["png"] = p_file.name
            except Exception as e:
                print(f"[warn] matrix_seg_global {sm.true_label} PNG: {e}", file=sys.stderr)
            try:
                h_file = write_similarity_heatmap_html(
                    sm.matrix, sm.row_labels, base.with_suffix(".html"), t
                )
                row["html"] = h_file.name
            except Exception as e:
                print(f"[warn] matrix_seg_global {sm.true_label} HTML: {e}", file=sys.stderr)
            same_item_seg_global.append(row)

        by_f = aggregate_all_frames_by_item(per_video_for_global)
        if args.log_reg:
            iim: object | None = None
            method = "logistic_regression"
            try:
                iim = build_item_to_item_matrix_logreg(by_f)
            except Exception as e:
                print(
                    f"[warn] item–item log-reg failed ({e}); falling back to centroids.",
                    file=sys.stderr,
                )
                iim = None
            if iim is not None and (not iim.item_labels or iim.matrix.size == 0):
                print(
                    "[warn] item–item log-reg had no class separation; "
                    "falling back to centroids.",
                    file=sys.stderr,
                )
                iim = None
            if iim is None:
                c = item_centroids_from_aggregated_frames(by_f)
                iim = build_item_to_item_matrix(c)
                method = "spherical_mean_centroid_fallback"
        else:
            c = item_centroids_from_aggregated_frames(by_f)
            iim = build_item_to_item_matrix(c)
            method = "spherical_mean_centroid"
        if iim.matrix.size > 0 and iim.item_labels:
            if method == "logistic_regression":
                t = (
                    "Global: item–item cosine (rows = L2-normalized one-vs-rest "
                    "logistic-regression weight vectors, all CARRY-labeled frames)"
                )
            elif method == "spherical_mean_centroid_fallback":
                t = (
                    "Global: item–item cosine (spherical-mean centroids) — log-reg unavailable"
                )
            else:
                t = (
                    "Global: item–item cosine similarity (spherical mean of all "
                    "cached frames per item, all videos/segments)"
                )
            b = out / "matrix_item_to_item_global"
            try:
                p_path = str(
                    write_similarity_heatmap_png(
                        iim.matrix, iim.item_labels, b.with_suffix(".png"), t
                    ).name
                )
            except Exception as e:
                print(f"[warn] item-item PNG: {e}", file=sys.stderr)
                p_path = ""
            try:
                h_path = str(
                    write_similarity_heatmap_html(
                        iim.matrix, iim.item_labels, b.with_suffix(".html"), t
                    ).name
                )
            except Exception as e:
                print(f"[warn] item-item HTML: {e}", file=sys.stderr)
                h_path = ""
            global_item = {
                "n_items": len(iim.item_labels),
                "item_labels": iim.item_labels,
                "matrix": iim.matrix.tolist(),
                "method": method,
                "png": p_path,
                "html": h_path,
            }
    with sjson.open("w", encoding="utf-8") as f:
        payload: dict = {
            "runs": summary,
            "item_to_item_global": global_item,
            "same_item_segment_matrices_global": same_item_seg_global,
        }
        json.dump(payload, f, indent=2, default=str)
    print(f"Summary: {sjson}")
    if same_item_seg_global:
        n = len(same_item_seg_global)
        print(
            f"  same-item global segment–segment heatmaps: {n} item(s) in {out} "
            f"(matrix_seg_global_*.png / .html)"
        )
    if global_item:
        p = global_item.get("png") or ""
        h = global_item.get("html") or ""
        n = global_item.get("n_items", 0)
        print(
            f"  item–item global ({n} items): "
            f"{(out / p).resolve() if p else '—'}, "
            f"{(out / h).resolve() if h else '—'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
