"""Render ground-truth state overlays for picklist videos.

This utility helps inspect label timing quality by drawing state banners from
`picklist_labels/*.csv` directly onto the corresponding video.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import pandas as pd

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
_STATE_COLORS = {
    "PICK": (0, 220, 0),
    "CARRY_WITH": (0, 220, 220),
    "PLACE": (0, 30, 220),
    "CARRY_EMPTY": (140, 140, 140),
}
_DEFAULT_COLOR = (200, 200, 200)
_BANNER_HEIGHT = 60


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
        raise RuntimeError("No matched video/label pairs found.")
    return pairs


def _load_label_segments(label_csv: str) -> List[Tuple[float, float, str]]:
    out: List[Tuple[float, float, str]] = []
    with open(label_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append((float(row["timestamp_start"]), float(row["timestamp_end"]), row["state"].strip()))
    return out


def _state_at(timestamp: float, segs: List[Tuple[float, float, str]]) -> Optional[str]:
    for s, e, st in segs:
        if s <= timestamp <= e:
            return st
    return None


def _draw_state_banner(frame, state: Optional[str]):
    out = frame.copy()
    h, w = out.shape[:2]
    label = state if state else "UNKNOWN"
    color = _STATE_COLORS.get(label, _DEFAULT_COLOR)
    cv2.rectangle(out, (0, 0), (w, _BANNER_HEIGHT), color, thickness=-1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 3
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    tx = (w - tw) // 2
    ty = (_BANNER_HEIGHT + th) // 2
    cv2.putText(out, label, (tx + 2, ty + 2), font, scale, (0, 0, 0), thickness + 2)
    cv2.putText(out, label, (tx, ty), font, scale, (255, 255, 255), thickness)
    return out


def render_ground_truth_overlays(
    video_dir: str,
    label_dir: str,
    output_dir: str,
    quiet: bool = False,
) -> pd.DataFrame:
    pairs = _discover_pairs(video_dir, label_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for video_path, label_path in pairs:
        segs = _load_label_segments(label_path)
        stem = Path(video_path).stem
        out_video = str(Path(output_dir) / f"{stem}_gt_overlay.mp4")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = cv2.VideoWriter(
            out_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            t = idx / fps if fps > 0 else 0.0
            st = _state_at(t, segs)
            writer.write(_draw_state_banner(frame, st))
        cap.release()
        writer.release()
        rows.append(
            {
                "video": video_path,
                "labels": label_path,
                "output_video": out_video,
                "frames": total,
                "fps": fps,
                "segments": len(segs),
            }
        )
        if not quiet:
            print(f"[hmm_gt_overlay] wrote: {out_video}")

    df = pd.DataFrame(rows)
    out_index = str(Path(output_dir) / "gt_overlay_index.csv")
    df.to_csv(out_index, index=False)
    if not quiet:
        print(f"[hmm_gt_overlay] wrote index: {out_index}")
    return df


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m symbiote.hmm_gt_overlay",
        description="Render ground-truth label overlays for matched picklist videos.",
    )
    p.add_argument("--video-dir", default=str(_ROOT / "hmm-testing" / "picklist_videos"))
    p.add_argument("--label-dir", default=str(_ROOT / "hmm-testing" / "picklist_labels"))
    p.add_argument("--output-dir", default=str(_ROOT / "outputs" / "gt_overlay"))
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    render_ground_truth_overlays(
        video_dir=args.video_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()

