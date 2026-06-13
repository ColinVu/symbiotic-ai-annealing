#!/usr/bin/env python3
"""
Extract every Nth frame from a video and save as JPG.

Usage:
    python extract_frames.py video.mp4 [--output-dir OUTPUT_DIR] [--every N]

Example:
    python extract_frames.py input.mp4 --output-dir frames --every 5
"""

import argparse
import os
import sys
import cv2
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Save every Nth frame from a video as JPG images"
    )
    parser.add_argument(
        "video",
        type=str,
        help="Path to input MP4 (or other video) file"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output folder for images (default: <video_name>_frames)"
    )
    parser.add_argument(
        "--every",
        "-e",
        type=int,
        default=5,
        help="Save every Nth frame (default: 5)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    # Default output dir: same name as video + "_frames"
    if args.output_dir is None:
        base = os.path.splitext(os.path.basename(args.video))[0]
        args.output_dir = base + "_frames"

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_path.absolute()}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video: {args.video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames, {fps:.1f} fps")
    print(f"Saving every {args.every}th frame...")

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.every == 0:
            out_file = out_path / f"frame_{saved:05d}.jpg"
            cv2.imwrite(str(out_file), frame)
            saved += 1
            if saved % 50 == 0:
                print(f"  Saved {saved} images...")

        frame_idx += 1

    cap.release()
    print(f"Done. Saved {saved} images to {out_path.absolute()}")


if __name__ == "__main__":
    main()
