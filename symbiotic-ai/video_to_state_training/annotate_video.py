from __future__ import annotations

import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from .hand_segmentation import HandSegmenter


Segment = Dict[str, float | str]

STATE_COLOURS = {
    "a": (0, 255, 0),  # green
    "e": (255, 165, 0),  # orange
    "i": (0, 0, 255),  # red
    "m": (255, 0, 255),  # magenta
}


def _load_segments(csv_path: Path) -> List[Segment]:
    with csv_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        segments: List[Segment] = []
        for row in reader:
            segments.append(
                {
                    "start_time": float(row["start_time"]),
                    "end_time": float(row["end_time"]),
                    "state_symbol": row["state_symbol"],
                    "state_name": row["state_name"],
                }
            )
    if not segments:
        raise ValueError(f"No rows found in {csv_path}")
    return segments


def _pick_segment(segments: List[Segment], timestamp: float, current_index: int) -> int:
    idx = current_index
    while idx < len(segments) and timestamp >= segments[idx]["end_time"]:
        idx += 1
    if idx >= len(segments):
        return len(segments) - 1
    if timestamp >= segments[idx]["start_time"]:
        return idx
    # timestamp before first segment start
    return 0


def annotate_video(
    video_path: Path,
    csv_path: Path,
    output_path: Optional[Path] = None,
    font_scale: float = 1.0,
    thickness: int = 2,
    margin: int = 32,
) -> Path:
    """Write an annotated copy of `video_path` using the state timeline in `csv_path`.

    Returns the path to the annotated video.
    """
    if output_path is None:
        output_path = video_path.with_name(f"{video_path.stem}_annotated{video_path.suffix}")

    segments = _load_segments(csv_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open writer for {output_path}")

    idx = 0
    frame_idx = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    with HandSegmenter() as segmenter:
        with tqdm(total=total_frames or None, desc="Annotating video") as pbar:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                timestamp = frame_idx / fps
                idx = _pick_segment(segments, timestamp, idx)
                segment = segments[idx]

                symbol = segment["state_symbol"]
                name = segment["state_name"]
                colour = STATE_COLOURS.get(symbol, (255, 255, 255))

                # Draw bounding box if a hand is detected
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                crop = segmenter.extract_hand(frame_rgb)
                if crop is not None:
                    left, top, right, bottom = crop.bounding_box
                    cv2.rectangle(frame, (left, top), (right, bottom), colour, max(2, thickness))

                    # Draw thumb and middle finger landmarks
                    landmarks = crop.landmarks
                    thumb_tip = landmarks[4]
                    middle_tip = landmarks[12]
                    palm = landmarks[0]

                    thumb_pt = (
                        int(thumb_tip[0] * width),
                        int(thumb_tip[1] * height),
                    )
                    middle_pt = (
                        int(middle_tip[0] * width),
                        int(middle_tip[1] * height),
                    )
                    palm_pt = (
                        int(palm[0] * width),
                        int(palm[1] * height),
                    )

                    cv2.circle(frame, thumb_pt, 6, (0, 255, 255), -1)  # yellow
                    cv2.circle(frame, middle_pt, 6, (255, 0, 0), -1)  # blue
                    cv2.circle(frame, palm_pt, 6, (0, 255, 0), -1)  # green for palm
                    cv2.line(frame, palm_pt, thumb_pt, (255, 255, 255), 2)
                    cv2.line(frame, palm_pt, middle_pt, (255, 255, 255), 2)

                    if crop.orientation is not None:
                        orientation_vec = crop.orientation
                        orientation_str = f"dir: [{orientation_vec[0]:+.2f}, {orientation_vec[1]:+.2f}, {orientation_vec[2]:+.2f}]"
                        cv2.putText(
                            frame,
                            orientation_str,
                            (left, max(0, top - margin // 2)),
                            font,
                            font_scale * 0.6,
                            colour,
                            max(1, thickness - 1),
                            lineType=cv2.LINE_AA,
                        )

                label = f"{symbol.upper()} - {name}"

                cv2.putText(
                    frame,
                    label,
                    (margin, margin + int(font_scale * 20)),
                    font,
                    font_scale,
                    colour,
                    thickness,
                    lineType=cv2.LINE_AA,
                )

                writer.write(frame)
                frame_idx += 1
                pbar.update(1)

    cap.release()
    writer.release()

    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate a video with hand-state labels.")
    parser.add_argument("--video", type=Path, required=True, help="Path to the source video.")
    parser.add_argument("--csv", type=Path, required=True, help="CSV produced by process_video.")
    parser.add_argument("--output", type=Path, default=None, help="Destination video path.")
    parser.add_argument("--font-scale", type=float, default=1.0, help="Scaling factor for annotation text.")
    parser.add_argument("--thickness", type=int, default=2, help="Thickness of the annotation text.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    annotate_video(
        video_path=args.video,
        csv_path=args.csv,
        output_path=args.output,
        font_scale=args.font_scale,
        thickness=args.thickness,
    )


if __name__ == "__main__":
    main()


