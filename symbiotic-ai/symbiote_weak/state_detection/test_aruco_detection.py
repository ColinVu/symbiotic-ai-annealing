"""ARUCO Detection Testing Tool.

Processes a video and outputs an annotated version showing:
- Detected ARUCO markers with bounding boxes
- Marker IDs and types (pick/place)
- Frame-by-frame weighted bin context score
- Hand position tracking

Usage::

    python -m symbiote.state_detection.test_aruco_detection \\
        --video path/to/test_video.mp4 \\
        --output path/to/output_annotated.mp4 \\
        --aruco-config config/aruco_bins.json
"""

import argparse
import os
import sys

import cv2
import numpy as np

# Reuse pipeline modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from symbiote.lib.hand_detection import hand_pos
from symbiote.state_detection.aruco_detection import ArucoDetector


def test_aruco_detection(
    video_path: str,
    output_path: str,
    aruco_config_path: str,
    frame_skip: int = 1,
    verbose: bool = True,
) -> None:
    """Process video and create annotated output with ARUCO visualisation.

    Args:
        video_path: Input video file path.
        output_path: Output annotated video file path.
        aruco_config_path: Path to ARUCO configuration JSON.
        frame_skip: Process every Nth frame (default 1 = all frames).
        verbose: Print progress.
    """
    import mediapipe as mp

    aruco_detector = ArucoDetector()
    aruco_detector.load_bin_config(aruco_config_path)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.3,
        max_num_hands=2,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if verbose:
        print(f"Processing video: {video_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Frame skip: {frame_skip}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = fps / frame_skip if frame_skip > 0 else fps
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))

    frame_count = 0
    processed_count = 0
    weights_history: list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        processed_count += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Determine hand position
        hand_position = (width // 2, height // 2)  # fallback
        if results.multi_hand_landmarks:
            all_hand_points = []
            for hlm in results.multi_hand_landmarks:
                pts = [[lm.x, lm.y, lm.z] for lm in hlm.landmark[:21]]
                all_hand_points.append(pts)
            # Pick rightmost hand (leftmost x in mirrored view)
            if len(all_hand_points) > 1:
                positions = [hand_pos(hp, frame_rgb) for hp in all_hand_points]
                idx = min(range(len(positions)), key=lambda i: positions[i][0])
                hand_position = hand_pos(all_hand_points[idx], frame_rgb)
            else:
                hand_position = hand_pos(all_hand_points[0], frame_rgb)

        annotated, weight = aruco_detector.visualize_bin_context(
            frame, hand_position
        )
        weights_history.append(weight)

        # Overlay frame number & weight text
        cv2.putText(
            annotated,
            f"Frame: {frame_count}  Weight: {weight:+.2f}",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        out.write(annotated)

        if verbose and processed_count % 100 == 0:
            print(f"  Processed {processed_count} frames ...")

    cap.release()
    out.release()
    hands.close()

    if verbose:
        print(f"\nDone! Processed {processed_count} frames.")
        print(f"Output saved to: {output_path}")
        if weights_history:
            w = np.array(weights_history)
            print(f"  Weight stats - min: {w.min():+.3f}, max: {w.max():+.3f}, "
                  f"mean: {w.mean():+.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="ARUCO Detection Testing Tool - visualise markers and bin context"
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output annotated video path")
    parser.add_argument("--aruco-config", required=True, help="ARUCO bins JSON")
    parser.add_argument("--frame-skip", type=int, default=1, help="Frame skip (default 1)")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    test_aruco_detection(
        video_path=args.video,
        output_path=args.output,
        aruco_config_path=args.aruco_config,
        frame_skip=args.frame_skip,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
