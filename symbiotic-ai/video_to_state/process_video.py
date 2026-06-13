from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
from tqdm import tqdm

from .clip_embedder import ClipEmbedder, normalize_embeddings
from .hand_segmentation import HandCrop, HandSegmenter
from .hmm_model import HandStateHMM, STATE_LABELS, summarise_states


@dataclass
class FrameRecord:
    index: int
    timestamp: float
    crop: Optional[HandCrop]


def _load_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {path}")
    return cap


def _collect_frames(video_path: Path, skip_no_hand: bool) -> tuple[List[FrameRecord], float]:
    cap = _load_video(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_duration = 1.0 / fps

    records: List[FrameRecord] = []

    with HandSegmenter() as segmenter, tqdm(total=total_frames or None, desc="Extracting hands") as pbar:
        frame_index = 0
        while True:
            success, frame_bgr = cap.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            crop = segmenter.extract_hand(frame_rgb)

            if crop is None and skip_no_hand:
                pbar.update(1)
                frame_index += 1
                continue

            records.append(
                FrameRecord(
                    index=frame_index,
                    timestamp=frame_index * frame_duration,
                    crop=crop,
                )
            )

            frame_index += 1
            pbar.update(1)

    cap.release()
    return records, frame_duration


def _prepare_embeddings(records: Sequence[FrameRecord], embedder: ClipEmbedder, batch_size: int) -> np.ndarray:
    crops = [record.crop.image for record in records if record.crop is not None]
    if not crops:
        return np.empty((0, 0))

    with tqdm(total=len(crops), desc="Embedding frames") as pbar:
        embeddings: List[np.ndarray] = []
        for start in range(0, len(crops), batch_size):
            batch = crops[start : start + batch_size]
            batch_embeddings = embedder.embed(batch, batch_size=batch_size)
            embeddings.append(batch_embeddings)
            pbar.update(len(batch))

    embedded = normalize_embeddings(np.vstack(embeddings))
    return embedded


UP_AXIS = np.array([0.0, 0.0, 1.0])  # Use Z component for upward detection


def _align_embeddings(records: Sequence[FrameRecord], embeddings: np.ndarray) -> tuple[np.ndarray, List[float], Optional[np.ndarray]]:
    timestamps: List[float] = []
    usable_embeddings: List[np.ndarray] = []
    orientations: List[np.ndarray] = []

    embedding_index = 0
    for record in records:
        if record.crop is None:
            continue

        usable_embeddings.append(embeddings[embedding_index])
        timestamps.append(record.timestamp)
        
        # Collect orientation if available
        if record.crop.orientation is not None:
            orientations.append(record.crop.orientation)
        else:
            orientations.append(np.array([np.nan, np.nan, np.nan]))
        
        embedding_index += 1

    embeddings_array = np.vstack(usable_embeddings)
    orientations_array = np.vstack(orientations) if orientations else None

    if orientations_array is not None:
        # Replace zero-length vectors with NaNs
        norms = np.linalg.norm(orientations_array, axis=1)
        invalid = norms < 1e-6
        orientations_array[invalid] = np.array([np.nan, np.nan, np.nan])

    return embeddings_array, timestamps, orientations_array


def _combine_embeddings(
    embeddings: np.ndarray,
    orientations: Optional[np.ndarray],
    orientation_weight: float,
) -> np.ndarray:
    if orientations is None or orientation_weight <= 0:
        return embeddings

    orientations_clean = np.nan_to_num(orientations, nan=0.0)
    weighted_orientation = orientations_clean * orientation_weight
    return np.hstack([embeddings, weighted_orientation])


def _smooth_orientation_vectors(orientations: np.ndarray, window: int) -> np.ndarray:
    if orientations is None or len(orientations) == 0:
        return orientations

    if window <= 1:
        return orientations.copy()

    smoothed = []
    previous_valid: Optional[np.ndarray] = None

    for i in range(len(orientations)):
        start = max(0, i - window // 2)
        end = min(len(orientations), i + window // 2 + 1)
        window_vecs = orientations[start:end]
        valid_mask = ~np.isnan(window_vecs).any(axis=1)
        valid_vecs = window_vecs[valid_mask]

        if valid_vecs.size == 0:
            if previous_valid is not None:
                smoothed.append(previous_valid)
            else:
                smoothed.append(np.array([np.nan, np.nan, np.nan]))
            continue

        averaged = np.mean(valid_vecs, axis=0)
        norm = np.linalg.norm(averaged)
        if norm < 1e-6:
            if previous_valid is not None:
                smoothed.append(previous_valid)
            else:
                smoothed.append(np.array([np.nan, np.nan, np.nan]))
            continue

        averaged /= norm
        smoothed.append(averaged)
        previous_valid = averaged

    return np.array(smoothed)


def _compute_upward_flags(
    orientations: Optional[np.ndarray],
    smoothing_window: int,
    up_threshold: float,
    down_threshold: float,
) -> Optional[List[bool]]:
    if orientations is None:
        return None

    smoothed = _smooth_orientation_vectors(orientations, smoothing_window)
    if smoothed is None:
        return None

    flags: List[bool] = []
    current_up = False

    for vec in smoothed:
        if np.isnan(vec).any():
            flags.append(current_up)
            continue

        dot = float(np.dot(vec, UP_AXIS))

        if current_up:
            if dot < down_threshold:
                current_up = False
        else:
            if dot > up_threshold:
                current_up = True

        flags.append(current_up)

    return flags


def _derive_states_from_flags(up_flags: Optional[List[bool]]) -> Optional[List[str]]:
    if up_flags is None:
        return None

    cycle = ["e", "i", "m", "a"]  # carry, place, empty, pick
    phase = len(cycle) - 1  # Start from "pick" before first upward event
    states: List[str] = []
    started = False

    for is_up in up_flags:
        if not started:
            if is_up:
                phase = 2  # enter "carry_empty"
                started = True
        else:
            expecting_up = (phase % 2 == 0)  # carry/empty expect up orientation
            if expecting_up and not is_up:
                phase = (phase + 1) % len(cycle)
            elif not expecting_up and is_up:
                phase = (phase + 1) % len(cycle)

        current_state = cycle[phase]
        states.append(current_state)

    return states


def process_video(
    video_path: Path,
    output_path: Path,
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 16,
    max_dim: int = 356,
    skip_no_hand: bool = False,
    cycle_strength: float = 0.10,
    orientation_threshold: float = 0.3,
    orientation_smoothing: int = 5,
    upward_threshold: float = 0.35,
    downward_threshold: float = 0.2,
    orientation_weight: float = 25.0,
) -> None:
    records, frame_duration = _collect_frames(video_path, skip_no_hand=skip_no_hand)

    if not records:
        raise RuntimeError("No frames available for processing.")

    embedder = ClipEmbedder(model_name=model_name)
    embeddings = _prepare_embeddings(records, embedder, batch_size=batch_size)

    if embeddings.size == 0:
        raise RuntimeError("Hand segmentation failed for all frames.")

    usable_embeddings, timestamps, orientations = _align_embeddings(records, embeddings)
    combined_embeddings = _combine_embeddings(usable_embeddings, orientations, orientation_weight)

    hmm = HandStateHMM(
        max_dim=max_dim,
        cycle_strength=cycle_strength,
        orientation_threshold=orientation_threshold,
        orientation_smoothing=orientation_smoothing,
    )
    result = hmm.infer(combined_embeddings, timestamps=timestamps, orientations=orientations)

    orientation_states = None
    if orientations is not None:
        up_flags = _compute_upward_flags(
            orientations,
            smoothing_window=orientation_smoothing,
            up_threshold=upward_threshold,
            down_threshold=downward_threshold,
        )
        orientation_states = _derive_states_from_flags(up_flags)

    state_sequence = orientation_states or result.state_sequence
    if not state_sequence:
        raise RuntimeError("Unable to determine state sequence.")

    state_iter = iter(state_sequence)
    frame_states: List[str] = []
    last_state = state_sequence[0]
    for record in records:
        if record.crop is None:
            frame_states.append(last_state)
        else:
            try:
                current_state = next(state_iter)
            except StopIteration:
                raise RuntimeError("Mismatch between embeddings and frame records.")
            last_state = current_state
            frame_states.append(current_state)

    try:
        next(state_iter)
        raise RuntimeError("More HMM states produced than frame records.")
    except StopIteration:
        pass

    timestamps_full = [record.timestamp for record in records]
    segments = summarise_states(frame_states, timestamps_full, frame_duration=frame_duration)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["start_time", "end_time", "state_symbol", "state_name"])
        writer.writeheader()
        for row in segments:
            writer.writerow(row)

    print(f"Wrote {len(segments)} segments to {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert video hand interactions to state timeline.")
    parser.add_argument("--video", type=Path, required=True, help="Path to the input MP4 video.")
    parser.add_argument("--output", type=Path, required=True, help="Destination CSV file.")
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32", help="Hugging Face model id.")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of frames to embed per batch.")
    parser.add_argument("--max-dim", type=int, default=356, help="Maximum dimension after PCA reduction.")
    parser.add_argument("--skip-no-hand", action="store_true", help="Drop frames without detected hands.")
    parser.add_argument("--cycle-strength", type=float, default=0.10, help="Base probability of transitioning to next state (0.0-1.0).")
    parser.add_argument("--orientation-threshold", type=float, default=0.3, help="Minimum orientation change to allow state transition (0.0-1.0).")
    parser.add_argument("--orientation-smoothing", type=int, default=5, help="Number of frames to average for orientation smoothing.")
    parser.add_argument("--up-threshold", type=float, default=0.35, help="Cosine threshold to classify palm as upward.")
    parser.add_argument("--down-threshold", type=float, default=0.20, help="Cosine threshold to classify palm as not upward.")
    parser.add_argument("--orientation-weight", type=float, default=25.0, help="Weight applied to orientation vector when combining features for the HMM.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    process_video(
        video_path=args.video,
        output_path=args.output,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_dim=args.max_dim,
        skip_no_hand=args.skip_no_hand,
        cycle_strength=args.cycle_strength,
        orientation_threshold=args.orientation_threshold,
        orientation_smoothing=args.orientation_smoothing,
        upward_threshold=args.up_threshold,
        downward_threshold=args.down_threshold,
        orientation_weight=args.orientation_weight,
    )


if __name__ == "__main__":
    main()


