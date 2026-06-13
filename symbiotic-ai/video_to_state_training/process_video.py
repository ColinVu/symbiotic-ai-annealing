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


def _align_embeddings(records: Sequence[FrameRecord], embeddings: np.ndarray) -> tuple[np.ndarray, List[float]]:
    timestamps: List[float] = []
    usable_embeddings: List[np.ndarray] = []

    embedding_index = 0
    for record in records:
        if record.crop is None:
            continue

        usable_embeddings.append(embeddings[embedding_index])
        timestamps.append(record.timestamp)
        embedding_index += 1

    embeddings_array = np.vstack(usable_embeddings)
    return embeddings_array, timestamps


def process_video(
    video_path: Path,
    output_path: Path,
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 16,
    max_dim: int = 356,
    skip_no_hand: bool = False,
    cycle_strength: float = 0.10,
    n_iter: int = 100,
    hmm_model: Optional[HandStateHMM] = None,
) -> None:
    """Process a video and output state predictions.
    
    Args:
        video_path: Path to input video
        output_path: Path to output CSV file
        model_name: CLIP model name
        batch_size: Batch size for embedding
        max_dim: Maximum PCA dimensions
        skip_no_hand: Skip frames without detected hands
        cycle_strength: HMM cycle strength parameter
        n_iter: Number of HMM training iterations (if model not provided)
        hmm_model: Optional pre-trained HMM model. If None, creates a new one.
    """
    records, frame_duration = _collect_frames(video_path, skip_no_hand=skip_no_hand)

    if not records:
        raise RuntimeError("No frames available for processing.")

    embedder = ClipEmbedder(model_name=model_name)
    embeddings = _prepare_embeddings(records, embedder, batch_size=batch_size)

    if embeddings.size == 0:
        raise RuntimeError("Hand segmentation failed for all frames.")

    usable_embeddings, timestamps = _align_embeddings(records, embeddings)

    if hmm_model is None:
        hmm = HandStateHMM(
            max_dim=max_dim,
            cycle_strength=cycle_strength,
            n_iter=0,  # No training, just initialization
        )
        # For inference without training, we need to initialize the model
        # This will use circular assignment
        reduced = hmm._reduce(usable_embeddings)  # Fit PCA
        hmm.model = hmm._build_model(reduced)
    else:
        hmm = hmm_model

    result = hmm.infer(usable_embeddings, timestamps=timestamps)

    state_sequence = result.state_sequence
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
    parser.add_argument("--n-iter", type=int, default=100, help="Number of HMM training iterations.")
    parser.add_argument("--hmm-model", type=Path, default=None, help="Path to saved HMM model (pickle file).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    hmm_model = None
    if args.hmm_model:
        import pickle
        with open(args.hmm_model, "rb") as f:
            hmm_model = pickle.load(f)
    
    process_video(
        video_path=args.video,
        output_path=args.output,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_dim=args.max_dim,
        skip_no_hand=args.skip_no_hand,
        cycle_strength=args.cycle_strength,
        n_iter=args.n_iter,
        hmm_model=hmm_model,
    )


if __name__ == "__main__":
    main()


