# Video to State Pipeline

This module processes an input video, extracts hand regions, converts each frame into a CLIP embedding, and predicts the hand interaction state sequence using a constrained Hidden Markov Model (HMM).

## Features

- Uses MediaPipe Hands to crop hand regions frame-by-frame.
- Generates CLIP embeddings for every cropped frame with live progress reporting.
- Reduces embedding dimensionality (capped at 356 dimensions) via PCA for efficient HMM inference.
- Runs a deterministic-cycle HMM across the four interaction states:
  - `a` → `pick`
  - `e` → `carry`
  - `i` → `place`
  - `m` → `carry_empty`
- Produces a timeline (`CSV`) indicating the start time, end time, and state for each segment, starting from the first detected `carry` state.

## Installation

1. Ensure Python 3.10+ is available.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** If you manage environments manually, install the packages listed in `requirements.txt` inside the environment where you plan to run the pipeline.

## Usage

```bash
python -m video_to_state.process_video \
  --video path/to/video.mp4 \
  --output path/to/output.csv
```

Optional arguments:

- `--model-name`: Hugging Face model id for CLIP (default: `openai/clip-vit-base-patch32`).
- `--batch-size`: Number of frames to embed per batch (default: `16`).
- `--max-dim`: Maximum embedding dimensionality after PCA (default: `356`).
- `--skip-no-hand`: If provided, frames without detected hands will be excluded from HMM inference (otherwise they inherit the last known state).
- `--cycle-strength`: Baseline probability of advancing to the next state in the cycle (default: `0.10`).
- `--orientation-threshold`: Minimum orientation change (after smoothing) required before advancing to the next state (default: `0.30`).
- `--orientation-smoothing`: Window size (frames) used to smooth raw orientation vectors (default: `5`).
- `--up-threshold`: Cosine threshold (vs. the upward axis) to classify the palm as “upward” (default: `0.35`).
- `--down-threshold`: Cosine threshold to fall back to “not upward”, creating hysteresis (default: `0.20`).
- `--orientation-weight`: Multiplier applied to the 3D orientation vector before concatenating it with CLIP embeddings for the HMM (default: `25.0`).

## Output Format

The CSV contains the following columns:

| start_time | end_time | state_symbol | state_name |
|------------|----------|--------------|------------|
| 0.000      | 1.067    | a            | pick       |
| 1.067      | 2.533    | e            | carry      |
| ...        | ...      | ...          | ...        |

All timestamps are expressed in seconds.

## Notes

- The CLIP model weights are downloaded on first use; ensure internet access during the initial run.
- MediaPipe Hands performs best on RGB frames sized around 720p–1080p. Very small or very large frames may reduce detection accuracy.
- The HMM enforces the cycle `pick → carry → place → carry_empty → pick`. Frames without a detected hand inherit the most recent state.

## Annotating a Video with States

After generating the CSV timeline, you can overlay the predicted states onto the original video:

```bash
python -m video_to_state.annotate_video \
  --video path/to/video.mp4 \
  --csv path/to/output.csv \
  --output path/to/video_annotated.mp4
```

If `--output` is omitted, the script writes alongside the source video using the suffix `_annotated`.


