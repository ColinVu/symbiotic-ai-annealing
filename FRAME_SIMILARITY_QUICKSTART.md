# Frame Similarity Visualizer - Quick Start

## What I Created

I've created a new script `frame_similarity_visualizer.py` that:

1. **Extracts middle frames** from every CARRY segment across all videos in your dataset
2. **Applies hand cropping** using the same MediaPipe-based preprocessing as your original pipeline
3. **Applies hand neutralization** using PCA (configurable via `--hand-neutralize` flag)
4. **Creates visual comparison PNGs** showing:
   - A query frame in the top-left
   - ~20 comparison frames (50% same item, 50% different items)
   - Cosine similarity scores for each comparison
   - Color-coded borders (green=same item, red=different item)

## Files Created

- `frame_similarity_visualizer.py` - Main script
- `FRAME_SIMILARITY_VISUALIZER_README.md` - Detailed documentation
- `run_frame_similarity.sh` - Convenience script for easy execution

## Quick Usage

### Option 1: Using the convenience script

```bash
# Run with 20 samples and 50 PCA components for hand neutralization
./run_frame_similarity.sh 20 50

# Run with 10 samples and no hand neutralization
./run_frame_similarity.sh 10 0

# Default (20 samples, 50 components)
./run_frame_similarity.sh
```

### Option 2: Direct Python command

```bash
# With hand neutralization (recommended)
python3 frame_similarity_visualizer.py \
  --models-root models/classifier \
  --manual-labels symbiotic-ai/hmm-testing/picklist_labels \
  --hand-neutralize 50 \
  --n-samples 20 \
  --output-dir frame_similarity_out_50
```

## Command-Line Flags You Requested

✅ **PCA dimension control**: `--hand-neutralize N` (where N is the number of components to remove)
✅ **Number of samples**: `--n-samples N` (number of random query frames to visualize)

## How It Works

1. **Frame Extraction**:
   - Loads cached embeddings for all videos
   - Applies hand neutralization (if enabled)
   - Finds middle frame of each segment
   - If middle frame is unusable, tries next frames until finding a usable one
   - Extracts actual video frame and applies hand crop

2. **Comparison Generation**:
   - For each randomly sampled query frame:
     - Finds other frames with same item label (from ground_truth.csv)
     - Finds frames with different item labels
     - Samples ~50% from each pool
     - Computes cosine similarities
     - Creates visualization PNG

3. **Output**:
   - One PNG per query frame showing the visual comparisons
   - `summary.json` with metadata about the run

## Example Output

Each PNG will look like:
```
┌─────────────────────────────┐
│  Query Frame (large)        │
│  Item: c11                  │
│  Video: picklist_061#seg2   │
└─────────────────────────────┘

┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│ c11   │ │ c11   │ │ c25   │ │ c11   │  <- Green borders = same item
│ 0.856 │ │ 0.743 │ │ 0.234 │ │ 0.698 │  <- Red borders = different item
└───────┘ └───────┘ └───────┘ └───────┘
        ... (more comparison frames)
```

## Requirements

The script uses your existing infrastructure:
- Cached embeddings in `models/classifier/.cache/`
- Ground truth labels in `models/classifier/ground_truth.csv`
- Manual labels in `symbiotic-ai/hmm-testing/picklist_labels/`
- Video files in `symbiotic-ai/hmm-testing/picklist_videos/`
- Hand embeddings in `symbiotic-ai/hmm-testing/hand_embeddings/` (if using neutralization)

## Notes

- **No changes to existing code**: The script is completely standalone
- **Same preprocessing**: Uses identical hand segmentation as your original pipeline
- **Same neutralization**: Uses the same `HandNeutralizer` class as `embedding_analysis`
- **Fallback logic**: If middle frame extraction fails, automatically tries next frames
- **Reproducible**: Set `--seed` for deterministic sampling

## Troubleshooting

If you get import errors, make sure:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/symbiotic-ai"
```

For detailed documentation, see `FRAME_SIMILARITY_VISUALIZER_README.md`.
