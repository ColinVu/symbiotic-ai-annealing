# Frame Similarity Visualizer

This script extracts middle frames from video segments and creates visual comparison PNGs showing cosine similarities between frames.

## Features

- Extracts the middle frame from every CARRY segment across all videos in the dataset
- Falls back to the next available frame if the middle frame is unusable
- Applies hand cropping (same preprocessing as the original embedding pipeline)
- Applies hand neutralization using PCA (configurable number of components)
- Creates visual comparison PNGs with:
  - Query frame in top-left
  - ~50% comparison frames from the same item
  - ~50% comparison frames from different items
  - Cosine similarity scores displayed for each comparison
  - Color-coded borders (green=same item, red=different item)

## Requirements

- Python 3.8+
- Dependencies from `embedding_analysis`
- `symbiotic-ai` on PYTHONPATH
- MediaPipe for hand detection
- PIL/Pillow for image generation

## Usage

### Basic usage (no hand neutralization):

```bash
python3 frame_similarity_visualizer.py \
  --models-root models/classifier \
  --manual-labels symbiotic-ai/hmm-testing/picklist_labels \
  --n-samples 20
```

### With hand neutralization (recommended):

```bash
python3 frame_similarity_visualizer.py \
  --models-root models/classifier \
  --manual-labels symbiotic-ai/hmm-testing/picklist_labels \
  --hand-neutralize 50 \
  --n-samples 20 \
  --output-dir frame_similarity_out_50
```

### Full options:

```bash
python3 frame_similarity_visualizer.py \
  --models-root models/classifier \
  --ground-truth models/classifier/ground_truth.csv \
  --manual-labels symbiotic-ai/hmm-testing/picklist_labels \
  --video-dir symbiotic-ai/hmm-testing/picklist_videos \
  --output-dir frame_similarity_out \
  --symbiotic-ai symbiotic-ai \
  --frame-skip 4 \
  --compact-frame-indexing opencv0 \
  --hand-neutralize 50 \
  --hand-embeddings-dir symbiotic-ai/hmm-testing/hand_embeddings \
  --n-samples 20 \
  --comparisons-per-query 20 \
  --seed 42
```

## Command-line Arguments

### Required:
- `--manual-labels`: Directory containing `{video_stem}.csv` files with manual state labels

### Optional:
- `--models-root`: Directory with `ground_truth.csv` and `.cache/{stem}/` (default: `models/classifier`)
- `--ground-truth`: Path to ground_truth.csv (default: `<models-root>/ground_truth.csv`)
- `--video-dir`: Directory containing video files (default: `symbiotic-ai/hmm-testing/picklist_videos`)
- `--output-dir`: Where to save output PNGs (default: `frame_similarity_out`)
- `--symbiotic-ai`: Path to symbiotic-ai directory (default: auto-detect sibling directory)
- `--frame-skip`: Frame skip value, must match cache generation (default: 4)
- `--compact-frame-indexing`: Frame indexing mode (default: `opencv0`, options: `opencv0`, `pipeline1`)
- `--hand-neutralize`: Number of PCA components to remove for hand neutralization (default: 0, disabled)
- `--hand-embeddings-dir`: Directory with empty-hand embeddings for PCA (default: `symbiotic-ai/hmm-testing/hand_embeddings`)
- `--n-samples`: Number of random query frames to visualize (default: 20)
- `--comparisons-per-query`: Number of comparison frames per query visualization (default: 20)
- `--seed`: Random seed for reproducible sampling (default: 42)

## Output

The script creates:

1. **Comparison PNGs**: Named `similarity_{idx:03d}_{video_stem}_seg{segment_idx}_{item_label}.png`
   - Each PNG shows one query frame and up to 20 comparison frames
   - Comparison frames are sorted by cosine similarity (highest first)
   - Green border = same item, Red border = different item
   - Cosine similarity score shown below each comparison frame

2. **summary.json**: Contains:
   - Total frames extracted
   - Number of unique items
   - Number of visualizations created
   - Hand neutralization settings
   - Frame count per item

## Example Output Structure

```
frame_similarity_out/
├── summary.json
├── similarity_000_picklist_061_seg0_c25.png
├── similarity_001_picklist_071_seg2_c23.png
├── similarity_002_picklist_111_seg1_c25.png
└── ...
```

## How It Works

1. **Extract Middle Frames**: For each CARRY segment in each video:
   - Load cached embeddings
   - Apply hand neutralization (if enabled)
   - Find the middle frame index
   - Extract the actual video frame
   - Apply hand segmentation/cropping
   - Store frame, embedding, and metadata

2. **Sample Query Frames**: Randomly select N frames to visualize

3. **For Each Query Frame**:
   - Find other frames with the same item label (same-item pool)
   - Find frames with different item labels (different-item pool)
   - Sample ~50% from same-item pool, ~50% from different-item pool
   - Compute cosine similarities between query and all comparison frames
   - Create visualization PNG

## Notes

- The script uses the same hand segmentation method as the original embedding pipeline (MediaPipe)
- Hand neutralization is applied to embeddings after loading from cache (same as `embedding_analysis`)
- If a middle frame cannot be extracted (e.g., hand detection fails), the script tries subsequent frames in the segment
- Frames are sampled randomly, so different runs with different seeds will produce different visualizations
- The script requires pre-computed embeddings in the `.cache/` directory

## Troubleshooting

**"No frames extracted!"**: 
- Check that video files exist in `--video-dir`
- Check that cache directories exist in `models/classifier/.cache/{video_stem}/`
- Check that manual label CSVs exist in `--manual-labels`

**"Hand neutralizer could not be enabled"**:
- Verify that `--hand-embeddings-dir` points to a directory with `.npy` files
- Check that there are at least 2 empty-hand embedding files

**Import errors**:
- Ensure `symbiotic-ai` is on your PYTHONPATH
- Ensure `embedding_analysis` module is in the same directory as the script
