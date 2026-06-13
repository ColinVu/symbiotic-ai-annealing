# Frame Similarity Visualizer - Implementation Summary

## What Was Created

I've created a complete frame similarity visualization system that extracts middle frames from video segments and creates visual comparison PNGs showing cosine similarities.

## Files Created (No Existing Files Modified)

### 1. Main Script
- **`frame_similarity_visualizer.py`** (617 lines)
  - Core implementation
  - Extracts middle frames from all CARRY segments
  - Applies hand cropping and neutralization
  - Creates comparison PNGs with cosine similarities
  - CLI interface with all requested flags

### 2. Documentation
- **`FRAME_SIMILARITY_QUICKSTART.md`**
  - Quick start guide
  - Usage examples
  - Command-line flags
  
- **`FRAME_SIMILARITY_VISUALIZER_README.md`**
  - Complete documentation
  - All command-line options
  - Troubleshooting guide
  - Output format description

### 3. Helper Scripts
- **`run_frame_similarity.sh`**
  - Convenience wrapper for common usage
  - Usage: `./run_frame_similarity.sh [n_samples] [hand_neutralize_components]`
  - Example: `./run_frame_similarity.sh 20 50`

- **`test_frame_similarity.py`**
  - Component tests
  - Validates imports and core functions
  - Run: `python3 test_frame_similarity.py`

## Key Features Implemented

### ✅ All Requirements Met

1. **Middle frame extraction**: ✓
   - Takes middle frame from every segment across all videos
   - Falls back to next available frame if middle frame unusable
   - Uses same hand cropping as original preprocessing

2. **Video frame extraction**: ✓
   - Extracts actual frames from video files
   - Applies MediaPipe hand segmentation
   - Saves hand-cropped images

3. **Hand neutralization**: ✓
   - Applies `HandNeutralizer` to embeddings
   - PCA dimension configurable via `--hand-neutralize N` flag

4. **Visual comparisons**: ✓
   - Creates page-sized PNGs
   - Query frame in top-left
   - ~20 comparison frames (configurable)
   - 50% same-item, 50% different-item frames
   - Cosine similarity scores displayed
   - Color-coded borders (green=same, red=different)

5. **Random sampling**: ✓
   - Number of PNGs controlled by `--n-samples N` flag
   - Reproducible with `--seed` flag

6. **No existing file changes**: ✓
   - All new files only
   - Uses existing `embedding_analysis` modules as library

## Usage Examples

### Basic usage:
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

### Using convenience script:
```bash
# 20 samples, 50 PCA components
./run_frame_similarity.sh 20 50

# 10 samples, no hand neutralization
./run_frame_similarity.sh 10 0
```

## Technical Implementation

### Architecture
```
frame_similarity_visualizer.py
├── MiddleFrameExtractor (class)
│   └── extract_all_middle_frames()
│       ├── Loads cached embeddings
│       ├── Applies hand neutralization (optional)
│       ├── Builds segments
│       ├── Extracts middle frame from video
│       └── Returns list of frame data
│
├── extract_frame_from_video()
│   ├── Extracts specific frame from video
│   └── Applies hand segmentation
│
├── compute_cosine_similarity()
│   └── L2 normalize and dot product
│
├── create_comparison_png()
│   ├── Layouts grid of comparison frames
│   ├── Color-codes borders
│   └── Displays cosine similarities
│
└── main()
    ├── Parses CLI arguments
    ├── Extracts all middle frames
    ├── Samples query frames
    ├── For each query:
    │   ├── Samples same-item frames
    │   ├── Samples different-item frames
    │   ├── Computes similarities
    │   └── Creates PNG
    └── Writes summary.json
```

### Dependencies
- `embedding_analysis` (local modules)
- `symbiotic-ai` (on PYTHONPATH for hand detection)
- `numpy`, `cv2`, `PIL`, `mediapipe`

### Data Flow
1. Load cached embeddings from `.cache/{video_stem}/`
2. Apply hand neutralization (PCA-based)
3. Build segments using ground truth labels
4. Extract middle frame from video file
5. Apply hand segmentation to frame
6. Collect all frames with embeddings
7. Sample query frames randomly
8. For each query: sample comparisons (50/50 split)
9. Compute cosine similarities
10. Generate visual comparison PNG

## Output Format

### Directory Structure
```
frame_similarity_out_50/
├── summary.json
├── similarity_000_picklist_061_seg0_c25.png
├── similarity_001_picklist_071_seg2_c23.png
├── similarity_002_picklist_111_seg1_c25.png
└── ...
```

### PNG Layout
```
┌────────────────────────────────────────┐
│                                        │
│  Query Frame (300x300)                 │
│  Label: c11                            │
│  picklist_061#seg2 frame1234           │
│                                        │
└────────────────────────────────────────┘

┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│         │ │         │ │         │ │         │
│  150px  │ │  150px  │ │  150px  │ │  150px  │
│  GREEN  │ │  GREEN  │ │  RED    │ │  GREEN  │
│  c11    │ │  c11    │ │  c25    │ │  c11    │
│ cos=0.8 │ │ cos=0.7 │ │ cos=0.2 │ │ cos=0.7 │
└─────────┘ └─────────┘ └─────────┘ └─────────┘
      ... (4 columns × N rows)
```

## Testing

Run component tests:
```bash
python3 test_frame_similarity.py
```

Expected output:
```
✓ All tests passed! The script should work correctly.
```

## Notes

- **Performance**: Processing all videos takes time (proportional to dataset size)
- **Memory**: Keeps all extracted frames in memory; reduce `--n-samples` if issues occur
- **Reproducibility**: Use `--seed 42` for deterministic sampling
- **Hand neutralization**: Recommended to use 50 components (same as your analysis)

## Next Steps

1. Run the test: `python3 test_frame_similarity.py`
2. Run a small test: `./run_frame_similarity.sh 5 50`
3. Check output in `frame_similarity_out_50/`
4. Run full analysis: `./run_frame_similarity.sh 20 50`
5. Examine PNGs to evaluate embedding quality visually

## Comparison with embedding_analysis

| Feature | embedding_analysis | frame_similarity_visualizer |
|---------|-------------------|---------------------------|
| Purpose | Numerical similarity analysis | Visual similarity analysis |
| Input | Cached embeddings | Cached embeddings + videos |
| Output | Heatmap matrices (PNG) | Frame comparisons (PNG) |
| Hand neutralization | ✓ | ✓ |
| Segment-level | ✓ | ✓ |
| Frame-level visual | ✗ | ✓ |
| Interactive | ✗ | ✓ (visual inspection) |

Both tools complement each other—use `embedding_analysis` for quantitative metrics and `frame_similarity_visualizer` for qualitative visual inspection.
