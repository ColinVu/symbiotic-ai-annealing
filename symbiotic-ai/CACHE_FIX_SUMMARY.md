# Cache Implementation Fix - Summary

## Problem
Embeddings were being re-computed on EVERY training run, even when cached files existed. This made re-training with multiple videos extremely slow (~12+ hours for 17 videos).

## Root Cause
The `process_video_frames()` function in `video_processor.py` had logic to **write** cache files but **no logic to read** them. Every frame was being:
1. Decoded from video
2. Hand-segmented with MediaPipe
3. Embedded with CLIP
4. Saved to cache (overwriting existing cache)

The cache files were accumulating in `.cache/` directories but were never being used.

## Solution Implemented

### 1. Added `load_frame_from_cache()` function
**File**: `symbiote_weak/embeddings/cache_manager.py`
- New function to read cached embeddings by label + frame number
- Returns `np.ndarray` if cached, `None` if not found
- Uses same MD5 hash-based naming as `save_frame_to_cache()`

### 2. Integrated cache-read into video processing loop
**File**: `symbiote_weak/preprocessing/video_processor.py`
- **Before processing each frame**: Check if embedding exists in cache
- **If cached**: Load embedding directly, skip hand segmentation + CLIP inference
- **If not cached**: Process frame normally (segment + embed + save to cache)
- Added cache hit statistics to verbose output

### 3. Removed redundant parameter
**Files**: `video_processor.py`, `video_training.py`
- Removed `save_frame_to_cache_func` parameter (now imported directly)
- Simplified function signatures across the codebase

## Expected Behavior

### First Run (no cache):
```
✓ First pass complete!
  Frames embedded: 124
  Cache hits: 0 (0.0%)
  New embeddings: 124
```

### Second Run (with cache):
```
✓ First pass complete!
  Frames embedded: 124
  Cache hits: 124 (100.0%)
  New embeddings: 0
```

### Benefits:
- **Multi-video training** now reuses embeddings from previous runs
- Re-training with same videos is **instant** (only ILR runs, no re-embedding)
- Adding 10-15 new videos only embeds the NEW videos (~1 hour each)
- Can experiment with different ILR parameters (epochs, seed) without re-embedding

## Cache Location
- **Single-video train**: `<model_dir>/.cache/<video_stem>/`
- **Multi-video train**: `<model_dir>/.cache/<video_stem>/`
- **Incremental**: `<model_dir>/.incremental_cache/<video_stem>/`
- **Refinement**: `<output_dir>/.cache_refine/<video_stem>/`

## Usage Recommendation
Now that cache works, you can:
1. **Delete old model**: `rm -rf ../models/classifier`
2. **Train on ALL videos** (existing 7 + new 10-15) with ONE command:
   ```bash
   python -m symbiote_weak.cli.main train \
       --videos ./hmm-testing/videos \
       --picklist-json-dir ./hmm-testing/picklist_jsons \
       --manual-labels-dir ./hmm-testing/picklist_labels \
       --output ../models/classifier \
       --pca-dims 128 \
       --ilr-epochs 500 \
       --equal-video-weight \
       --random-seed 42
   ```
3. Embeddings will be cached per video, so re-running only re-does ILR (fast)
4. Joint ILR on all videos = optimal centroids with no sequential bias

## Files Modified
1. `symbiote_weak/embeddings/cache_manager.py` - Added `load_frame_from_cache()`
2. `symbiote_weak/preprocessing/video_processor.py` - Integrated cache-read logic
3. `symbiote_weak/pipelines/video_training.py` - Removed redundant parameter

## Testing
Cache directory already exists with 124 embedding files for picklist_011:
```
/Users/colinhvu/Documents/coding/symbai/022026/models/classifier/.cache/picklist_011/
```
Running the same video again should show 100% cache hits.
