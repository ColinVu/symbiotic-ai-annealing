# Placeholder Segments Implementation Summary

## Overview

Implemented explicit placeholder segments to preserve temporal alignment when CARRY_WITH intervals yield zero valid embedded frames. Previously, such intervals were silently dropped or handled through fallback repartitioning logic, which could misalign segments with their corresponding picklist entries.

## Changes Made

### 1. **Segment Dataclass Enhancement** (`weak_supervision.py`)

Added `is_placeholder: bool = False` field to the `Segment` dataclass:

```python
@dataclass
class Segment:
    segment_id: int
    embeddings: np.ndarray
    video_id: str
    candidate_labels: Optional[Tuple[str, ...]] = None
    is_placeholder: bool = False  # NEW
```

### 2. **Segment Creation Logic** (`video_training.py`)

Modified `_group_frames_into_segments()` to:

- **Always create a segment** for each CARRY_WITH interval from manual label CSVs
- If an interval has zero valid frames (all filtered out by blur/hand detection):
  - Create a placeholder segment with a zero embedding vector
  - Mark it with `is_placeholder=True`
  - Assign its `candidate_labels` from the corresponding picklist entry

Key changes:
- Added `pca_dim` parameter to infer embedding dimension for zero vectors
- Creates placeholder segments instead of skipping empty intervals
- Improved verbose output to report placeholder count

### 3. **Fallback Logic Improvement** (`video_training.py`)

Updated segment count mismatch handling in `_process_single_video_to_segments()`:

- **Before**: Complex repartitioning logic redistributed existing embeddings across expected picklist length
- **After**: Simple padding with additional placeholder segments if count is still short
- More informative warning messages distinguish placeholder warnings from alignment errors

### 4. **Cluster Voting Integration** (`cluster_voting.py`)

Modified `cluster_based_initialization_with_details()` to:

- **Filter out** placeholder segments before K-means clustering (they have zero/meaningless embeddings)
- Assign placeholder segments labels from their `candidate_labels` multiset (random choice)
- Set their confidence to `0.0` (no clustering evidence)
- Merge placeholder labels back into the full label dictionary after clustering
- Add verbose logging to report how many placeholders were excluded from clustering

### 5. **ILR (Iterative Label Refinement) Integration** (`weak_supervision.py`)

Modified `refine_labels()` to:

- Create `real_segments` list excluding all placeholders
- Only consider real segments for swap candidates
- Use `real_segments` for all centroid/cost computations during ILR epochs
- Placeholder segments retain their initial labels throughout refinement
- Add verbose logging to report placeholder exclusion

### 6. **PCA Fitting Integration** (`weak_supervision.py`)

Modified `fit()` to:

- Exclude placeholder segments when collecting embeddings for PCA fitting
- After PCA transform, assign zero vectors in PCA space to placeholder segments
- Filter out placeholders for final centroid and std computation
- Enhanced verbose output to distinguish real vs placeholder segment counts

## Behavior Summary

### Before
1. CARRY_WITH interval with no valid frames → segment not created
2. Segment count < expected picklist length → repartition existing embeddings
3. Risk of picklist misalignment and unpredictable segment ID gaps

### After
1. CARRY_WITH interval with no valid frames → placeholder segment created with zero embedding
2. Segment count always matches CARRY_WITH interval count (1:1 correspondence)
3. Placeholders excluded from clustering, ILR swaps, and centroid computation
4. Placeholders retain their position in the sequence, preserving temporal/picklist alignment
5. If final count still short, pad with additional placeholders (rare edge case)

## Testing Recommendations

1. **Test with all-blurry segments**: Video with intervals where all frames are too blurry
2. **Test with hand-detection failures**: Intervals where no valid hand is detected
3. **Test mixed scenarios**: Some intervals valid, some placeholder within same video
4. **Verify CSV output**: Check that `initial_cluster_voting.csv` handles placeholders correctly
5. **Verify inference**: Ensure model can handle videos with placeholder segments during prediction

## Files Modified

1. `symbiote_weak/training/weak_supervision.py`
   - Added `is_placeholder` field to `Segment`
   - Updated `fit()` to exclude placeholders from PCA and final centroid computation
   - Updated `refine_labels()` to exclude placeholders from ILR

2. `symbiote_weak/pipelines/video_training.py`
   - Updated `_group_frames_into_segments()` to create placeholder segments
   - Simplified fallback logic for count mismatches

3. `symbiote_weak/training/cluster_voting.py`
   - Updated `cluster_based_initialization_with_details()` to handle placeholders

## Backward Compatibility

- Existing code that doesn't check `is_placeholder` will see these segments as normal segments with single-frame zero embeddings
- The `mean_embedding` property will return a zero vector for placeholders
- Label assignments for placeholders come from their `candidate_labels`, maintaining picklist consistency
