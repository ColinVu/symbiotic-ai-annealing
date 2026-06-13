# Ablation Testing Guide

## Overview
After the weak HMM upgrade, you need to identify which changes caused performance regression.

## Quick Start (Recommended)

**Run the fast test first** to check if constraints are the issue:

```bash
cd symbiotic-ai
./quick_ablation_check.sh
```

This runs **2 experiments**:
1. No constraints (29D features)
2. With constraints (29D features, default)

**Runtime:** ~2× one train+tune cycle

**Output:** Clear comparison showing if constraints are the problem.

---

## Full Ablation Suite

If the quick test doesn't identify the issue, run the full suite:

```bash
cd symbiotic-ai
./test_ablations.sh
```

This runs **4 experiments**:
1. No constraints (29D)
2. With constraints (29D) 
3. No color descriptors (17D + constraints)
4. Single ARUCO channel (27D + constraints)

**Runtime:** ~4× one train+tune cycle

**Note:** Tests 3 and 4 require **manual code edits** (script will pause and prompt you).

### Manual Edits for Test 3 (No Color)

When prompted, edit `symbiote_weak/state_detection/feature_extraction.py`:

```python
# Around line 220, comment out color_hs:
features = np.concatenate([
    hand_center_norm,       # 2
    velocity,               # 2
    acceleration,           # 2
    bbox_features,          # 4
    orientation,            # 3
    [obj_conf],             # 1
    # color_hs,             # 12  ← COMMENT THIS OUT
    [aruco_signed, aruco_pick, aruco_place],  # 3
])
```

And edit `symbiote_weak/state_detection/config.py`:
```python
feature_dim: int = 17  # was 29
```

### Manual Edits for Test 4 (Single ARUCO)

When prompted, edit `symbiote_weak/state_detection/feature_extraction.py`:

```python
# Around line 220, use only aruco_signed:
features = np.concatenate([
    hand_center_norm,       # 2
    velocity,               # 2
    acceleration,           # 2
    bbox_features,          # 4
    orientation,            # 3
    [obj_conf],             # 1
    color_hs,               # 12
    [aruco_signed],         # 1  ← ONLY SIGNED, not pick/place
])
```

And edit `symbiote_weak/state_detection/config.py`:
```python
feature_dim: int = 27  # was 29
```

---

## Analyzing Results

After tests complete, run:

```bash
python3 analyze_ablations.py
```

This generates:
- **Summary table** comparing all tests
- **Ablation comparison CSV** with full metrics
- **Automated recommendations** based on metric deltas

### Reading the Output

The analyzer shows:

**Key Metrics:**
- `boundary_rmse`: Lower is better (target: <5s)
- `macro_f1`: Higher is better (target: >0.6)
- `recall_*`: Per-class recall (balanced is good)

**Red Flags:**
- One recall >> others (class imbalance)
- F1 drops >0.05 → that change hurt
- All F1 <0.3 → systemic issue (not just one feature)

**Decision Logic:**

| If this pattern appears | Then |
|------------------------|------|
| No-constraints F1 >> With-constraints F1 | **Constraints are the problem**. Check label CSV quality. |
| With-color F1 << No-color F1 | **Color descriptors hurt**. Remove them. |
| 3-ch ARUCO F1 << 1-ch ARUCO F1 | **Extra ARUCO channels hurt**. Use single signed. |
| All configs F1 <0.3 | **Systemic issue**. Check train data quantity/quality. |

---

## Detailed Manual Analysis

If you want deeper insight, check individual test results:

### 1. Read the tuning grid

```bash
# For test 1 (no constraints)
docker-compose run --rm symbiotic-ai cat /models/htk_weak_test1_no_constraints/models/hmm_final/tuning_grid.csv
```

Look for:
- Is there ANY `(p, s)` combo with good F1?
- Are all rows similarly bad, or is there variance?

### 2. Run inference on a sample video

```bash
docker-compose run --rm symbiotic-ai python -m symbiote_weak.hmm_infer \
    --video /data/hmm-testing/picklist_videos/picklist_1.MP4 \
    --model-dir /models/htk_weak_test1_no_constraints \
    --output-csv /outputs/test1_pred.csv \
    --output-video /outputs/test1_annotated.mp4 \
    --aruco-config /data/aruco_config/aruco_bins.json
```

Then **watch the annotated video** to see:
- Is it stuck in one state?
- Are transitions happening at wrong times?
- Does it match ground truth at all?

### 3. Compare against ground truth visually

```bash
# Check what the label CSV expects
cat hmm-testing/picklist_labels/picklist_1.csv

# Compare with predictions
cat outputs/test1_pred.csv
```

---

## What to Do After Ablation

Based on results:

**If constraints are the problem:**
- Verify label CSVs match video filenames
- Check if label sequences are actually correct
- Consider disabling constraints until labels are validated

**If color descriptors hurt:**
- Remove color (keep 17D)
- Or use feature masks to zero out color channels

**If 3-ch ARUCO hurts:**
- Revert to single signed ARUCO channel

**If everything is bad:**
- Check training data:
  - How many videos? (Need 5-10+ for 29D)
  - Are labels accurate?
  - Are train and dev from same distribution?
- Try legacy pipeline: `--pipeline legacy`
- Consider using feature masks to select top-K dimensions only

---

## Files Created

- `test_ablations.sh` — Full 4-test suite (requires manual edits for tests 3-4)
- `quick_ablation_check.sh` — Fast 2-test (constraints only, no edits needed)
- `analyze_ablations.py` — Results parser and comparison tool

## Recommended Workflow

1. **Start with quick test:**  
   `./quick_ablation_check.sh`

2. **If constraints are clearly the issue:**  
   Fix label CSVs or disable constraints, retrain, done.

3. **If constraints aren't the issue:**  
   Run full `./test_ablations.sh` to test color and ARUCO changes.

4. **Always analyze:**  
   `python3 analyze_ablations.py` after tests complete.
