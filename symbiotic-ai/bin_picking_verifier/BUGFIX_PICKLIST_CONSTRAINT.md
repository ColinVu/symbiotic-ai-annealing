# Bug Fixes: Picklist Constraint Issues

## Problem
Accuracy was **below random baseline** due to two critical bugs in the picklist constraint logic.

## Bug 1: Scoring irrelevant bins (FIXED)

**Root cause**: Scored all 120 bins from ArUco map (shelves C, D, E, F, G) instead of only the 24 shelf C bins that appear in picklists.

**Impact**: Model could predict `d23`, `e15`, etc. (non-existent in picklists) → guaranteed wrong.

**Fix in `voting_logic.py`**:
- Added `valid_skus` parameter to `score_pick_segment()` 
- In `predict_for_segments()`, build set of all SKUs from picklists: `valid_skus = union of all picklist items`
- Only score bins where `sku in valid_skus`
- Logs: `"Constraining scoring to N SKUs from picklists: [...]"`

**Before**:
```python
scores: Dict[str, float] = {b.sku: 0.0 for b in aruco_map.bins}  # All 120
```

**After**:
```python
if valid_skus is None:
    scores = {b.sku: 0.0 for b in aruco_map.bins}
else:
    scores = {b.sku: 0.0 for b in aruco_map.bins if b.sku in valid_skus}  # Only picklist items
```

## Bug 2: No fallback when predictions fail constraint (FIXED)

**Root cause**: When `ranked_skus` contained no valid picklist items (due to tracking failures or scoring irrelevant bins), returned `None` → 0% accuracy instead of random baseline.

**Impact**: Guaranteed wrong predictions instead of ~1/picklist_size random chance (16-50%).

**Fix in `PicklistState.choose()`**:

Added **two fallback tiers** after primary depletion logic fails:

1. **Fallback 2**: If no ranked candidate matches, pick **first remaining item** from active picklist
2. **Fallback 3**: If block exhausted, pick **first item from original picklist**

Both fallbacks log warnings for debugging.

**Before**:
```python
# After trying depletion and set membership
return block, None, True  # Returns None → guaranteed wrong
```

**After**:
```python
# Fallback 2: pick first remaining
if remain:
    chosen = remain[0]
    remain.remove(chosen)
    logger.warning("No ranked candidates; defaulting to first remaining: %s", chosen)
    return block, chosen, True

# Fallback 3: pick any from original
if self.picklists[block]:
    chosen = self.picklists[block][0]
    logger.warning("Block exhausted; defaulting to first item: %s", chosen)
    return block, chosen, True

return block, None, True  # Only if picklist is empty (should never happen)
```

## Expected improvement

**Before fixes**:
- Accuracy below random (e.g., 5-15% when random baseline = 25-50%)
- Many `predicted_sku: null` in output JSONs
- Predictions like `d23`, `e15` that don't exist in picklists

**After fixes**:
- Minimum accuracy = random baseline (~1/picklist_size)
- All predictions constrained to valid picklist items
- Tracking failures gracefully degrade to random within correct picklist
- Better than random when hand/ArUco tracking works

## How to test

```bash
cd symbiotic-ai/bin_picking_verifier
./run_pipeline.sh
```

Check logs for:
- `"Constraining scoring to N SKUs from picklists"` (should be ~8-12 SKUs per video, not 120)
- `"No ranked candidates; defaulting to..."` (indicates fallback triggered)

Check output JSONs:
- No more `"predicted_sku": null`
- No more non-shelf-C predictions (e.g., `d23`, `e15`)
- `"fallback_to_set_membership": true` when fallbacks triggered
