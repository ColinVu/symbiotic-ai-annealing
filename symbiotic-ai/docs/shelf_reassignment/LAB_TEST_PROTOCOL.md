# Lab protocol: measure shelf-swap accuracy on new warehouse video

Use this protocol after the shelf decision engine recommends a layout. The engine's reported gain is a replay estimate on existing embeddings. This trial measures whether recognition accuracy changes when people pick from the **physical** layouts and the camera records **new** video.

The primary result is weighted shelf-constrained **carry-segment Top-1 accuracy**: correctly predicted SKU / all labeled carry segments. Use the same saved model, camera setup, preprocessing, segmentation rules, and scoring method in both layouts. Keep true labels out of the recommendation input.

## Before scheduling a trial

Assign one person to own each of these: inventory and physical moves; video capture and state labels; independent true-SKU labels; scoring and results review. One person may fill multiple roles, but the true-SKU record must come from the actual pick, not the model prediction.

Collect the following inputs in a trial folder outside the checked-in result tree:

| Input | Check before starting |
| --- | --- |
| Saved model directory | Contains `model_metadata.json`, `centroids.npy`, and any saved transform files; record its path and hash. |
| Inventory map | Every SKU has exactly one current shelf and slot; counts match shelf capacities. |
| Decision snapshot | Follows [INTEGRATION.md](INTEGRATION.md): current inventory, capacities, and only observations revealed so far. Do not put true labels in it. |
| Camera and recording setup | Same mount, resolution, frame rate, lighting plan, and capture settings can be used for both layouts. |
| Picklists | Include proposed moved SKUs, items already on their destination shelves, and unchanged comparison SKUs. Record the intended and actually picked SKU for every pick. |
| Carry-state labels | A state-label CSV can identify each carry interval for each video. Confirm the frame-indexing convention. |

The checked-in `cv_results_subset` has model metadata but no `centroids.npy`, and `hmm-testing/picklist_videos/` has no checked-in videos. Obtain complete artifacts and new recordings before reporting a lab accuracy result.

## 1. Generate and review a proposal

From `symbiotic-ai/`, with Python dependencies installed, run the following after replacing the paths:

```bash
python -m shelf_layout preflight \
  --snapshot /path/to/trial/snapshot.json \
  --model-dir /path/to/saved/model \
  --output-dir /path/to/trial/preflight

python -m shelf_layout recommend \
  --snapshot /path/to/trial/snapshot.json \
  --model-dir /path/to/saved/model \
  --output-dir /path/to/trial/proposal
```

If `preflight` exits with code 2, read `preflight/missing_inputs.json`, fix the input, and rerun it. Do not proceed on an incomplete model or inventory. Save the proposal's `summary.json`, `summary.md`, `move_transactions.csv`, and `candidate_layouts.csv` with the trial record. Check that the transaction rows identify the intended SKUs and valid source/destination slots. If `move_transactions.csv` is empty, record a **no-move recommendation**; gather more observations before testing a recommendation.

Record the proposal's predicted baseline and changed proxy error counts, fixed errors, new errors, and moved-SKU count. These are predictions to compare with the later physical result.

## 2. Prepare the test order and labels

Create a run sheet before recording. Each row should have:

```text
session_id,layout,worker_id,pick_number,intended_sku,actual_sku,pickup_shelf,pickup_slot,video_file,carry_segment_id,notes
```

Give every planned SKU comparable coverage in each layout. Include the destination shelf occupants because a swap can fix one confusion while creating another. Vary pick order between sessions. If multiple workers are available, have some start with the original layout and others with the swapped layout, restoring the original layout between sessions as needed. Record the order used. Do not select only the easiest or most confusing picks after seeing results.

Choose a primary comparison and a minimum number of labeled picks **before** scoring the main trial. A small 20–30-pick pilot is useful for checking capture and labels; it is not by itself evidence of an accuracy improvement. Use the pilot error rate to plan a larger main trial if needed.

## 3. Pilot the recording and scoring path

With the original layout still installed, record the pilot. For each pick, verify that:

1. The video file opens and the object is visible during carry.
2. The state-label CSV produces the expected number and order of carry intervals.
3. One independent `actual_sku` aligns with each carry interval. Record wrong physical picks as such; do not silently substitute the intended SKU.
4. The pickup shelf and slot match the physical inventory map.
5. The embedding pipeline yields the expected dimension for the saved model. Keep segments with no usable frames in the accuracy denominator as failures.

Fix capture, interval, alignment, or model-loading problems before the main trial. Save the raw pilot files and note any protocol changes.

## 4. Record the original-layout baseline

Record the planned baseline picklists with the original inventory. Save raw videos, state-label CSVs, and the completed run sheet. Record camera settings, worker, date/time, lighting changes, model path/hash, preprocessing settings, and inventory version. Freeze the model and these settings for the swapped-layout session; do not retrain or tune from the baseline results.

## 5. Move products and verify the new layout

Apply only the approved rows in `proposal/move_transactions.csv`. Check each SKU's physical destination against `proposal/summary.json` → `recommendation.layout`. Have a second person verify every moved SKU and confirm shelf capacities and unique slots. Save a post-move inventory map and the move time.

Update the pickup-shelf/marker mapping used by capture or inference. A SKU ID such as `f14` stays `f14` after it moves; its prefix is historical and must not be used to infer its new shelf. Do a few dry-run picks to confirm the observed pickup shelf agrees with the new inventory map.

## 6. Record the swapped-layout session

Record the matched picklists with the same frozen model and capture settings. Complete the same run-sheet columns. Capture new video after the move; replaying old embeddings is not a physical test. Note any unavoidable changes in lighting, worker behavior, marker visibility, or segment detection.

If practical, restore the original layout for another session and alternate layout order across workers or days. This gives a check against practice and session effects. Verify the inventory after each restoration.

## 7. Score the new videos with the correct shelf map

Produce one scored row per **labeled carry segment**:

```text
session_id,layout,segment_id,true_sku,pickup_shelf,predicted_sku,correct,usable_frame_count,model_hash,inventory_version
```

The scorer must take the pickup shelf from that pick and restrict candidates using the explicit SKU-to-shelf map for that session. It must apply the weighted top-three margin vote in [EVALUATION_SPEC.md](EVALUATION_SPEC.md) with the saved model transforms. Before scoring the main trial, verify that it agrees per segment with the canonical evaluator on an **unchanged-layout** reference set.

**Implementation handoff:** this physical-trial scorer is not currently a turnkey command. `symbiote_weak_generalized/experiments/evaluator.py` derives shelf candidates from the video filename and SKU prefixes, which becomes wrong after a swap. Adapt it to accept per-pick pickup shelves and the explicit inventory map. `scripts/evaluate_picklist_video.py` reports a different, unweighted voting metric and must not be used as the primary shelf-swap score. Keep the scorer version or commit hash with the trial results.

For each session, check that the number of scored rows equals the number of independently labeled carry segments. Investigate any mismatch; do not truncate to the shorter list. Count segments with no usable embeddings as incorrect. Do not feed the true-SKU file into the recommender. The CLI's optional `--truth-json` is retrospective scoring of simulation observations, not the new-video lab result.

## 8. Summarize and review

For each layout report `correct / all labeled segments`, the percentage, and counts of empty-frame or failed segments. Also report the same numbers separately for moved SKUs, destination-shelf occupants, and unchanged SKUs. Show a confusion table by true SKU and predicted SKU. Compare old and new error counts for matched SKU groups; distinguish errors that disappeared from new errors that appeared. Report the number of moved SKUs and the actual time and effort of the move.

Compare the **measured** change on new video with the proposal's **predicted** proxy change. If the result differs, inspect raw video, carry intervals, shelf-marker events, and per-SKU errors before changing the model. Do not compare this trial's percentage directly with the paper's 96.9% or the checked-in five-fold subset; they use different data and protocols.

Treat the swap as promising when the prespecified main comparison improves on new recordings and destination-shelf occupants do not show a material regression. If the result is uncertain, report the counts and uncertainty plainly and collect more matched picks. Preserve the videos, run sheet, inventories, model hash, proposal, scoring code version, and segment-level results so teammates can reproduce the conclusion.
