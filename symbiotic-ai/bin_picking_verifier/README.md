# Bin picking verifier

Egocentric 4K video inference: locate the 4×6 bin grid with ArUco (`DICT_5X5_1000`), detect plane-breaking intrusions using DepthAnythingV2, and predict which SKU was picked during each `PICK` state using depth-based pixel counting and picklist constraints.

**No hand tracking required** - the system uses monocular depth estimation to detect when any object (hand, arm, etc.) breaks the plane and reaches into bins, making it robust to occlusion.

## Quick start

### Full pipeline (ArUco map + inference + evaluation)

```bash
cd symbiotic-ai/bin_picking_verifier
pip install -r requirements.txt
./run_pipeline.sh --debug  # Add --debug for annotated frames
```

This runs:
1. `parse_aruco_excel.py` — generates `aruco_map.json` from `randomized_Object_List.xlsx`
2. `main.py` — runs inference on all videos (depth gates enabled by default)
3. `evaluate.py` — compares predictions to `ground_truth.csv`

Results in `./out/` with accuracy metrics in `./out/evaluation.json`.

**Note:** First run downloads DepthAnything V2 Small weights (~80MB) via HuggingFace.

## Install

```bash
cd symbiotic-ai/bin_picking_verifier
pip install -r requirements.txt
```

Requires `torch`, `transformers`, and `pillow` for depth estimation. MediaPipe is included for legacy mode (`--no-depth`) but not used by default.

## 3D plane intrusion detection (default scoring)

Each frame is sampled at stride 3 (every 3rd frame). **MediaPipe hand tracking is not used** — the system fits a true 3D plane to the bin grid and detects intrusions geometrically:

1. **Depth estimation**:
   - `DepthAnythingV2-Small` predicts a full-frame depth map.

2. **3D plane fitting**:
   - Unproject visible ArUco marker centers to 3D using approximate camera intrinsics.
   - Fit a plane to those 3D points (SVD). Normal is oriented toward the camera.

3. **Plane-crossing pixel detection** (inside bin polygons only):
   - Unproject sampled pixels inside each bin polygon to 3D.
   - Signed distance to plane: `normal·point + offset`.
   - **Intrusion** when signed distance `< -plane_cross_margin` (past the bin surface plane).

4. **Bin assignment**:
   - Score bins by intrusion pixel count inside their polygon.
   - Occlusion bonus if baseline-visible marker is now missing.
   - `PicklistState` constrains final SKU to the active sub-picklist block.

**Why this works**:
- True 2D plane in 3D space (not circular depth-from-camera comparison).
- Robust to camera angle and tilted bin grids.
- No hand detection needed — works when the hand is occluded.

Use `--no-depth` to revert to legacy 2D MediaPipe point-in-polygon scoring.

## Run inference

### With live per-video accuracy (recommended)

```bash
python main.py \
  --videos ../hmm-testing/picklist_videos \
  --state-csvs ../hmm-testing/picklist_labels \
  --picklist-jsons ../hmm-testing/picklist_jsons \
  --aruco-map aruco_map.json \
  --output-dir ./out \
  --ground-truth ../ground_truth.csv \
  --debug
```

### Debug visualization (live window)

```bash
python main.py ... --visualized
```

Shows:
- ArUco markers + SKU labels
- Bin polygons
- **Intrusion mask overlay** (red tint for pixels crossing the 3D bin plane)
- Fitted plane normal and offset
- Intrusion pixel count
- Gate status (`PLANE BROKEN` / `NO INTRUSION`)
- Optional depth map inset

**Space** pause, **Esc**/**q** quit current video.

**Note**: Hand landmarks and cross product vectors are no longer shown - the system uses depth-based intrusion detection only.

### CLI flags (depth)

| Flag | Description |
|------|-------------|
| `--depth-model ID` | HuggingFace model id (default: Depth-Anything-V2-Small-hf) |
| `--depth-device auto\|cpu\|cuda\|mps` | Torch device |
| `--no-depth` | Legacy 2D scoring only |

**Note**: Frames are sampled at stride 3 (every 3rd frame) to reduce computational cost while maintaining temporal coverage.

## Temporal windowing

The **entire PICK segment** is processed (sampled every 3rd frame). Baseline frames record visible ArUco IDs for occlusion bonus; all sampled frames run depth + 3D plane intrusion scoring.

## Config tunables ([config.py](config.py))

| Field | Default | Meaning |
|-------|---------|---------|
| `plane_cross_margin` | `0.015` | Signed 3D distance past fitted plane to count as intrusion |
| `min_markers_for_depth_ref` | `3` | Min ArUcos to fit bin surface plane |
| `frame_stride` | `3` | Process every Nth frame |
| `intrusion_pixel_stride` | `2` | Subsample pixels inside bin polygons |
| `viz_show_depth_inset` | `true` | Depth colormap inset in `--visualized` |

## ArUco map

See [examples/aruco_map.example.json](examples/aruco_map.example.json) or generate from Excel:

```bash
python parse_aruco_excel.py --excel ../randomized_Object_List.xlsx --output aruco_map.json
```

## Evaluate against ground truth

```bash
python evaluate.py \
  --predictions ./out \
  --ground-truth ../ground_truth.csv \
  --output eval_results.json
```

## Outputs

- `<output-dir>/<video_stem>.json` — per-pick predictions and scores
- With `--debug`: `<output-dir>/debug/<video_stem>/pick_XX.png`
- With `--visualized`: live window only (no extra files)

## Pipeline architecture

```
parse_aruco_excel.py → aruco_map.json
                           ↓
main.py: DepthAnything V2 → 3D plane fit → plane-crossing pixels → picklist
                           ↓
evaluate.py: predictions + ground_truth.csv → accuracy metrics
```
