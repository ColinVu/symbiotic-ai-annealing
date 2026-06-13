# Data Requirements for Symbiote Pipeline

> **🐳 Docker Setup**: For containerized deployment (recommended for HTK on Windows), see `DOCKER_SETUP.md`. This eliminates HTK installation issues and provides a reproducible environment.

## Overview

This document lists ALL data files, configuration files, and setup requirements needed before running the symbiote pipeline. Use this as a checklist to ensure your system is properly configured.

**Last Updated**: February 23, 2026 (rev 2)

---

## Required Before ANY Pipeline Use

### 1. ARUCO Marker Configuration (`config/aruco_bins.json`)

**Location**: `symbiotic-ai/config/aruco_bins.json`

**Purpose**: Maps ARUCO marker IDs to pick/place bins and objects

**Required Fields**:
```json
{
  "marker_dict": "DICT_5X5_1000",
  "bins": {
    "MARKER_ID": {
      "type": "pick" or "place",
      "object": "object_name",
      "name": "bin_name",
      "description": "human readable description"
    }
  },
  "distance_decay": 5.0
}
```

**Current Configuration** (`config/aruco_bins.json` — already created):

- **Dictionary**: `DICT_5X5_1000` (5×5 bit markers, supports IDs 0–999)
- **Pick bins**: 120 bins, IDs 1–120, one per object from the randomized object list
- **Place bins**: 3 bins, IDs 990–992

| ID  | Object / Name         | Type  |
|-----|-----------------------|-------|
| 1   | Block_A               | pick  |
| 2   | Block_B               | pick  |
| 3   | Block_C               | pick  |
| 4   | Block_D               | pick  |
| 5   | Block_E               | pick  |
| 6   | Block_F               | pick  |
| 7   | Block_G               | pick  |
| 8   | Block_H               | pick  |
| 9   | Block_I               | pick  |
| 10  | Block_J               | pick  |
| 11  | Block_K               | pick  |
| 12  | Block_L               | pick  |
| 13  | Block_M               | pick  |
| 14  | Block_N               | pick  |
| 15  | Block_O               | pick  |
| 16  | Block_P               | pick  |
| 17  | Block_Q               | pick  |
| 18  | Block_R               | pick  |
| 19  | Block_S               | pick  |
| 20  | Block_T               | pick  |
| 21  | Block_U               | pick  |
| 22  | Block_V               | pick  |
| 23  | Block_W               | pick  |
| 24  | Block_X               | pick  |
| 25  | Block_Y               | pick  |
| 26  | Block_Z               | pick  |
| 27  | Spoon_Red             | pick  |
| 28  | Spoon_Orange          | pick  |
| 29  | Spoon_Yellow          | pick  |
| 30  | Spoon_Green           | pick  |
| 31  | Spoon_Blue            | pick  |
| 32  | Spoon_Purple          | pick  |
| 33  | Fork_Red              | pick  |
| 34  | Fork_Orange           | pick  |
| 35  | Fork_Yellow           | pick  |
| 36  | Fork_Green            | pick  |
| 37  | Fork_Blue             | pick  |
| 38  | Fork_Purple           | pick  |
| 39  | Knife_Red             | pick  |
| 40  | Knife_Orange          | pick  |
| 41  | Knife_Yellow          | pick  |
| 42  | Knife_Green           | pick  |
| 43  | Knife_Blue            | pick  |
| 44  | Knife_Purple          | pick  |
| 45  | Bead_Red              | pick  |
| 46  | Bead_Yellow           | pick  |
| 47  | Bead_Green            | pick  |
| 48  | Bead_Blue             | pick  |
| 49  | Bead_Wood             | pick  |
| 50  | Orange_Clip_Leaf      | pick  |
| 51  | Orange_Clip_Owl_Flying | pick |
| 52  | Orange_Clip_Owl_Resting | pick |
| 53  | Yellow_Clip_Leaf      | pick  |
| 54  | Yellow_Clip_Owl_Flying | pick |
| 55  | Yellow_Clip_Owl_Resting | pick |
| 56  | Green_Clip_Leaf       | pick  |
| 57  | Green_Clip_Owl_Flying | pick  |
| 58  | Green_Clip_Owl_Resting | pick |
| 59  | Tile_Clear            | pick  |
| 60  | Tile_Red              | pick  |
| 61  | Tile_Wood             | pick  |
| 62  | Chipclip_Red          | pick  |
| 63  | Chipclip_Blue         | pick  |
| 64  | Chipclip_Green        | pick  |
| 65  | Chipclip_Black        | pick  |
| 66  | Chipclip_White        | pick  |
| 67  | Candle                | pick  |
| 68  | Alligator_Clip        | pick  |
| 69  | Casing_Black          | pick  |
| 70  | Casing_Blue           | pick  |
| 71  | Paperclip_Red         | pick  |
| 72  | Paperclip_Green       | pick  |
| 73  | Clothespin_Red        | pick  |
| 74  | Clothespin_Orange     | pick  |
| 75  | Clothespin_Yellow     | pick  |
| 76  | Clothespin_Green      | pick  |
| 77  | Clothespin_Blue       | pick  |
| 78  | Clothespin_Teal       | pick  |
| 79  | Clothespin_Purple     | pick  |
| 80  | Clothespin_Pink       | pick  |
| 81  | Clothespin_Magenta    | pick  |
| 82  | Clothespin_Brown      | pick  |
| 83  | Red_Gem_Teardrop      | pick  |
| 84  | Red_Gem_Oval          | pick  |
| 85  | Red_Gem_Square        | pick  |
| 86  | Orange_Gem_Convex     | pick  |
| 87  | Orange_Gem_Oval       | pick  |
| 88  | Orange_Gem_Round      | pick  |
| 89  | Yellow_Gem_Oval       | pick  |
| 90  | Yellow_Gem_Round      | pick  |
| 91  | Yellow_Gem_Teardrop   | pick  |
| 92  | Yellow_Gem_Square     | pick  |
| 93  | Green_Gem_Convex      | pick  |
| 94  | Green_Gem_Oval        | pick  |
| 95  | Green_Gem_Teardrop    | pick  |
| 96  | Green_Gem_Round       | pick  |
| 97  | Green_Gem_Square      | pick  |
| 98  | Lightblue_Gem_Square  | pick  |
| 99  | Lightblue_Gem_Convex  | pick  |
| 100 | Lightblue_Gem_Teardrop | pick |
| 101 | Lightblue_Gem_Round   | pick  |
| 102 | Lightblue_Gem_Oval    | pick  |
| 103 | Blue_Gem_Convex       | pick  |
| 104 | Blue_Gem_Square       | pick  |
| 105 | Purple_Gem_Convex     | pick  |
| 106 | Purple_Gem_Teardrop   | pick  |
| 107 | Purple_Gem_Round      | pick  |
| 108 | Purple_Gem_Square     | pick  |
| 109 | Magenta_Gem_Convex    | pick  |
| 110 | Magenta_Gem_Square    | pick  |
| 111 | Magenta_Gem_Oval      | pick  |
| 112 | Pink_Gem_Convex       | pick  |
| 113 | Pink_Gem_Oval         | pick  |
| 114 | Pink_Gem_Round        | pick  |
| 115 | Pink_Gem_Square       | pick  |
| 116 | Pink_Gem_Teardrop     | pick  |
| 117 | Clear_Gem_Oval        | pick  |
| 118 | Clear_Gem_Round       | pick  |
| 119 | Clear_Gem_Square      | pick  |
| 120 | Clear_Gem_Teardrop    | pick  |
| 990 | N/A (place_1)         | place |
| 991 | N/A (place_2)         | place |
| 992 | N/A (place_3)         | place |

**How to Create / Modify**:
1. Dictionary is fixed as `DICT_5X5_1000` (5×5 bit markers, IDs 0–999)
2. Pick bin IDs 1–120 map directly to the 120 objects from `randomized_Object_List.xlsx`
3. Place bin IDs 990, 991, 992 are generic drop zones (`place_1`, `place_2`, `place_3`)
4. The file already exists at `config/aruco_bins.json` — edit it there if bin assignments change
5. Print markers from the `DICT_5X5_1000` dictionary and affix to physical bins

**Validation**: Run ARUCO test tool (see section below)

---

## Required for Basic Training (Image-Based)

### 2. Image Dataset

**Location**: User-defined (e.g., `symbiotic-ai/images/training/`)

**Structure**:
```
images/training/
├── apple/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── banana/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── orange/
    ├── image001.jpg
    └── ...
```

**Requirements**:
- One folder per object class
- Folder name = object label
- Images must show hand holding object
- Minimum: 20 images per class
- Recommended: 50-100 images per class
- Supported formats: JPG, PNG, HEIC

**Already in Pipeline**: 
- Hand segmentation (automatic)
- Blur filtering (automatic)
- CLIP embedding (automatic)

---

## Required for Video Training (Basic, No HMM)

### 3. Training Videos

**Location**: User-defined (e.g., `symbiotic-ai/videos/training/`)

**Requirements**:
- Clear view of hand and objects
- ARUCO markers visible in frame when near bins
- Hand must be segmentable by MediaPipe
- Minimum resolution: 720p
- Recommended: 1080p, 30 FPS
- Format: MP4, AVI, MOV

**What You Provide**:
- Videos showing pick/place operations
- No annotation needed for basic training
- ARUCO config must match physical setup

**CLI Usage**:
```bash
python -m symbiote.cli.main train \
    --video videos/training/pick_apple_001.mp4 \
    --label "apple" \
    --aruco-config config/aruco_bins.json
```

---

## Required for HTK HMM State Detection Training

### 4. Annotated Training Videos

**Location**: Same as training videos

**Structure**:
```
videos/training/
├── pick_place_001.mp4
├── pick_place_001_annotations.csv    # ← REQUIRED
├── pick_place_002.mp4
├── pick_place_002_annotations.csv    # ← REQUIRED
└── ...
```

**Annotation CSV Format**:

**File**: `video_name_annotations.csv`

**Columns**: `timestamp_start`, `timestamp_end`, `state`

**Example**:
```csv
timestamp_start,timestamp_end,state
0.0,1.5,CARRY_EMPTY
1.5,3.2,PICK
3.2,8.7,CARRY_WITH
8.7,10.5,PLACE
10.5,15.0,CARRY_EMPTY
15.0,16.8,PICK
16.8,23.1,CARRY_WITH
23.1,25.0,PLACE
25.0,30.0,CARRY_EMPTY
```

**Rules**:
- States MUST follow cycle: PICK → CARRY_WITH → PLACE → CARRY_EMPTY
- No state skipping allowed
- Timestamps in seconds (float)
- No gaps between consecutive states
- States must cover entire video duration

**How to Create Annotations**:

**Option 1: Manual annotation**
1. Watch video frame-by-frame
2. Note timestamps where state changes occur
3. Create CSV with transitions
4. Validate state cycle is correct

**Option 2: Video annotation tools**
- Use tools like CVAT, Label Studio, or VGG Video Annotator
- Export to CSV format matching above schema

**Minimum Data Volume**:
- 5-10 annotated videos
- 3-5 complete pick-place cycles per video
- Total: 15-50 state cycles

**Recommended Data Volume**:
- 10-20 annotated videos
- 5-10 cycles per video
- Total: 50-200 state cycles
- Diverse conditions (different objects, lighting, speeds)

---

## Required for HTK HMM Training

### 5. HTK Toolkit Installation

**What**: Hidden Markov Model Toolkit from Cambridge University

**Download**: http://htk.eng.cam.ac.uk/

**License**: Free for research use (HTK License required)

**Installation**:
1. Register and download HTK source code
2. Compile for your platform (Linux/Mac/Windows)
3. Add HTK binaries to PATH
4. Required executables: `HCompV`, `HERest`, `HVite`, `HHEd`

**Verify Installation**:
```bash
HCompV -V  # Should print HTK version
HERest -V  # Should print HTK version
```

**Alternative**: Pre-compiled binaries (if available for your platform)

---

## Physical Setup Requirements

### 6. ARUCO Marker Printing

**What**: Physical ARUCO markers for bin identification

**Steps**:
1. Generate markers from `DICT_5X5_1000` dictionary (5×5 bit, IDs 0–999)
   - Use online generator: https://chev.me/arucogen/ (select "5x5" dictionary)
   - Or use OpenCV Python script below
2. Print markers (recommended size: 6–8 inches square)
3. Laminate or mount on rigid backing
4. Affix to bins in clear view
5. Ensure good lighting (avoid glare)

**Example Python Generator**:
```python
import cv2

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)

# Generate pick bin markers (IDs 1-120) and place bin markers (990-992)
marker_ids = list(range(1, 121)) + [990, 991, 992]
for marker_id in marker_ids:
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 400)
    cv2.imwrite(f"aruco_marker_{marker_id}.png", marker_image)
```

### 7. Physical Bin Setup

**Requirements**:
- Separate bins for "pick" and "place" operations
- Each bin labeled with unique ARUCO marker (IDs 1–120 for pick, 990–992 for place)
- Bins arranged so camera can see markers
- Consistent bin positions during training
- One object type per bin

**Bin Layout Summary**:
```
[PICK BINS — IDs 1–120]          [WORK AREA]       [PLACE BINS — IDs 990–992]
  ARUCO 1  (Block_A)                                   ARUCO 990 (place_1)
  ARUCO 2  (Block_B)                [CAMERA]           ARUCO 991 (place_2)
  ARUCO 3  (Block_C)                                   ARUCO 992 (place_3)
  ... (120 pick bins total)
```

---

## Optional Configuration Files

### 8. Custom CLIP Model (Optional)

**Default**: Uses `openai/clip-vit-base-patch32` from Hugging Face

**Custom Model Location**: User-defined

**When to Use Custom**:
- Training on specialized objects not in CLIP's training data
- Need higher accuracy for specific domain
- Already have fine-tuned CLIP model

**How to Specify**:
```python
# In symbiote/core/config.py
MODEL = "path/to/your/custom/clip/model"
```

### 9. Training Configuration (Optional)

**Default**: Uses `DEFAULT_CONFIG` in `symbiote/core/config.py`

**Customizable Parameters**:
```python
{
    "learning_rate": 0.001,
    "batch_size": 32,
    "max_epochs": 100,
    "early_stopping_patience": 10,
    "hidden_dim": 256,
    "dropout": 0.3,
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_seed": 42
}
```

**Override via CLI**:
```bash
python -m symbiote.cli.main train \
    --video video.mp4 \
    --label "apple" \
    --lr 0.0005 \
    --epochs 50 \
    --hidden-dim 512
```

---

## Directory Structure Checklist

Before running the pipeline, ensure this structure exists:

```
symbiotic-ai/
├── config/
│   └── aruco_bins.json                 # ✓ REQUIRED (already created — 120 pick bins + 3 place bins)
│
├── videos/
│   ├── training/
│   │   ├── video_001.mp4
│   │   ├── video_001_annotations.csv   # For HMM training only
│   │   └── ...
│   └── testing/
│       └── ...
│
├── images/
│   └── training/
│       ├── apple/
│       ├── banana/
│       └── ...
│
├── models/
│   └── classifier/
│       ├── .cache/                     # Auto-created
│       └── htk_models/                 # Auto-created for HMM
│
├── hmm-testing/
│   ├── picklist_videos/            # Training videos (mp4/mov/…)
│   └── picklist_labels/            # Matching annotation CSVs (or .eaf files)
│
└── symbiote/
    ├── eaf_to_csv.py               # ← EAF → CSV annotation converter
    ├── hmm_train.py                # ← batch HMM training script (auto-converts EAF)
    ├── hmm_infer.py                # ← HMM inference + annotation script
    └── ...  # Core pipeline code
```

---

## Validation Checklist

Before training, verify:

### Configuration
- [ ] `config/aruco_bins.json` exists
- [ ] All physical ARUCO IDs are in config
- [ ] Marker types (pick/place) are correct
- [ ] Object names match training data labels

### Physical Setup
- [ ] ARUCO markers printed and affixed to bins
- [ ] Markers are clearly visible
- [ ] Good lighting (no glare on markers)
- [ ] Camera can see markers and hand simultaneously

### Data
- [ ] Training videos/images exist
- [ ] Videos show clear hand movements
- [ ] For HMM: Annotation CSVs exist and match video names
- [ ] For HMM: Annotations follow state cycle rules

### Software
- [ ] Python environment has all dependencies
- [ ] For HMM: HTK toolkit installed and in PATH
- [ ] For HMM: Can run `HCompV -V` successfully

### Test Tools
- [ ] Can run ARUCO test tool:
```bash
python -m symbiote.state_detection.test_aruco_detection \
    --video test_video.mp4 \
    --output test_annotated.mp4 \
    --aruco-config config/aruco_bins.json
```
- [ ] Annotated video shows correct marker detection
- [ ] Weighted scores look reasonable

---

## Quick Start Workflows

### Workflow 1: Image-Based Training (Simplest)
```bash
# 1. Setup (one-time)
# - Create config/aruco_bins.json
# - Print and affix ARUCO markers
# - Collect training images in folders by class

# 2. Train
python -m symbiote.cli.main train \
    --image-dir images/training \
    --aruco-config config/aruco_bins.json
```

**Required Data**:
- ✓ ARUCO config
- ✓ Training images
- ✗ Training videos
- ✗ Annotations
- ✗ HTK toolkit

### Workflow 2: Video-Based Training (No HMM)
```bash
# 1. Setup (one-time)
# - Same as Workflow 1
# - Record training videos

# 2. Train
python -m symbiote.cli.main train \
    --video videos/training/pick_apple.mp4 \
    --label "apple" \
    --aruco-config config/aruco_bins.json
```

**Required Data**:
- ✓ ARUCO config
- ✓ Training videos
- ✗ Training images (optional)
- ✗ Annotations
- ✗ HTK toolkit

### Workflow 3: Full HMM State Detection Training
```bash
# 1. Setup (one-time)
# - Same as Workflow 2
# - Annotate videos with state timestamps
# - Install HTK toolkit

# 2. Test ARUCO detection
python -m symbiote.state_detection.test_aruco_detection \
    --video videos/training/pick_place_001.mp4 \
    --output test_annotated.mp4 \
    --aruco-config config/aruco_bins.json

# 3. Train HMM (two-stage pipeline default)
python -m symbiote.hmm_train \
    --video-dir hmm-testing/picklist_videos \
    --label-dir hmm-testing/picklist_labels \
    --output-dir models/htk \
    --aruco-config config/aruco_bins.json \
    --pipeline two-stage
```

**Required Data**:
- ✓ ARUCO config
- ✓ Training videos
- ✓ Annotations (CSV files)
- ✓ HTK toolkit installed
- ✗ Training images (optional)

---

## HMM Testing Scripts

Two dedicated scripts live in `symbiote/` for the HMM training/inference workflow.

### `symbiote/eaf_to_csv.py` — EAF → CSV Annotation Converter

**Purpose**: Converts ELAN Linguistic Annotator (`.eaf`) annotation files to
the HMM training CSV format. Also runs automatically inside `hmm_train.py`
when `.eaf` files are detected in the label directory.

**EAF label → HMM state mapping**:

| EAF annotation value | HMM state    |
|----------------------|--------------|
| `carry_empty`        | `CARRY_EMPTY` |
| `pick_<anything>`    | `PICK`        |
| `carry_<anything>`   | `CARRY_WITH`  |
| `place_<anything>`   | `PLACE`       |

Times in the EAF are stored as integer milliseconds and are converted to
floating-point seconds in the output CSV.

**Run (from `symbiotic-ai/`)**:
```bash
# Convert a single file (CSV saved alongside it)
python -m symbiote.eaf_to_csv path/to/annotation.eaf

# Convert to an explicit output path
python -m symbiote.eaf_to_csv path/to/annotation.eaf --output path/to/out.csv

# Batch convert all .eaf files in a directory → picklist_labels/
python -m symbiote.eaf_to_csv --input-dir hmm-testing/eaf_source \
                               --output-dir hmm-testing/picklist_labels

# Auto-convert any .eaf already inside picklist_labels/ (default behavior)
python -m symbiote.eaf_to_csv
```

> **Note**: `hmm_train.py` calls this converter automatically before pair
> discovery, so you can drop `.eaf` files directly into `picklist_labels/`
> alongside (or instead of) `.csv` files and training will still work.

---

### `symbiote/hmm_train.py` — Batch HMM Training

**Purpose**: Automatically discovers all matched video/label pairs in the
`hmm-testing/` directories and trains the HTK HMM detector.

**Pipeline modes**:
- `two-stage` (default): trains coarse + subtype HTK models:
  - coarse: `INTERACT/CARRY`
  - interact subtype: `PICK/PLACE`
  - carry subtype: `CARRY_WITH/CARRY_EMPTY`
- `legacy`: keeps single-stage 4-state decoding behavior.

**Input layout**:
```
hmm-testing/
├── picklist_videos/
│   ├── session_01.mp4       # any video extension
│   ├── session_02.mov
│   └── ...
└── picklist_labels/
    ├── session_01.csv       # MUST match video filename stem
    ├── session_02.csv       # .eaf files also accepted (auto-converted)
    └── ...
```

Each label CSV must have columns `timestamp_start`, `timestamp_end`, `state`
and follow the state cycle `PICK → CARRY_WITH → PLACE → CARRY_EMPTY`.

**Run (from `symbiotic-ai/`)**:
```bash
# Default — uses hmm-testing/ dirs and outputs to models/htk/
python -m symbiote.hmm_train

# Custom paths
python -m symbiote.hmm_train \
    --video-dir  hmm-testing/picklist_videos \
    --label-dir  hmm-testing/picklist_labels \
    --output-dir models/htk \
    --aruco-config config/aruco_bins.json \
    --pipeline two-stage \
    --frame-skip 4

# Legacy single-stage behavior
python -m symbiote.hmm_train --legacy

# Two-stage with task-specific feature masks (from feature reports)
python -m symbiote.hmm_train \
    --pipeline two-stage \
    --coarse-feature-top-k 8 \
    --interact-feature-top-k 6 \
    --carry-feature-top-k 6
```

**Output**: Trained HTK HMM at `models/htk/models/hmm_final/`

---

### `symbiote/hmm_infer.py` — HMM Inference on New Video

**Purpose**: Runs the trained HMM on a new input video and produces:
1. A CSV of predicted state timestamps
2. An annotated copy of the video with a coloured state banner overlay

**State colour coding**:

| State        | Banner colour |
|--------------|---------------|
| PICK         | Green         |
| CARRY_WITH   | Yellow        |
| PLACE        | Red           |
| CARRY_EMPTY  | Grey          |

**Run (from `symbiotic-ai/`)**:
```bash
# Full output — CSV + annotated video (auto-named next to input)
python -m symbiote.hmm_infer --video path/to/test_video.mp4

# Explicit output paths
python -m symbiote.hmm_infer \
    --video path/to/test_video.mp4 \
    --model-dir models/htk \
    --output-csv results/predicted_states.csv \
    --output-video results/test_video_annotated.mp4 \
    --aruco-config config/aruco_bins.json \
    --pipeline two-stage

# CSV only (skip annotated video)
python -m symbiote.hmm_infer --video path/to/test_video.mp4 --no-video

# Legacy single-stage inference
python -m symbiote.hmm_infer --video path/to/test_video.mp4 --legacy
```

---

### `symbiote/hmm_gt_overlay.py` — Ground-Truth Overlay QA

**Purpose**: Renders labels from `picklist_labels/*.csv` directly on matching
videos to visually audit label timing quality.

**Run (from `symbiotic-ai/`)**:
```bash
python -m symbiote.hmm_gt_overlay \
    --video-dir hmm-testing/picklist_videos \
    --label-dir hmm-testing/picklist_labels \
    --output-dir outputs/gt_overlay
```

**Outputs**:
- Annotated videos: `outputs/gt_overlay/*_gt_overlay.mp4`
- Index CSV: `outputs/gt_overlay/gt_overlay_index.csv`

**Output CSV format**:
```csv
timestamp_start,timestamp_end,state
0.00,1.73,CARRY_EMPTY
1.73,3.12,PICK
3.12,8.40,CARRY_WITH
8.40,10.05,PLACE
...
```

**Default output files** (when paths not specified): placed in the same
directory as the input video, named `<stem>_states.csv` and
`<stem>_annotated.mp4`.

---

## Common Issues and Solutions

### Issue: "Cannot find aruco_bins.json"
**Solution**: Create `config/aruco_bins.json` with proper structure (see section 1)

### Issue: "No ARUCO markers detected"
**Solution**: 
- Check lighting (avoid glare)
- Verify markers are printed from correct dictionary (DICT_5X5_1000, 5×5 bit markers)
- Ensure markers are in camera view
- Run ARUCO test tool to validate

### Issue: "Hand detection failed"
**Solution**:
- Ensure hand is clearly visible
- Check lighting conditions
- Verify MediaPipe can detect hand (21 landmarks)

### Issue: "State cycle validation failed"
**Solution**:
- Check annotation CSV follows: PICK → CARRY_WITH → PLACE → CARRY_EMPTY
- Ensure no state skipping
- Verify timestamps are monotonically increasing

### Issue: "HTK command not found"
**Solution**:
- Verify HTK installation: `which HCompV`
- Add HTK bin directory to PATH
- Recompile HTK if needed

---

## Data Storage Estimates

### Disk Space Requirements

**Per Training Video** (30 seconds, 1080p):
- Original video: ~50-100 MB
- CLIP embedding cache: ~2-4 MB (for extracted frames)
- HTK feature cache: ~15 KB (very compact)
- Annotation CSV: <1 KB

**Per Training Image** (1920x1080):
- Original image: ~2-5 MB
- CLIP embedding cache: ~2-4 KB

**Trained Models**:
- CLIP classifier: ~10-50 MB
- HTK HMM models: ~500 KB - 1 MB (very compact)

**Total for Complete Setup**:
- 20 training videos: ~2-3 GB
- 500 training images: ~2-3 GB
- All caches and models: ~500 MB
- **Total: ~5-7 GB**

---

## Summary: What You MUST Have

### Minimum to Run Anything:
1. ✓ `config/aruco_bins.json`
2. ✓ Physical ARUCO markers (printed and affixed)

### To Train Classifier (Image):
3. ✓ Training images organized by class

### To Train Classifier (Video):
4. ✓ Training videos showing pick/place operations

### To Train HMM State Detection:
5. ✓ Annotated training videos (CSV files)
6. ✓ HTK toolkit installed

### To Test/Validate:
7. ✓ Test videos
8. ✓ ARUCO test tool working

---

**Next Steps**: 
1. Use this document as a checklist
2. Create required data files
3. Validate with test tools
4. Begin training pipeline

For implementation details, see:
- `HTK_STATE_DETECTION_IMPLEMENTATION.md` - Full HTK system design
- `QUICK_START_NEW_FEATURES.md` - Usage examples
- `README_REFACTORED.md` - Pipeline overview

---

## symbiote_weak constrained HTK inputs

For the `symbiote_weak` constrained mode, the following are additionally
recommended:

### Picklist-specific sequence constraints (default on)

- **Training / tuning**: Each dev/training video should have a matching label
  CSV (same filename stem as the video). Expected fine-state sequences are
  derived from those segments and passed into coarse / interact / carry
  decoders during `hmm_tune` (unless you pass `--no-sequence-constraint`).
  `hmm_train` records `use_sequence_constraint` in `pipeline_config.json`.
- **Inference**: Pass `--sequence-label-csv` pointing at the per-video label
  CSV for that picklist run, **or** place `<video_stem>.csv` next to the input
  video (same columns as training: `timestamp_start`, `timestamp_end`,
  `state`). Disable entirely with `--no-sequence-constraint`.
- **Feature cache**: After upgrading feature dimensionality, delete
  `{model_dir}/feature_cache/` once so `.npy` caches match the new layout
  (manifest includes `feature_dim` for automatic invalidation on new code).

1. Label CSVs at inference/tuning time (see above) for picklist-specific grammar
2. ARUCO temporal settings tuned per camera/setup:
   - `--aruco-persistence-frames` (old forward-fill style persistence)
   - `--aruco-smoothing-window` (old Gaussian/temporal smoothing intent)
3. Boundary cleanup threshold:
   - `--min-segment-seconds`
4. Boundary quality tracking outputs:
   - `symbiote_weak.hmm_tune` writes boundary RMSE in tuning grid
   - `symbiote_weak.hmm_boundary_eval` writes per-file/average RMSE summary
