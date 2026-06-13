# File Organization Summary

## What Was Done

### ✅ Created `old-symbiote/` Directory
Archive for pre-refactor code at: `symbiotic-ai/old-symbiote/`

### ✅ Moved Old Files to `old-symbiote/`
**All old scripts moved (no longer in symbiote/):**
- `video_to_classification_pipeline.py` → `old-symbiote/video_to_classification_pipeline.py`
- `video_to_classification_pipeline copy.py` → `old-symbiote/video_to_classification_pipeline_backup.py`
- `classifier_pipeline.py` → `old-symbiote/classifier_pipeline.py`
- `batch_compare.py` → `old-symbiote/batch_compare.py`
- `one_on_one.py` → `old-symbiote/one_on_one.py`
- `main.py` → `old-symbiote/main.py`
- `extract_hand_snippets.py` → `old-symbiote/extract_hand_snippets.py`
- `CLASSIFIER_README.md` → `old-symbiote/CLASSIFIER_README.md`
- `BATCH_COMPARE_README.md` → `old-symbiote/BATCH_COMPARE_README.md`

### ✅ Copied Shared Library Files
**Copied to `old-symbiote/lib/` (originals remain in `symbiote/lib/`):**
- `embedding.py` - Used by BOTH old and new code
- `hand_detection.py` - Used by BOTH old and new code  
- `blurry.py` - Copied for old code reference
- `inference.py` - Copied for old code reference
- `state_detection.py` - Copied for old code reference

### ✅ Kept Essential Files in `symbiote/lib/`
**Still in place (required by new refactored code):**
- `symbiote/lib/embedding.py` - Provides MODEL constant
- `symbiote/lib/hand_detection.py` - Provides segment_hand function
- `symbiote/lib/blurry.py` - Available for reference
- `symbiote/lib/inference.py` - Available for reference
- `symbiote/lib/state_detection.py` - Available for reference

## Current Structure

```
symbiotic-ai/
├── old-symbiote/                           # ARCHIVE of ALL pre-refactor code
│   ├── README.md                          # Documentation
│   ├── video_to_classification_pipeline.py
│   ├── video_to_classification_pipeline_backup.py
│   ├── classifier_pipeline.py
│   ├── batch_compare.py
│   ├── one_on_one.py
│   ├── main.py
│   ├── extract_hand_snippets.py
│   ├── CLASSIFIER_README.md
│   ├── BATCH_COMPARE_README.md
│   └── lib/                               # COPIES of shared files
│       ├── embedding.py
│       ├── hand_detection.py
│       ├── blurry.py
│       ├── inference.py
│       └── state_detection.py
│
└── symbiote/                              # ACTIVE refactored code ONLY
    ├── lib/                               # ORIGINALS (still needed!)
    │   ├── embedding.py                   # Used by new code
    │   ├── hand_detection.py              # Used by new code
    │   ├── blurry.py
    │   ├── inference.py
    │   └── state_detection.py
    │
    ├── core/                              # New modular structure
    ├── preprocessing/
    ├── embeddings/
    ├── datasets/
    ├── models/
    ├── training/
    ├── persistence/
    ├── visualization/
    ├── inference/
    ├── state_detection/                   # 🆕 NEW: State detection framework
    │   ├── __init__.py
    │   └── detector.py                    # HandState enum, detect_states_from_video()
    ├── pipelines/
    │   ├── video_training.py              # ✏️ UPDATED: Integrated state detection
    │   ├── video_inference.py             # 🆕 NEW: CSV inference pipeline
    │   └── image_training.py
    ├── cli/
    │   └── main.py                        # ✏️ UPDATED: Added infer command
    ├── test_imports.py
    ├── test_new_features.py               # 🆕 NEW: Tests for new features
    ├── README_REFACTORED.md
    └── REFACTORING_SUMMARY.md
```

## Important Notes

⚠️ **DO NOT DELETE `symbiote/lib/`** - The new refactored code depends on:
- `symbiote/lib/embedding.py`
- `symbiote/lib/hand_detection.py`

✅ **Safe to delete** (if you want):
- Entire `old-symbiote/` directory (it's just an archive)

## What This Achieves

1. **Clean Separation** - Old code isolated from new code
2. **Preserved History** - Original implementation archived
3. **Working New System** - Refactored code remains fully functional
4. **Shared Dependencies** - lib files available to both systems
5. **Clear Documentation** - READMEs explain what's where
6. **New Features** - State detection and video inference pipelines added

## Recent Updates (Feb 15, 2026)

### 🆕 Features Added

**State Detection Framework:**
- New `state_detection/` module
- `HandState` enum (PICK, CARRY_WITH, PLACE, CARRY_WITHOUT)
- Placeholder `detect_states_from_video()` function
- Integrated into video training pipeline

**Video Inference Pipeline:**
- New `pipelines/video_inference.py`
- Inference-only workflow with CSV output
- Does not pollute training cache
- New CLI `infer` command

**Modified Files:**
- `preprocessing/video_processor.py` - Added state detection support
- `pipelines/video_training.py` - Integrated state filtering
- `cli/main.py` - Added infer command

**New Documentation:**
- `IMPLEMENTATION_SUMMARY.md` - Full implementation details
- `QUICK_START_NEW_FEATURES.md` - Usage guide

## Next Steps (Optional)

If you want the new code to be **completely independent** from old lib files:
1. Extract MODEL constant into `core/config.py` directly
2. Move `segment_hand` into a new `preprocessing/hand_segmentation.py` module
3. Then `symbiote/lib/` could be removed entirely

For now, both systems can coexist with shared lib dependencies.
