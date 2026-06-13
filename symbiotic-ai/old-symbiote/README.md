# Old Symbiote (Pre-Refactor Code)

This directory contains ALL the original code before the refactoring.

## Contents

### Pipeline Files
- `video_to_classification_pipeline.py` - Original 2,140-line video training pipeline
- `video_to_classification_pipeline_backup.py` - Backup copy of the original
- `classifier_pipeline.py` - Original classifier training pipeline (1,354 lines)
- `batch_compare.py` - Batch comparison script
- `one_on_one.py` - One-on-one comparison script
- `main.py` - Original main entry point
- `extract_hand_snippets.py` - Hand snippet extraction utility (328 lines)

### Documentation
- `README.md` - This file
- `CLASSIFIER_README.md` - Original classifier pipeline documentation
- `BATCH_COMPARE_README.md` - Original batch compare documentation

### Library Files (Copied from `symbiote/lib/`)
These files are **copies** - the originals remain in `symbiote/lib/` because the new refactored code still uses them:

- `lib/embedding.py` - MODEL constant for CLIP model
- `lib/hand_detection.py` - Hand segmentation functionality  
- `lib/blurry.py` - Blur detection utilities
- `lib/inference.py` - Legacy inference utilities
- `lib/state_detection.py` - State detection utilities

## Why These Files Are Here

This directory serves as an archive of the pre-refactor code for:
1. **Reference** - Compare old vs new implementations
2. **Backup** - Preserve original working code
3. **Documentation** - Understanding how the refactor was done

## Important Notes

⚠️ **The lib files are COPIES** - The new refactored code in `symbiote/` still depends on:
- `symbiote/lib/embedding.py` (for MODEL constant)
- `symbiote/lib/hand_detection.py` (for segment_hand function)

These files must remain in `symbiote/lib/` for the new system to work!

## Using Old Code

If you need to use the old implementation:

```bash
cd old-symbiote
python video_to_classification_pipeline.py train --video path/to/video.mp4 --label "object"
```

## New Code Location

The refactored, modular code is in `../symbiote/` organized into:
- `core/`, `preprocessing/`, `embeddings/`, `datasets/`, `models/`
- `training/`, `persistence/`, `visualization/`, `inference/`
- `pipelines/`, `cli/`

See `../symbiote/README_REFACTORED.md` for details on the new structure.
