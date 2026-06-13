# Quick Start Guide: New Features

> **🐳 Docker Users**: See `DOCKER_SETUP.md` for containerized setup instructions. All commands below can be run inside Docker by prefixing with `docker-compose run --rm symbiotic-ai` and using container paths (e.g., `/data/videos/` instead of `../videos/`).

## Video Inference Pipeline

### What It Does
Runs inference on a video and outputs frame-by-frame predictions to CSV **without** adding data to the training cache.

### Basic Usage
```bash
python -m symbiote.cli.main infer \
    --video path/to/video.mp4 \
    --model-dir path/to/trained/model \
    --output results.csv
```

### Full Options
```bash
python -m symbiote.cli.main infer \
    --video path/to/video.mp4 \
    --model-dir ../models/classifier/model_name \
    --output results.csv \
    --frame-skip 5 \              # Process every 5th frame (default: 5)
    --threshold 100.0 \           # Blur threshold (default: 100.0)
    --verbose                     # Show progress (default: True)
```

### Output CSV Format
```csv
frame_number,timestamp,predicted_label,confidence,top_3_labels,top_3_confidences
10,0.333,apple,0.95,"apple;banana;orange","0.95;0.03;0.02"
15,0.500,apple,0.97,"apple;orange;banana","0.97;0.02;0.01"
20,0.667,banana,0.88,"banana;apple;orange","0.88;0.09;0.03"
```

### When to Use
- Testing model performance on new videos
- Batch processing videos for analysis
- Generating predictions without training
- Creating labeled datasets for review

---

## State Detection Framework

### What It Does
Filters video frames during training to only cache frames in specific states (e.g., only frames where hand is carrying an object).

### Current Behavior (Placeholder)
- **All frames** are marked as "CARRY_WITH" state
- **No filtering** occurs (backward compatible)
- Framework is ready for future algorithm integration

### How It Works (Training Pipeline)
```bash
python -m symbiote.cli.main train \
    --video path/to/video.mp4 \
    --label "object_name"
```

**Automatically:**
1. Extracts frames from video
2. Generates CLIP embeddings
3. **NEW**: Runs state detection on all embeddings
4. **NEW**: Filters to only cache "CARRY_WITH" frames
5. **NEW**: Saves state detection results to CSV
6. Trains model on accumulated cache

### Output Files
```
models/classifier/video_name/
├── model_weights.pth
├── model_metadata.json
├── training_history.png
├── confusion_matrix.png
├── evaluation_results.json
└── state_detection.csv         ← NEW!
```

### State Detection CSV Format
```csv
timestamp_start,timestamp_end,state
0.0,10.5,CARRY_WITH
```

### Future: When Real Algorithm is Added
Once the actual state detection algorithm is implemented:
- Replace function in `state_detection/detector.py`
- Training will automatically filter frames
- Only CARRY_WITH frames added to cache
- Reduced false positives from empty-hand frames

### The Four States
1. **PICK** - Hand reaching/grabbing object
2. **CARRY_WITH** - Hand holding object (what gets trained)
3. **PLACE** - Hand releasing object
4. **CARRY_WITHOUT** - Hand visible but no object

---

## Programmatic Usage

### Video Inference (Python)
```python
from symbiote.pipelines.video_inference import run_video_inference

csv_path = run_video_inference(
    video_path="path/to/video.mp4",
    model_dir="path/to/model",
    output_csv="results.csv",
    threshold=100.0,
    frame_skip=5,
    verbose=True
)
```

### State Detection (Python)
```python
from symbiote.state_detection import HandState, detect_states_from_video

# Currently returns placeholder (all CARRY_WITH)
state_results = detect_states_from_video(
    video_path="path/to/video.mp4",
    embeddings=list_of_embeddings,
    frame_numbers=[10, 15, 20, 25],
    fps=30.0
)

print(state_results)
# timestamp_start  timestamp_end  state
# 0.333            0.833          CARRY_WITH
```

### Custom Training with State Filter
```python
from symbiote.pipelines.video_training import run_video_training
from symbiote.core.config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["max_epochs"] = 50

run_video_training(
    video_path="path/to/video.mp4",
    label="my_object",
    base_output_dir="../models/classifier",
    config=config,
    threshold=100.0,
    frame_skip=4,
    verbose=True
)
# State detection automatically applied!
```

---

## Common Workflows

### Workflow 1: Train Model with Videos
```bash
# Add training data from multiple videos
python -m symbiote.cli.main train --video video1.mp4 --label "apple"
python -m symbiote.cli.main train --video video2.mp4 --label "banana"
python -m symbiote.cli.main train --video video3.mp4 --label "apple"
# Each run accumulates data in cache and retrains model
```

### Workflow 2: Test Model on New Video
```bash
# Run inference on test video
python -m symbiote.cli.main infer \
    --video test_video.mp4 \
    --model-dir ../models/classifier/video3 \
    --output test_results.csv

# Review results.csv
# If accuracy is poor, add more training data
```

### Workflow 3: Batch Inference
```bash
# Process multiple videos
for video in test_videos/*.mp4; do
    python -m symbiote.cli.main infer \
        --video "$video" \
        --model-dir ../models/classifier/best_model \
        --output "results_$(basename $video .mp4).csv"
done
```

---

## Tips and Best Practices

### Frame Skip
- **Training**: Use 4-6 for balance (more data vs speed)
- **Inference**: Use 5-10 for faster processing
- Lower = more data, slower processing
- Higher = less data, faster processing

### Blur Threshold
- **Default**: 100.0 (works for most cases)
- **Increase**: 120-150 for stricter filtering
- **Decrease**: 70-90 if too many frames rejected

### Model Selection
- Use most recent model from output directory
- Models are saved per video: `models/classifier/video_name/`
- Each training run creates new model with all accumulated data

### State Detection (Current)
- Currently no functional effect (placeholder)
- Output CSV shows all frames as CARRY_WITH
- Safe to ignore until real algorithm added
- Architecture is ready for seamless integration

---

## Troubleshooting

### "No frames could be processed"
- Check blur threshold (try lowering it)
- Verify hands are visible in video
- Try lower frame skip value

### "Hand detection failed"
- Ensure hand is clearly visible
- Check lighting conditions
- Try different frames/videos

### Import Errors
```bash
# Test imports
python symbiote/test_new_features.py
```

### See Available Commands
```bash
python -m symbiote.cli.main --help
python -m symbiote.cli.main train --help
python -m symbiote.cli.main infer --help
python -m symbiote.cli.main predict --help
```

---

## Next Steps

1. **Try Video Inference**
   - Test on sample video
   - Review CSV output
   - Adjust frame-skip and threshold

2. **Train with State Detection**
   - Note `state_detection.csv` in output
   - Currently shows placeholder data
   - Ready for algorithm integration

3. **Integrate Real Algorithm** (Future)
   - Update `state_detection/detector.py`
   - Keep same function signature
   - Automatic integration with pipeline
