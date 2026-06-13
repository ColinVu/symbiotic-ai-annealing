# Object Recognition Classifier Pipeline

Train a lightweight classifier on CLIP embeddings to recognize objects held in hand images.

## Overview

This pipeline:
1. Uses the **same embedding pipeline** as `batch_compare.py` (hand segmentation + CLIP)
2. **Caches embeddings** in `output_dir/.cache/` so subsequent runs skip re-embedding unchanged images
3. Trains a **lightweight MLP classifier** on frozen CLIP embeddings
4. Supports **folder-based dataset organization** (folder name = class label)
5. Includes **train/val/test splitting** with stratification
6. Provides an **inference API** for production use

## Data Directory Structure

Organize your training data like this:

```
images/image-testing/
├── a/
│   ├── item01-1.JPG
│   ├── item01-2.JPG
│   └── ...
├── b/
│   ├── item02-1.JPG
│   └── ...
├── c/
│   └── ...
└── ...
```

- Each **subfolder name** is the class label (single character recommended)
- Place training images inside each class folder
- Images should show hands holding objects (same as other tools in this repo)

## Usage

### Training

```bash
cd symbiote

# Basic training (uses default settings)
python classifier_pipeline.py train --data-dir ../images/image-testing

# With custom output directory
python classifier_pipeline.py train --data-dir ../images/image-testing --output-dir ../models/my_classifier

# With custom hyperparameters
python classifier_pipeline.py train \
    --data-dir ../images/image-testing \
    --output-dir ../models/my_classifier \
    --epochs 200 \
    --patience 20 \
    --lr 0.0005 \
    --hidden-dim 256

# Disable embedding cache (re-embed all images every run)
python classifier_pipeline.py train --data-dir ../images/image-testing --no-cache
```

### Inference (Single Image)

```bash
# Basic prediction
python classifier_pipeline.py predict --model-dir ../models/classifier --image path/to/image.jpg

# Top-3 predictions
python classifier_pipeline.py predict --model-dir ../models/classifier --image path/to/image.jpg --top-k 3
```

### Python API

```python
from classifier_pipeline import ObjectRecognizer

# Load trained model
recognizer = ObjectRecognizer("path/to/model_dir")

# Predict
result = recognizer.predict("path/to/image.jpg")
print(f"Label: {result['label']}, Confidence: {result['confidence']:.2f}")

# Get all class probabilities
print(result['all_scores'])

# Top-k predictions
top_3 = recognizer.predict_top_k("path/to/image.jpg", k=3)
for label, score in top_3:
    print(f"{label}: {score:.2f}")
```

## Output Files

After training, the output directory contains:

| File | Description |
|------|-------------|
| `model_weights.pth` | Trained classifier weights |
| `model_metadata.json` | Label mapping, dimensions, config |
| `training_history.png` | Loss and accuracy curves |
| `confusion_matrix.png` | Per-class performance visualization |
| `evaluation_results.json` | Accuracy metrics and class labels |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIP Encoder (Frozen)                     │
│  Image → Hand Segmentation → CLIP → 512-dim embedding        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Classifier Head (Trained)                  │
│  Linear(512, 128) → ReLU → Dropout(0.3) → Linear(128, N)    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                     Softmax → Predicted Class
```

- **CLIP encoder**: Completely frozen, provides powerful image features
- **Classifier head**: Small MLP, fast to train, lightweight

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_ratio` | 0.70 | Fraction of data for training |
| `val_ratio` | 0.15 | Fraction of data for validation |
| `test_ratio` | 0.15 | Fraction of data for testing |
| `batch_size` | 16 | Training batch size |
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `max_epochs` | 100 | Maximum training epochs |
| `early_stopping_patience` | 10 | Epochs to wait before early stopping |
| `hidden_dim` | 128 | Hidden layer size in classifier |
| `dropout` | 0.3 | Dropout rate |

## Training Details

### Data Splitting
- **Stratified split**: Each class appears proportionally in train/val/test
- **Deterministic**: Same random seed produces same splits
- **No leakage**: Same image never appears in multiple splits

### Training Process
1. Embed all images with CLIP (frozen)
2. Split into train (70%), validation (15%), test (15%)
3. Train classifier with cross-entropy loss
4. Monitor validation loss for early stopping
5. Restore best model after training

### Early Stopping
- Monitors validation loss
- Stops if no improvement for `patience` epochs
- Restores best model weights

## Evaluation Metrics

- **Top-1 Accuracy**: Correct if highest-confidence prediction matches true label
- **Top-3 Accuracy**: Correct if true label is in top 3 predictions
- **Confusion Matrix**: Shows which classes get confused with each other
- **Per-class Precision/Recall/F1**: Detailed performance breakdown

## Embedding cache

- **First run**: All images are embedded and results are saved under `output_dir/.cache/`
- **Later runs**: Cached embeddings are loaded for unchanged images (keyed by path + modification time)
- **When to use `--no-cache`**: After changing images, or to force a full re-embed (e.g. after changing CLIP or preprocessing)
- **Large datasets**: Caching makes re-runs much faster when you only add or change a few images

## Tips

1. **Minimum samples per class**: Aim for at least 5-10 images per class
2. **Balanced classes**: Try to have similar numbers of images per class
3. **Image quality**: Use clear, well-lit images with visible hands
4. **Early stopping**: If training stops early, your model likely converged
5. **Overfitting**: If train accuracy >> val accuracy, reduce hidden_dim or increase dropout

## Troubleshooting

### "No subfolders found"
- Make sure your data directory has subfolders (one per class)
- Example: `images/image-testing/a/`, `images/image-testing/b/`, etc.

### "No images could be embedded"
- Check that images show clear, visible hands
- Try with images that work in `batch_compare.py` first

### Low accuracy
- Add more training images
- Check for mislabeled images
- Ensure classes are visually distinct
- Try increasing `hidden_dim` or training longer

### Model too slow
- Reduce `hidden_dim` for faster inference
- The bottleneck is CLIP encoding, not the classifier

## Integration with Other Tools

This pipeline uses the **exact same embedding function** as `batch_compare.py`:
- Same hand segmentation (`segment_hand`)
- Same CLIP model (`openai/clip-vit-base-patch32`)
- Same preprocessing

So if images work in `batch_compare.py`, they'll work here.
