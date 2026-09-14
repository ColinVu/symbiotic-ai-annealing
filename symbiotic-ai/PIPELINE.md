# Consolidated pipeline

Run everything from this `symbiotic-ai/` directory.

```
videos  →  objectdetector notebook  →  processed_features_3d
        →  nn_model.train           →  checkpoint
        →  nn_model.bulk_predict    →  picklist_labels
        →  annealing train-from-cache
```

## Commands

```bash
# 1. Extract 3-D features (notebook)
#    objectdetector/hand_object_detector_100DOH_inference.ipynb
#    reads hmm-testing/picklist_videos
#    writes hmm-testing/processed_features_3d

# 2. Train NN
python3 -m nn_model.train --output-dir nn_model/checkpoints --epochs 50

# 3. Label all videos for annealing
python3 -m nn_model.bulk_predict \
  --checkpoint nn_model/checkpoints/best_model.pt \
  --min-duration 8 --prior-scale 0.5 --trailing-m

# 4. Anneal
python -m symbiote_weak_generalized.cli.main train-from-cache \
  --videos hmm-testing/picklist_videos \
  --picklist-json-dir hmm-testing/picklist_jsons \
  --manual-labels-dir hmm-testing/picklist_labels \
  --output-dir models/classifier/my_run
```

## K-fold cross-validation

Same pipeline, with one immutable split shared by every stage. First `nn_model.train --k-fold` writes `dataset_manifest.json` and `fold_NN/fold_manifest.json` under `--cv-results-dir`. Later stages reuse those manifests; they do not resplit. The shared CLIP cache stays in the existing classifier `.cache/` (outside the CV tree). Fold checkpoints, generated labels, annealing models, predictions, and metrics all live under `--cv-results-dir`.

CV dataset discovery uses matching video and picklist JSON stems. It does not
read the legacy `hmm-testing/picklist_labels/` directory; fold-specific
annealing labels are generated in Stage 2.

```bash
# 1. Train one NN per fold (creates the CV manifests)
python3 -m nn_model.train --k-fold 5 --cv-results-dir cv_results --epochs 50

# 2. Generate train/test compact labels from each fold checkpoint
python3 -m nn_model.create_labels --k-fold 5 --cv-results-dir cv_results \
  --trailing-m --min-duration 8 --prior-scale 0.5

# 3. Fit one annealing model per fold on that fold's train IDs + train labels
python -m symbiote_weak_generalized.cli.main train-from-cache \
  --k-fold 5 --cv-results-dir cv_results \
  --videos hmm-testing/picklist_videos \
  --picklist-json-dir hmm-testing/picklist_jsons \
  --cache-dir models/classifier/my_run/.cache \
  --ground-truth-csv ground_truth.csv

# 4. Held-out test + summary (uses generated test labels only)
python3 -m cross_validation.test --k-fold 5 --cv-results-dir cv_results \
  --videos-dir hmm-testing/picklist_videos \
  --ground-truth-csv ground_truth.csv \
  --cache-dir models/classifier/my_run/.cache
```

Interrupted stages resume completed folds unless you pass `--overwrite`. Summaries are written to `cv_results/summary/` (`fold_metrics.csv`, `fold_metrics.json`, `aggregate_metrics.json`) and never mix training-fit assignment scores with held-out top-1/top-3.

The held-out annealing summary also reports frame-level accuracy, the existing
segment Top-1 accuracy, and segment Top-1 accuracy with predictions restricted
to the video's shelf. `aggregate_metrics.json` includes all three metrics
overall and separately for shelves `c` through `g`; shelf selection maps
picklist suffixes `1` through `5` to `c` through `g`.

Full `train` also accepts `--k-fold` / `--cv-results-dir` / `--cache-dir`: it reuses cached embeddings and only embeds missing in-interval frames.

## Files you still need to add

Already in this repo: `hmm-testing/picklist_jsons/`, `hmm-testing/picklist_labels/`
(existing annealing labels, overwritten by bulk predict), `ground_truth.csv`,
`config/aruco_bins.json`, `hmm-testing/hand_embeddings/`.

Add locally (not copied by this consolidation):

| Input | Where |
|-------|--------|
| Picklist videos `picklist_NNN.MP4` | `hmm-testing/picklist_videos/` |
| Hand annotations `picklist_NNN.csv` | `hmm-testing/csv_labels/` (copy from segmentation `csv_labels/`) |
| DINOv2 holding head | `objectdetector/holding_head.pt` |
| MediaPipe landmarker | `objectdetector/hand_landmarker.task` |
| 3-D features (if skipping extraction) | `hmm-testing/processed_features_3d/picklist_NNN` |
| Trained NN (if skipping training) | `nn_model/checkpoints/best_model.pt` |
| CLIP cache `.npy` files (train-from-cache) | cache dir passed to annealing, or run `train` to embed |

Python extras for extraction notebooks: `torch`, `torchvision`, `timm`,
`mediapipe`, `opencv-python`, `Pillow`, `numpy`, `matplotlib`.
NN training additionally needs `torch` and `numpy` (see `pyproject.toml`).
