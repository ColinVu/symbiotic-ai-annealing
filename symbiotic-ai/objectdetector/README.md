# objectdetector

Feature extraction for the consolidated annealing pipeline.

## Generate 3-D embeddings

1. Put picklist videos in [`../hmm-testing/picklist_videos`](../hmm-testing/picklist_videos).
2. Place these local model files in this directory (not in git):
   - `holding_head.pt` — DINOv2 holding-head weights
   - `hand_landmarker.task` — MediaPipe hand landmarker
3. Open `hand_object_detector_100DOH_inference.ipynb` and run:
   - cells 1–2 for config + model
   - cells 3–6 for a single video (writes HTK features)
   - cell 8 for the batch list (writes HTK features for every listed video)

Outputs:

- `csv_outputs/` — per-video detector timeseries (intermediates)
- `../hmm-testing/processed_features_3d/picklist_NNN` — 3-D features for `nn_model`

## Extra scripts

- `build_aruco_htk_features.py` — optional 9-D HTK features from `csv_outputs/`
- `split_3d_to_4d_features.py` — optional transform of the 3-D files
- `aruco_net_value_inference.ipynb` / `Mediapipe_feature_engineering.ipynb` — debug / legacy

Dependencies for the notebooks: `torch`, `torchvision`, `timm`, `mediapipe`,
`opencv-python`, `Pillow`, `numpy`, `matplotlib`.
