# nn_model

Supervised temporal state classifier for pick/carry/place/empty prediction.
Frame-level training plus Viterbi forced alignment produces compact
`frame_index,code` label files that annealing consumes from
[`hmm-testing/picklist_labels`](../hmm-testing/picklist_labels).

Run all commands from `symbiotic-ai/` (the directory that contains `nn_model/`).

## States

| Token | State        |
|-------|--------------|
| `m`   | CARRY_EMPTY  |
| `a`   | PICK         |
| `e`   | CARRY_WITH   |
| `i`   | PLACE        |

Legal cycle: `CARRY_EMPTY → PICK → CARRY_WITH → PLACE → CARRY_EMPTY → …`

## Default data layout

All runtime inputs live under [`hmm-testing/`](../hmm-testing):

| Path | Role |
|------|------|
| `hmm-testing/picklist_videos/` | Source MP4 videos (shared with annealing) |
| `hmm-testing/csv_labels/` | Hand-annotated training labels |
| `hmm-testing/processed_features_3d/` | Generated 3-D features (`p_hold direction net_aruco`) |
| `hmm-testing/picklist_jsons/` | Item lists used as Viterbi target sequences |
| `hmm-testing/picklist_labels/` | NN bulk output (and annealing `--manual-labels-dir`) |
| `ground_truth.csv` | Shared SKU / item-count matrix at the `symbiotic-ai/` root |

## End-to-end pipeline

### 1. Generate embeddings

Place `holding_head.pt` and `hand_landmarker.task` in `objectdetector/`, put
videos in `hmm-testing/picklist_videos/`, then run
[`objectdetector/hand_object_detector_100DOH_inference.ipynb`](../objectdetector/hand_object_detector_100DOH_inference.ipynb).

The notebook reads videos from `hmm-testing/picklist_videos` and writes
extensionless 3-D feature files to `hmm-testing/processed_features_3d`.
Detector CSVs stay in `objectdetector/csv_outputs/` as intermediates.

Alternatively copy already-generated `processed_features_3d/picklist_NNN` files
into `hmm-testing/processed_features_3d/`.

### 2. Train the NN

```bash
cd symbiotic-ai
python3 -m nn_model.train --output-dir nn_model/checkpoints --epochs 50
```

Training reads `hmm-testing/processed_features_3d` plus `hmm-testing/csv_labels`.

### 3. Run the NN on all videos (write annealing labels)

```bash
python3 -m nn_model.bulk_predict \
  --checkpoint nn_model/checkpoints/best_model.pt \
  --min-duration 8 --prior-scale 0.5 --trailing-m
```

Defaults:

- features: `hmm-testing/processed_features_3d`
- target sequences: `hmm-testing/picklist_jsons` (`n` items → `n × maei`)
- output: `hmm-testing/picklist_labels/picklist_NNN.csv`

Output format (matches existing annealing labels):

```csv
frame_index,code
0,m
6,a
109,e
218,i
```

### 4. Annealing

```bash
python -m symbiote_weak_generalized.cli.main train-from-cache \
  --videos hmm-testing/picklist_videos \
  --picklist-json-dir hmm-testing/picklist_jsons \
  --manual-labels-dir hmm-testing/picklist_labels \
  --output-dir models/classifier/my_run
```

`train-from-cache` needs matching CLIP `.npy` caches (or use `train`, which
embeds frames). See `symbiote_weak_generalized` CLI help for cache flags.

## Other commands

```bash
python3 -m nn_model.evaluate --checkpoint nn_model/checkpoints/best_model.pt --split val
python3 -m nn_model.predict --checkpoint nn_model/checkpoints/best_model.pt --picklist-ids 041
python3 -m nn_model.annotate --checkpoint nn_model/checkpoints/best_model.pt --picklist-ids 041
python3 -m nn_model.eval_labeled_picklists --checkpoint nn_model/checkpoints/best_model.pt
```

Optional editable install from `symbiotic-ai/`:

```bash
pip install -e .
```

## Feature inputs

Two modes via `--feature-source`:

- **`processed_3d` (default):** `hmm-testing/processed_features_3d/picklist_NNN` — `p_hold`, `direction`, `net_aruco`
- **`csv_outputs`:** raw detector CSVs from `objectdetector/csv_outputs/` expanded to 5-D

## Module layout

```
nn_model/
  states.py      # token/state constants
  data.py        # feature + label loaders and default paths
  models.py      # MLP and TCN
  decode.py      # constrained Viterbi decoder
  hmm_align.py   # forced alignment used by bulk_predict
  sequences.py   # target sequences from csv_labels / picklist_jsons
  labels_io.py   # HTK vs compact CSV read/write
  metrics.py     # accuracy, F1, carry-count, boundary RMSE
  train.py       # training CLI
  predict.py     # inference CLI
  annotate.py    # inference + video overlay CLI
  bulk_predict.py  # bulk label CSV export for annealing
  eval_labeled_picklists.py
  evaluate.py
```
