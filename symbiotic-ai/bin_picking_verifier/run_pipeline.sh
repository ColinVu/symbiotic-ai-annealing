#!/bin/bash
# Complete pipeline: ArUco map generation, inference, evaluation

set -e

cd "$(dirname "$0")"

echo "=== Bin Picking Verifier Pipeline ==="
echo

# Step 1: Generate ArUco map from Excel
echo "[1/3] Generating ArUco map from Excel..."
python3 parse_aruco_excel.py \
  --excel ../randomized_Object_List.xlsx \
  --output aruco_map.json
echo

# Step 2: Run inference on all videos (with live accuracy if ground truth exists)
echo "[2/3] Running inference on picklist videos..."
if [ -f "../ground_truth.csv" ]; then
  echo "(Live per-video accuracy enabled)"
  python3 main.py \
    --videos ../hmm-testing/picklist_videos \
    --state-csvs ../hmm-testing/picklist_labels \
    --picklist-jsons ../hmm-testing/picklist_jsons \
    --aruco-map aruco_map.json \
    --output-dir ./out \
    --ground-truth ../ground_truth.csv \
    "$@"
else
  python3 main.py \
    --videos ../hmm-testing/picklist_videos \
    --state-csvs ../hmm-testing/picklist_labels \
    --picklist-jsons ../hmm-testing/picklist_jsons \
    --aruco-map aruco_map.json \
    --output-dir ./out \
    "$@"
fi
echo

# Step 3: Evaluate against ground truth (final detailed report)
echo "[3/3] Generating detailed evaluation report..."
python3 evaluate.py \
  --predictions ./out \
  --ground-truth ../ground_truth.csv \
  --output ./out/evaluation.json
echo

echo "=== Pipeline complete ==="
echo "Results in: ./out/"
echo "Evaluation: ./out/evaluation.json"
