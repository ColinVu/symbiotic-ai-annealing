#!/bin/bash
# Convenience script to run frame similarity visualizer
# Usage: ./run_frame_similarity.sh [n_samples] [hand_neutralize_components]

set -e

N_SAMPLES=${1:-20}
HAND_NEUTRALIZE=${2:-50}

OUTPUT_DIR="frame_similarity_out_${HAND_NEUTRALIZE}"

echo "Running frame similarity visualizer..."
echo "  Samples: ${N_SAMPLES}"
echo "  Hand neutralization components: ${HAND_NEUTRALIZE}"
echo "  Output directory: ${OUTPUT_DIR}"
echo ""

python3 frame_similarity_visualizer.py \
  --models-root models/classifier \
  --manual-labels symbiotic-ai/hmm-testing/picklist_labels \
  --hand-neutralize "${HAND_NEUTRALIZE}" \
  --n-samples "${N_SAMPLES}" \
  --output-dir "${OUTPUT_DIR}" \
  --comparisons-per-query 20 \
  --seed 42

echo ""
echo "Done! Check ${OUTPUT_DIR}/ for results"
