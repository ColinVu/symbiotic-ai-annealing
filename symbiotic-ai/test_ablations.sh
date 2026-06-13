#!/bin/bash
# Ablation test suite for weak HMM upgrade
# Tests each change independently to identify performance regressions

set -e

VIDEOS="/data/hmm-testing/picklist_videos"
LABELS="/data/hmm-testing/picklist_labels"
ARUCO="/data/aruco_config/aruco_bins.json"
RESULTS_DIR="./ablation_results"

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Weak HMM Ablation Test Suite"
echo "=========================================="
echo ""
echo "This will run 4 experiments:"
echo "  1. Baseline (no constraints, full 29D)"
echo "  2. With constraints (full 29D)"
echo "  3. No color (17D + constraints)"
echo "  4. Single ARUCO channel (27D + constraints)"
echo ""
echo "Each run takes ~same time as normal train+tune."
echo "Results will be saved to: $RESULTS_DIR/"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Helper function to extract metrics from tune output
extract_metrics() {
    local log_file=$1
    local output_csv=$2
    
    # Extract the "Best:" line which has all key metrics
    grep "\\[hmm_tune\\] Best:" "$log_file" | tail -1 > "$output_csv.tmp"
    
    # Also extract full last grid line (has per-class recalls)
    grep "\\[hmm_tune\\] p=" "$log_file" | tail -1 >> "$output_csv.tmp"
    
    echo "Extracted metrics to: $output_csv"
}

# ============================================================
# TEST 1: Baseline (no constraints)
# ============================================================
echo ""
echo "========================================"
echo "TEST 1: NO CONSTRAINTS (29D features)"
echo "========================================"

TEST1_DIR="/models/htk_weak_test1_no_constraints"
TEST1_LOG="$RESULTS_DIR/test1_no_constraints.log"

echo "Clearing cache and training..."
docker-compose run --rm symbiotic-ai rm -rf /models/htk_weak/feature_cache
docker-compose run --rm symbiotic-ai python -m symbiote_weak.hmm_train \
    --video-dir "$VIDEOS" \
    --label-dir "$LABELS" \
    --output-dir "$TEST1_DIR" \
    --aruco-config "$ARUCO" \
    --pipeline two-stage \
    --aruco-persistence-frames 30 \
    --aruco-smoothing-window 9 \
    --min-segment-seconds 0.15 \
    --no-sequence-constraint \
    --tune-decode 2>&1 | tee "$TEST1_LOG"

extract_metrics "$TEST1_LOG" "$RESULTS_DIR/test1_metrics.txt"

# ============================================================
# TEST 2: With constraints (current full implementation)
# ============================================================
echo ""
echo "========================================"
echo "TEST 2: WITH CONSTRAINTS (29D features)"
echo "========================================"

TEST2_DIR="/models/htk_weak_test2_with_constraints"
TEST2_LOG="$RESULTS_DIR/test2_with_constraints.log"

echo "Clearing cache and training..."
docker-compose run --rm symbiotic-ai rm -rf "$TEST1_DIR/feature_cache"
docker-compose run --rm symbiotic-ai python -m symbiote_weak.hmm_train \
    --video-dir "$VIDEOS" \
    --label-dir "$LABELS" \
    --output-dir "$TEST2_DIR" \
    --aruco-config "$ARUCO" \
    --pipeline two-stage \
    --aruco-persistence-frames 30 \
    --aruco-smoothing-window 9 \
    --min-segment-seconds 0.15 \
    --tune-decode 2>&1 | tee "$TEST2_LOG"

extract_metrics "$TEST2_LOG" "$RESULTS_DIR/test2_metrics.txt"

# ============================================================
# TEST 3: No color descriptors (17D)
# ============================================================
echo ""
echo "========================================"
echo "TEST 3: NO COLOR (17D, with constraints)"
echo "========================================"
echo "MANUAL STEP REQUIRED:"
echo "  1. Edit symbiote_weak/state_detection/feature_extraction.py"
echo "  2. Comment out color_hs line in features concatenation"
echo "  3. Change config.py feature_dim to 17"
echo "  4. Press ENTER when ready..."
read -r

TEST3_DIR="/models/htk_weak_test3_no_color"
TEST3_LOG="$RESULTS_DIR/test3_no_color.log"

echo "Clearing cache and training..."
docker-compose run --rm symbiotic-ai rm -rf "$TEST2_DIR/feature_cache"
docker-compose run --rm symbiotic-ai python -m symbiote_weak.hmm_train \
    --video-dir "$VIDEOS" \
    --label-dir "$LABELS" \
    --output-dir "$TEST3_DIR" \
    --aruco-config "$ARUCO" \
    --pipeline two-stage \
    --aruco-persistence-frames 30 \
    --aruco-smoothing-window 9 \
    --min-segment-seconds 0.15 \
    --tune-decode 2>&1 | tee "$TEST3_LOG"

extract_metrics "$TEST3_LOG" "$RESULTS_DIR/test3_metrics.txt"

echo ""
echo "MANUAL STEP: Revert feature_extraction.py and config.py changes before continuing"
read -p "Press ENTER when reverted..." -r

# ============================================================
# TEST 4: Single ARUCO channel (27D)
# ============================================================
echo ""
echo "========================================"
echo "TEST 4: SINGLE ARUCO CHANNEL (27D)"
echo "========================================"
echo "MANUAL STEP REQUIRED:"
echo "  1. Edit feature_extraction.py to use only [aruco_signed] in concat"
echo "  2. Change config.py feature_dim to 27"
echo "  3. Press ENTER when ready..."
read -r

TEST4_DIR="/models/htk_weak_test4_single_aruco"
TEST4_LOG="$RESULTS_DIR/test4_single_aruco.log"

echo "Clearing cache and training..."
docker-compose run --rm symbiotic-ai rm -rf "$TEST3_DIR/feature_cache"
docker-compose run --rm symbiotic-ai python -m symbiote_weak.hmm_train \
    --video-dir "$VIDEOS" \
    --label-dir "$LABELS" \
    --output-dir "$TEST4_DIR" \
    --aruco-config "$ARUCO" \
    --pipeline two-stage \
    --aruco-persistence-frames 30 \
    --aruco-smoothing-window 9 \
    --min-segment-seconds 0.15 \
    --tune-decode 2>&1 | tee "$TEST4_LOG"

extract_metrics "$TEST4_LOG" "$RESULTS_DIR/test4_metrics.txt"

echo ""
echo "========================================"
echo "TESTS COMPLETE"
echo "========================================"
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "Next steps:"
echo "  1. Run: python analyze_ablations.py"
echo "  2. Review the comparison table"
echo "  3. Check detailed logs in $RESULTS_DIR/*.log"
echo ""
