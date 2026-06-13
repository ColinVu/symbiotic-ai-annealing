#!/bin/bash
# Quick check: Just test constraints on/off with current code
# Fastest ablation to identify if constraints are the main issue

set -e

VIDEOS="/data/hmm-testing/picklist_videos"
LABELS="/data/hmm-testing/picklist_labels"
ARUCO="/data/aruco_config/aruco_bins.json"
RESULTS_DIR="./quick_ablation_results"

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Quick Ablation: Constraints ON vs OFF"
echo "=========================================="
echo ""
echo "This tests whether sequence constraints are the problem."
echo "Runtime: ~same as one train+tune cycle"
echo ""

# ============================================================
# Clear cache once
# ============================================================
echo "Clearing feature cache..."
docker-compose run --rm symbiotic-ai rm -rf /models/htk_weak/feature_cache

# ============================================================
# Test WITHOUT constraints
# ============================================================
echo ""
echo "========================================"
echo "TEST: NO CONSTRAINTS"
echo "========================================"

docker-compose run --rm symbiotic-ai python -m symbiote_weak.hmm_train \
    --video-dir "$VIDEOS" \
    --label-dir "$LABELS" \
    --output-dir /models/htk_weak_no_constraints \
    --aruco-config "$ARUCO" \
    --pipeline two-stage \
    --aruco-persistence-frames 30 \
    --aruco-smoothing-window 9 \
    --min-segment-seconds 0.15 \
    --no-sequence-constraint \
    --tune-decode 2>&1 | tee "$RESULTS_DIR/no_constraints.log"

echo ""
echo "Extracting metrics..."
grep "\[hmm_tune\] Best:" "$RESULTS_DIR/no_constraints.log" | tail -1 > "$RESULTS_DIR/no_constraints_best.txt"
grep "\[hmm_tune\] p=" "$RESULTS_DIR/no_constraints.log" | tail -1 >> "$RESULTS_DIR/no_constraints_best.txt"

# ============================================================
# Test WITH constraints (default)
# ============================================================
echo ""
echo "========================================"
echo "TEST: WITH CONSTRAINTS (default)"
echo "========================================"

docker-compose run --rm symbiotic-ai rm -rf /models/htk_weak_no_constraints/feature_cache

docker-compose run --rm symbiotic-ai python -m symbiote_weak.hmm_train \
    --video-dir "$VIDEOS" \
    --label-dir "$LABELS" \
    --output-dir /models/htk_weak_with_constraints \
    --aruco-config "$ARUCO" \
    --pipeline two-stage \
    --aruco-persistence-frames 30 \
    --aruco-smoothing-window 9 \
    --min-segment-seconds 0.15 \
    --tune-decode 2>&1 | tee "$RESULTS_DIR/with_constraints.log"

echo ""
echo "Extracting metrics..."
grep "\[hmm_tune\] Best:" "$RESULTS_DIR/with_constraints.log" | tail -1 > "$RESULTS_DIR/with_constraints_best.txt"
grep "\[hmm_tune\] p=" "$RESULTS_DIR/with_constraints.log" | tail -1 >> "$RESULTS_DIR/with_constraints_best.txt"

# ============================================================
# Compare results
# ============================================================
echo ""
echo "=========================================="
echo "RESULTS COMPARISON"
echo "=========================================="
echo ""

echo "NO CONSTRAINTS:"
cat "$RESULTS_DIR/no_constraints_best.txt"
echo ""

echo "WITH CONSTRAINTS:"
cat "$RESULTS_DIR/with_constraints_best.txt"
echo ""

echo "=========================================="
echo "INTERPRETATION GUIDE"
echo "=========================================="
echo ""
echo "Compare the two 'Best:' lines above:"
echo ""
echo "1. MACRO-F1 (higher is better, range 0-1):"
echo "   - >0.6 = Good"
echo "   - 0.3-0.6 = Moderate"
echo "   - <0.3 = Poor"
echo ""
echo "2. BOUNDARY_RMSE (lower is better, in seconds):"
echo "   - <2s = Excellent"
echo "   - 2-5s = Good"
echo "   - 5-10s = Moderate"
echo "   - >10s = Poor"
echo ""
echo "3. Per-class Recalls R(P/CW/PL/CE) (higher is better, range 0-1):"
echo "   - Balanced (all >0.5) = Good"
echo "   - One high, others low = Class imbalance issue"
echo "   - All low (<0.3) = Model not learning"
echo ""
echo "DECISION GUIDE:"
echo ""
echo "If NO CONSTRAINTS has MUCH better F1/recalls:"
echo "  → Sequence constraints are the problem"
echo "  → Check: Are label CSVs correct for each video?"
echo "  → Consider running without constraints"
echo ""
echo "If BOTH are similarly bad:"
echo "  → Constraints are not the main issue"
echo "  → Test other ablations (color, ARUCO channels)"
echo "  → Check training data quality/quantity"
echo ""
echo "Full logs saved to: $RESULTS_DIR/"
echo ""
