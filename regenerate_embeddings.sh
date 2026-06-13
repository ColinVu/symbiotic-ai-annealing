#!/bin/bash
# Script to regenerate embeddings after hand selection bug fix
# Run this from the 022026 directory

set -e

echo "============================================================"
echo "REGENERATING EMBEDDINGS AFTER HAND SELECTION BUG FIX"
echo "============================================================"
echo ""
echo "This will:"
echo "  1. Re-extract empty-hand embeddings for HandNeutralizer"
echo "  2. Remind you to re-run training to regenerate item embeddings"
echo "  3. Re-run embedding analysis"
echo "  4. Re-run frame similarity visualizer"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "============================================================"
echo "STEP 1: Re-extracting empty-hand embeddings"
echo "============================================================"
echo ""

cd symbiotic-ai

# Backup old embeddings
if [ -d "hmm-testing/hand_embeddings" ]; then
    echo "Backing up old hand embeddings..."
    BACKUP_DIR="hmm-testing/hand_embeddings_backup_$(date +%Y%m%d_%H%M%S)"
    mv hmm-testing/hand_embeddings "$BACKUP_DIR"
    echo "  Old embeddings saved to: $BACKUP_DIR"
fi

echo "Extracting new empty-hand embeddings..."
python3 -m symbiote_weak_generalized.scripts.extract_empty_hand_embeddings \
  --videos-dir hmm-testing/picklist_videos \
  --labels-dir hmm-testing/picklist_labels \
  --output-dir hmm-testing/hand_embeddings

echo ""
echo "✓ Empty-hand embeddings regenerated"
echo ""

cd ..

echo "============================================================"
echo "STEP 2: Item embeddings (MANUAL STEP REQUIRED)"
echo "============================================================"
echo ""
echo "⚠️  You need to re-run your training pipeline to regenerate"
echo "    item embeddings in models/classifier/.cache/"
echo ""
echo "Suggested steps:"
echo "  1. Backup models/classifier/.cache/ to .cache_backup/"
echo "  2. Delete or rename models/classifier/.cache/"
echo "  3. Run your training command to regenerate embeddings"
echo ""
echo "Your training command might look like:"
echo "  cd symbiotic-ai"
echo "  python3 -m symbiote_weak_generalized.pipelines.video_training \\"
echo "    --videos-dir hmm-testing/picklist_videos \\"
echo "    --labels-dir hmm-testing/picklist_labels \\"
echo "    --output-dir ../models/classifier \\"
echo "    [your other flags...]"
echo ""
read -p "Have you regenerated item embeddings? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Please regenerate item embeddings, then run:"
    echo "  ./regenerate_embeddings.sh"
    echo ""
    echo "Or manually run steps 3-4 below."
    exit 0
fi

echo ""
echo "============================================================"
echo "STEP 3: Re-running embedding analysis"
echo "============================================================"
echo ""

# Backup old analysis
if [ -d "embedding_analysis_out_50" ]; then
    echo "Backing up old analysis..."
    BACKUP_DIR="embedding_analysis_out_50_OLD_$(date +%Y%m%d_%H%M%S)"
    mv embedding_analysis_out_50 "$BACKUP_DIR"
    echo "  Old analysis saved to: $BACKUP_DIR"
fi

echo "Running embedding analysis..."
python3 -m embedding_analysis \
  --models-root models/classifier \
  --manual-labels symbiotic-ai/hmm-testing/picklist_labels \
  --hand-neutralize 50

echo ""
echo "✓ Embedding analysis complete"
echo "  Check: embedding_analysis_out_50/"
echo ""

echo "============================================================"
echo "STEP 4: Re-running frame similarity visualizer"
echo "============================================================"
echo ""

# Backup old visualization
if [ -d "frame_similarity_out_50" ]; then
    echo "Backing up old visualization..."
    BACKUP_DIR="frame_similarity_out_50_OLD_$(date +%Y%m%d_%H%M%S)"
    mv frame_similarity_out_50 "$BACKUP_DIR"
    echo "  Old visualization saved to: $BACKUP_DIR"
fi

echo "Running frame similarity visualizer..."
./run_frame_similarity.sh 20 50

echo ""
echo "✓ Frame similarity visualizer complete"
echo "  Check: frame_similarity_out_50/"
echo ""

echo "============================================================"
echo "ALL DONE!"
echo "============================================================"
echo ""
echo "Results:"
echo "  - Empty-hand embeddings: symbiotic-ai/hmm-testing/hand_embeddings/"
echo "  - Item embeddings: models/classifier/.cache/"
echo "  - Analysis results: embedding_analysis_out_50/"
echo "  - Similarity visualizations: frame_similarity_out_50/"
echo ""
echo "Next steps:"
echo "  1. Compare new heatmaps with old backups"
echo "  2. Verify same-item similarity is now much higher"
echo "  3. Verify frame crops show only right hands"
echo "  4. If satisfied, delete backup directories"
echo ""
