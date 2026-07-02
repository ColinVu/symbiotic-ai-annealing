STRAIGHTFORWARD:

Command to run the "original" model to train a video based on a label

python -m symbiote_weak.cli.main train --video ./videos/5293.mov --label '["x","y","z"]' --threshold 10 --frame-skip 4 --pca-dim 64 --ilr-epochs 500 --output-dir ../models/classifier

Command to run the HMM model

python -m symbiote.hmm_train \
    --video-dir ../hmm-testing/picklist_videos \
    --label-dir ../hmm-testing/picklist_labels \
    --output-dir ../models/htk \
    --aruco-config ../config/aruco_bins.json \
    --pipeline two-stage




DOCKER:

cd /Users/colinhvu/Documents/coding/symbai/022026/symbiotic-ai
docker build --platform=linux/arm64 --progress=plain -t symbiotic-ai:latest .

docker run --platform=linux/arm64 --rm symbiotic-ai:latest HCompV -h

docker-compose run --rm symbiotic-ai python -m symbiote_weak.cli.main train \
    --video /data/videos/video.mp4 \
    --label '["x","y","z"]' \
    --pca-dim 64 \
    --ilr-epochs 500

docker-compose run --rm symbiotic-ai python -m symbiote_weak.cli.main infer \
    --video /data/videos/video.mp4 \
    --model-dir /models/classifier/video \
    --output /outputs/results.csv



TRAIN HMM ONLY:
docker-compose run --rm symbiotic-ai python -m symbiote.hmm_train \
    --video-dir /data/hmm-testing/picklist_videos \
    --label-dir /data/hmm-testing/picklist_labels \
    --output-dir /models/htk \
    --aruco-config /data/aruco_config/aruco_bins.json \
    --pipeline two-stage

TRAIN + TUNE IN ONE STEP (recommended for old):
docker-compose run --rm symbiotic-ai python -m symbiote.hmm_train \
    --video-dir /data/hmm-testing/picklist_videos \
    --label-dir /data/hmm-testing/picklist_labels \
    --output-dir /models/htk \
    --aruco-config /data/aruco_config/aruco_bins.json \
    --pipeline two-stage \
    --tune-decode

TRAIN + TUNE WITH TOP-K FEATURE MASK (from existing feature report):
docker-compose run --rm symbiotic-ai python -m symbiote.hmm_train \
    --video-dir /data/hmm-testing/picklist_videos \
    --label-dir /data/hmm-testing/picklist_labels \
    --output-dir /models/htk \
    --aruco-config /data/aruco_config/aruco_bins.json \
    --feature-top-k 8 \
    --pipeline two-stage \
    --tune-decode

TRAIN + TUNE WITH STAGE-SPECIFIC TOP-K MASKS (recommended for true two-stage):
docker-compose run --rm symbiotic-ai python -m symbiote.hmm_train \
    --video-dir /data/hmm-testing/picklist_videos \
    --label-dir /data/hmm-testing/picklist_labels \
    --output-dir /models/htk \
    --aruco-config /data/aruco_config/aruco_bins.json \
    --coarse-feature-top-k 8 \
    --interact-feature-top-k 6 \
    --carry-feature-top-k 6 \
    --pipeline two-stage \
    --tune-decode

TRAIN + TUNE (LEGACY SINGLE-STAGE PIPELINE):
docker-compose run --rm symbiotic-ai python -m symbiote.hmm_train \
    --video-dir /data/hmm-testing/picklist_videos \
    --label-dir /data/hmm-testing/picklist_labels \
    --output-dir /models/htk \
    --aruco-config /data/aruco_config/aruco_bins.json \
    --legacy \
    --tune-decode

CLEAR HMM FEATURE CACHE:
rm -rf ./models/htk/feature_cache

TUNE HMM DECODE PARAMS (dev-set sweep for HVite -p/-s):
docker-compose run --rm symbiotic-ai python -m symbiote.hmm_tune \
    --video-dir /data/hmm-testing/picklist_videos \
    --label-dir /data/hmm-testing/picklist_labels \
    --model-dir /models/htk \
    --aruco-config /data/aruco_config/aruco_bins.json \
    --pipeline two-stage

FEATURE RELIABILITY REPORT (per-feature signal strength):
docker-compose run --rm symbiotic-ai python -m symbiote.hmm_feature_report \
    --video-dir /data/hmm-testing/picklist_videos \
    --label-dir /data/hmm-testing/picklist_labels \
    --output-dir /models/htk \
    --aruco-config /data/aruco_config/aruco_bins.json \
    --task-mode coarse \
    --suggest-top-k 8

FEATURE RELIABILITY REPORT (PICK vs PLACE stage):
docker-compose run --rm symbiotic-ai python -m symbiote.hmm_feature_report \
    --video-dir /data/hmm-testing/picklist_videos \
    --label-dir /data/hmm-testing/picklist_labels \
    --output-dir /models/htk \
    --aruco-config /data/aruco_config/aruco_bins.json \
    --task-mode interact \
    --suggest-top-k 6

FEATURE RELIABILITY REPORT (CARRY_WITH vs CARRY_EMPTY stage):
docker-compose run --rm symbiotic-ai python -m symbiote.hmm_feature_report \
    --video-dir /data/hmm-testing/picklist_videos \
    --label-dir /data/hmm-testing/picklist_labels \
    --output-dir /models/htk \
    --aruco-config /data/aruco_config/aruco_bins.json \
    --task-mode carry \
    --suggest-top-k 6

# hmm_infer auto-loads /models/htk/models/hmm_final/infer_params.json if present

INFER HMM ONLY:
docker-compose run --rm symbiotic-ai python -m symbiote.hmm_infer \
    --video /data/videos/picklist_105.MP4 \
    --model-dir /models/htk \
    --output-csv /outputs/predicted_states.csv \
    --output-video /outputs/picklist_105_annotated.mp4 \
    --aruco-config /data/aruco_config/aruco_bins.json \
    --pipeline two-stage

INFER HMM (LEGACY SINGLE-STAGE):
docker-compose run --rm symbiotic-ai python -m symbiote.hmm_infer \
    --video /data/videos/picklist_105.MP4 \
    --model-dir /models/htk \
    --output-csv /outputs/predicted_states_legacy.csv \
    --output-video /outputs/picklist_105_annotated_legacy.mp4 \
    --aruco-config /data/aruco_config/aruco_bins.json \
    --legacy

GROUND-TRUTH OVERLAY QA (label timing visual check):
docker-compose run --rm symbiotic-ai python -m symbiote.hmm_gt_overlay \
    --video-dir /data/hmm-testing/picklist_videos \
    --label-dir /data/hmm-testing/picklist_labels \
    --output-dir /outputs/gt_overlay






docker compose build --no-cache symbiotic-ai
docker compose run --rm symbiotic-ai python -m symbiote_weak.cli.main --help
docker compose run --rm symbiotic-ai python -m symbiote_weak.cli.main train --video /data/videos/5293.mov --label '["x","y","z"]' --threshold 10 --frame-skip 4 --pca-dim 64 --ilr-epochs 500 --output-dir /models/classifier

SYMBIOTE_WEAK OLD-STYLE CONSTRAINED HMM (boundary-first):
docker-compose run --rm symbiotic-ai python -m symbiote_weak.hmm_train \
    --video-dir /data/hmm-testing/picklist_videos \
    --label-dir /data/hmm-testing/picklist_labels \
    --output-dir /models/htk_weak \
    --aruco-config /data/aruco_config/aruco_bins.json \
    --pipeline two-stage \
    --aruco-persistence-frames 30 \
    --aruco-smoothing-window 9 \
    --min-segment-seconds 0.15 \
    --tune-decode

SYMBIOTE_WEAK TUNE (boundary RMSE primary):
docker-compose run --rm symbiotic-ai python -m symbiote_weak.hmm_tune \
    --video-dir /data/hmm-testing/picklist_videos \
    --label-dir /data/hmm-testing/picklist_labels \
    --model-dir /models/htk_weak \
    --aruco-config /data/aruco_config/aruco_bins.json \
    --pipeline two-stage \
    --aruco-persistence-frames 30 \
    --aruco-smoothing-window 9 \
    --min-segment-seconds 0.15

SYMBIOTE_WEAK INFER (constrained):
docker-compose run --rm symbiotic-ai python -m symbiote_weak.hmm_infer \
    --video /data/videos/picklist_105.MP4 \
    --model-dir /models/htk_weak \
    --output-csv /outputs/predicted_states_weak.csv \
    --output-video /outputs/picklist_105_annotated_weak.mp4 \
    --aruco-config /data/aruco_config/aruco_bins.json \
    --pipeline two-stage \
    --sequence-label-csv /data/hmm-testing/picklist_labels/picklist_105.csv \
    --aruco-persistence-frames 30 \
    --aruco-smoothing-window 9 \
    --min-segment-seconds 0.15

SYMBIOTE_WEAK BOUNDARY RMSE REPORT:
docker-compose run --rm symbiotic-ai python -m symbiote_weak.hmm_boundary_eval \
    --pred-dir /outputs \
    --label-dir /data/hmm-testing/picklist_labels \
    --output-csv /outputs/boundary_rmse_summary.csv

SYMBIOTE_WEAK OBJECT INFERENCE (manual labels):
docker-compose run --rm symbiotic-ai python -m symbiote_weak.cli.main train \
    --video /data/hmm-testing/picklist_videos/picklist_1.MP4 \
    --label '["apple","banana","chocolate","..."]' \
    --manual-labels-dir /data/hmm-testing/picklist_labels \
    --threshold 10 \
    --frame-skip 4 \
    --pca-dim 64 \
    --ilr-epochs 500 \
    --output-dir /models/classifier_weak



SYMBIOTE_WEAK TRAIN NEW:
docker-compose run --rm symbiotic-ai python -m symbiote_weak.hmm_train \
  --video-dir /data/hmm-testing/picklist_videos \
  --label-dir /data/hmm-testing/picklist_labels \
  --output-dir /models/htk_weak \
  --aruco-config /data/aruco_config/aruco_bins.json \
  --pipeline two-stage \
  --coarse-feature-mask 2,3,4,5,6,7,10,11,12 \
  --interact-feature-mask 0,1,3,10,11,14 \
  --carry-feature-mask 0,1,2,3,6,7,10,11,12 \
  --threshold 50 \
  --tune-decode

+ TUNE
docker-compose run --rm symbiotic-ai python -m symbiote_weak.hmm_tune \
  --video-dir /data/hmm-testing/picklist_videos \
  --label-dir /data/hmm-testing/picklist_labels \
  --model-dir /models/htk_weak \
  --aruco-config /data/aruco_config/aruco_bins.json \
  --pipeline two-stage \
  --threshold 50

rm -rf ./models/htk_weak/feature_cache






docker-compose run --rm symbiotic-ai python -m symbiote_weak.hmm_train \
    --video-dir /data/hmm-testing/picklist_videos \
    --label-dir /data/hmm-testing/picklist_labels \
    --output-dir /models/htk_weak_cleaned \
    --aruco-config /data/aruco_config/aruco_bins.json \
    --pipeline two-stage \
    --aruco-persistence-frames 30 \
    --aruco-smoothing-window 9 \
    --min-segment-seconds 0.15 \
    --no-sequence-constraint \
    --feature-mask 0-12,14-24,26-28 \
    --tune-decode



    docker-compose run --rm symbiotic-ai python -m symbiote_weak.hmm_train \
    --video-dir /data/hmm-testing/picklist_videos \
    --label-dir /data/hmm-testing/picklist_labels \
    --output-dir /models/htk_weak_skip2 \
    --aruco-config /data/aruco_config/aruco_bins.json \
    --pipeline two-stage \
    --aruco-persistence-frames 60 \
    --aruco-smoothing-window 17 \
    --frame-skip 2 \
    --min-segment-seconds 0.15 \
    --no-sequence-constraint \
    --tune-decode \
    --feature-mask 0,1,2,3,4,5,6,7,8,9,10,11,12








    python3 -m symbiote_weak.cli.main train \
  --video ./hmm-testing/picklist_videos/picklist_091.mp4 \
  --video-config ./hmm-testing/picklist_jsons/picklist_091.json \
  --manual-labels-dir ./hmm-testing/picklist_labels \
  --compact-frame-indexing opencv0 \
  --output-dir ../models/classifier


  # From symbiotic-ai/ — after training saved a model under MODEL_DIR
python3 -m symbiote_weak.cli.main incremental \
  --video ./hmm-testing/picklist_videos/picklist_101.MP4 \
  --picklist-json-dir ./hmm-testing/picklist_jsons \
  --manual-labels-dir ./hmm-testing/picklist_labels \
  --model-dir ../models/classifier \
  --beta 0.9


  python3 scripts/evaluate_picklist_video.py \
  --video ./hmm-testing/picklist_videos/picklist_101.MP4 \
  --model-dir ../models/classifier \
  --state-label-csv ./hmm-testing/picklist_labels/picklist_101.csv \
  --frame-skip 3 \
  --threshold 50 \
  --compact-frame-indexing opencv0 

  python3 scripts/repair_picklist_state_csv.py hmm-testing/templ/ --batch                                         


python3 -m symbiote_weak.cli.main incremental \
  --video ./hmm-testing/picklist_videos/picklist_101.MP4 \
  --picklist-json-dir ./hmm-testing/picklist_jsons \
  --manual-labels-dir ./hmm-testing/picklist_labels \
  --model-dir ../models/classifier \
  --equal-video-weight

  --threshold 30 \
  --ilr-epochs 1000 \
  --random-seed 123 \
  --force-reembed




  # 1. Delete current model
rm -rf ../models/classifier

# 2. Train on 091 first (your initial video)
python3 -m symbiote_weak.cli.main train \
  --video ./hmm-testing/picklist_videos/picklist_091.mp4 \
  --video-config ./hmm-testing/picklist_jsons/picklist_091.json \
  --manual-labels-dir ./hmm-testing/picklist_labels \
  --compact-frame-indexing opencv0 \
  --output-dir ../models/classifier \
  --pca-dim 128

# 3. Add the "easier" videos first (101, 051, 071)
for vid in 101 051 061 071 021 011; do
  python3 -m symbiote_weak.cli.main incremental \
    --video ./hmm-testing/picklist_videos/picklist_${vid}.MP4 \
    --picklist-json-dir ./hmm-testing/picklist_jsons \
    --manual-labels-dir ./hmm-testing/picklist_labels \
    --compact-frame-indexing opencv0 \
    --model-dir ../models/classifier \
    --threshold 30 \
    --equal-video-weight
done

# 4. Add 061 LAST after model is better established
python3 -m symbiote_weak.cli.main incremental \
  --video ./hmm-testing/picklist_videos/picklist_061.MP4 \
  --picklist-json-dir ./hmm-testing/picklist_jsons \
  --manual-labels-dir ./hmm-testing/picklist_labels \
  --compact-frame-indexing opencv0 \
  --model-dir ../models/classifier \
  --equal-video-weight \
  --ilr-epochs 1000



  python3 -m symbiote_weak.cli.main infer \
  --model-dir ../models/classifier \
  --video ./hmm-testing/picklist_videos/picklist_061.MP4 \
  --output ./eval_061.csv \
  --picklist-json ./hmm-testing/picklist_jsons/picklist_061.json \
  --manual-labels-dir ./hmm-testing/picklist_labels \
  --threshold 30 \
  --apply-iterated-postprocess



  python3 -m symbiote_weak.cli.main train-from-cache \
  --videos ./hmm-testing/picklist_videos \
  --picklist-json-dir ./hmm-testing/picklist_jsons \
  --manual-labels-dir ./hmm-testing/picklist_labels \
  --compact-frame-indexing opencv0 \
  --output-dir ../models/classifier \
  --frame-skip 4 \
  --ilr-epochs 1000 \
  --iterated-model --sa-iters 100 --adapter-epochs 10 --refinement-loops 3


python3 -m symbiote_weak.cli.main sweep \
  --videos ./hmm-testing/videos \
  --picklist-json-dir ./hmm-testing/picklist_jsons \
  --manual-labels-dir ./hmm-testing/manual_labels \
  --ground-truth-csv ./ground_truth.csv \
  --cache-dir ../models/classifier/.cache \
  --output-dir ../experiments/sweep_001 \
  --search-type random \
  --num-samples 30 \
  --frame-skip 4 \
  --compact-frame-indexing opencv0

python3 -m symbiote_weak_generalized.scripts.synthetic_ilr_itemx_ablation

python3 -m experiments.confusion_analysis.cli \                 
  --run-dir experiments/sweep_001/run_0000_20260513_052858 \
  --output-dir experiments/sweep_001/run_0000_20260513_052858/confusion_analysis_out



adapter-epochs	Adapter passes per inner training block	10	↑ if loss still dropping; ↓ if overfitting / slow
adapter-lr	Adam step size	1e-3	↓ if unstable; ↑ slightly if loss flat
adapter-batch-size	Triplet minibatch	32	↓ on OOM
triplet-margin	Required separation in adapted L2 space	0.1	↑ to push harder; ↓ if too harsh
ilr_epochs	SA / refine_labels length	your 1000 default	main SA quality/time tradeoff
sa_iters
refinement-loops



  python3 -m symbiote_weak_generalized.cli.main train \           
  --videos ./hmm-testing/picklist_videos \
  --picklist-json-dir ./hmm-testing/picklist_jsons \
  --manual-labels-dir ./hmm-testing/picklist_labels \
  --compact-frame-indexing opencv0 \
  --output-dir ../models/classifier \
  --frame-skip 5 \
  --ilr-epochs 700