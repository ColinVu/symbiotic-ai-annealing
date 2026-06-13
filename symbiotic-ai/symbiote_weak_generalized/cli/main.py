"""Command-line interface for the weakly supervised video-to-classification pipeline."""

import argparse
import sys
import os
import json
from datetime import datetime

from ..core.config import DEFAULT_CONFIG
from ..pipelines.video_training import (
    run_video_training,
    run_multi_video_training,
    run_multi_video_training_from_cache,
    run_incremental_training,
    _list_videos_in_folder,
)
from ..pipelines.video_inference import run_video_inference
from ..inference.recognizer import ObjectRecognizer


def _merge_iterated_model_config(script_dir: str, args, config: dict) -> None:
    """Populate *config* with iterated-model options from CLI (symbiotic-ai-relative paths)."""
    im = bool(getattr(args, "iterated_model", False))
    config["use_iterated_model"] = im
    if not im:
        return
    he = getattr(args, "hand_embeddings_dir", None) or DEFAULT_CONFIG.get("hand_embeddings_dir")
    if he and not os.path.isabs(str(he)):
        he = os.path.normpath(os.path.join(script_dir, str(he)))
    config["hand_embeddings_dir"] = str(he) if he else ""
    config["sa_iters"] = int(getattr(args, "sa_iters", DEFAULT_CONFIG["sa_iters"]))
    config["adapter_epochs"] = int(getattr(args, "adapter_epochs", DEFAULT_CONFIG["adapter_epochs"]))
    config["adapter_lr"] = float(getattr(args, "adapter_lr", DEFAULT_CONFIG["adapter_lr"]))
    config["adapter_batch_size"] = int(
        getattr(args, "adapter_batch_size", DEFAULT_CONFIG["adapter_batch_size"])
    )
    config["refinement_loops"] = int(
        getattr(args, "refinement_loops", DEFAULT_CONFIG["refinement_loops"])
    )
    config["triplet_margin"] = float(getattr(args, "triplet_margin", DEFAULT_CONFIG["triplet_margin"]))
    config["proxy_energy_margin"] = float(
        getattr(args, "proxy_energy_margin", DEFAULT_CONFIG["proxy_energy_margin"])
    )


def main():
    parser = argparse.ArgumentParser(
        description="Weakly Supervised Video-to-Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with picklist (weakly supervised)
  python -m symbiote_weak.cli.main train --video ../videos/demo.mp4 --label '["apple", "banana", "apple"]'
  
  # Train with custom ILR parameters
  python -m symbiote_weak.cli.main train --video ../videos/demo.mp4 --label '["a", "b", "c"]' --ilr-epochs 500
  
  # Predict on a single image
  python -m symbiote_weak.cli.main predict --model-dir ../models/classifier/video_name --image ../images/test.jpg
  
  # Get top-3 predictions
  python -m symbiote_weak.cli.main predict --model-dir ../models/classifier/video_name --image ../images/test.jpg --top-k 3
  
  # Run inference on video and output CSV
  python -m symbiote_weak.cli.main infer --video ../videos/test.mp4 --model-dir ../models/classifier/video_name --output results.csv

  # Incremental centroid update (fit_iterative) on an existing model directory
  python -m symbiote_weak.cli.main incremental --video ../videos/new.mp4 --label '["a","b"]' --model-dir ../models/classifier/video_name --beta 0.9

  # Hyperparameter sweep (iterated ILR + adapter knobs; rank by final_assignments.csv hit %)
  python -m symbiote_weak.cli.main sweep --videos ./hmm-testing/videos \\
    --picklist-json-dir ./hmm-testing/picklist_jsons --manual-labels-dir ./hmm-testing/manual_labels \\
    --ground-truth-csv ./ground_truth.csv --search-type random --num-samples 10
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    train_parser = subparsers.add_parser("train", help="Train a classifier from video using weak supervision")
    train_video_src = train_parser.add_mutually_exclusive_group(required=True)
    train_video_src.add_argument(
        "--video",
        type=str,
        help="Path to a single video file to process",
    )
    train_video_src.add_argument(
        "--videos",
        type=str,
        help="Directory of .mp4/.MP4 files: joint train on all (requires --picklist-json-dir and --manual-labels-dir)",
    )
    train_parser.add_argument(
        "--label",
        type=str,
        required=False,
        default=None,
        help='Picklist as JSON array, or omit when using --video-config (can use \'[]\')',
    )
    train_parser.add_argument(
        "--video-config-path",
        "--video-config",
        type=str,
        default=None,
        help="Path to JSON with id + picklists (nested arrays), e.g. example.json (single --video only)",
    )
    train_parser.add_argument(
        "--picklist-json-dir",
        type=str,
        default=None,
        help="With --videos: directory containing picklist_{video_stem}.json for each video",
    )
    train_parser.add_argument(
        "--compact-frame-indexing",
        type=str,
        default="opencv0",
        choices=["opencv0", "pipeline1"],
        help="Frame numbering in compact state CSVs (default opencv0)",
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default="../models/classifier",
        help="Base directory to save model and results (subfolder will be created for video)"
    )
    train_parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Blur detection threshold (Laplacian variance, default 50.0)"
    )
    train_parser.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="Process every Nth frame (default 4)"
    )
    train_parser.add_argument(
        "--ilr-epochs",
        type=int,
        default=DEFAULT_CONFIG["ilr_epochs"],
        help="Number of Iterative Label Refinement epochs (default 500)"
    )
    train_parser.add_argument(
        "--no-annealing",
        action="store_true",
        help="Skip ILR swap/annealing; keep initial labels only (cluster voting if --cluster-voting, else random)",
    )
    train_parser.add_argument(
        "--cluster-voting",
        action="store_true",
        help="Use global K-means cluster voting for initial labels; default is random bijection per picklist multiset",
    )
    train_parser.add_argument(
        "--ilr-allow-cross-round-swaps",
        action="store_true",
        help="ILR may swap labels between carry segments from different picklist rounds "
        "in the same video, if each post-swap label is still in that segment's candidate multiset. "
        "Default: same picklist round only (omit this flag).",
    )
    train_parser.add_argument(
        "--initial-temp",
        type=float,
        default=DEFAULT_CONFIG["initial_temp"],
        help="Initial temperature for simulated annealing (default 1.0)"
    )
    train_parser.add_argument(
        "--temp-decay",
        type=str,
        default=DEFAULT_CONFIG["temp_decay"],
        choices=["exponential", "linear", "cosine"],
        help="Temperature decay schedule (default exponential)",
    )
    train_parser.add_argument(
        "--min-temp",
        type=float,
        default=DEFAULT_CONFIG.get("min_temp", 0.05),
        help="Floor temperature for cosine / annealing schedules (default: 0.05)",
    )
    train_parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Show detailed progress"
    )
    train_parser.add_argument(
        "--htk-model-dir",
        type=str,
        default=None,
        help="Path to trained HTK HMM model directory (for state detection)"
    )
    train_parser.add_argument(
        "--aruco-config",
        type=str,
        default=None,
        help="Path to ARUCO marker configuration JSON"
    )
    train_parser.add_argument(
        "--manual-labels-dir",
        type=str,
        default=None,
        help="Optional directory containing manual state CSVs (timestamp_start,timestamp_end,state) named after video stem"
    )
    train_parser.add_argument(
        "--variance-eps",
        type=float,
        default=DEFAULT_CONFIG["variance_eps"],
        help="Epsilon for variance-weighted ILR denominator (default from config)",
    )
    train_parser.add_argument(
        "--bad-swap-cool-divisor",
        type=float,
        default=DEFAULT_CONFIG["bad_swap_cool_divisor"],
        help="Epoch divisor for e^(-epoch/div) bad-swap probability (HSV-style ILR, default 50)",
    )
    train_parser.add_argument(
        "--min-frames-per-cluster",
        type=int,
        default=DEFAULT_CONFIG["min_frames_per_cluster"],
        help="Minimum frames per item/empty cluster when splitting carry segments (default 3)",
    )
    train_parser.add_argument(
        "--iterated-model",
        action="store_true",
        help="Replace standard ILR with iterated neutralize → proxy-triplet SA → CLIP adapter loop",
    )
    train_parser.add_argument(
        "--hand-embeddings-dir",
        type=str,
        default=None,
        help="Directory of .npy empty-hand CLIP embeddings for PCA neutralizer (default: hmm-testing/hand_embeddings)",
    )
    train_parser.add_argument(
        "--sa-iters",
        type=int,
        default=DEFAULT_CONFIG["sa_iters"],
        help="Simulated annealing epochs per refinement loop (iterated model)",
    )
    train_parser.add_argument(
        "--adapter-epochs",
        type=int,
        default=DEFAULT_CONFIG["adapter_epochs"],
        help="CLIPAdapter training epochs per refinement loop",
    )
    train_parser.add_argument(
        "--adapter-lr",
        type=float,
        default=DEFAULT_CONFIG["adapter_lr"],
        help="CLIPAdapter Adam learning rate",
    )
    train_parser.add_argument(
        "--adapter-batch-size",
        type=int,
        default=DEFAULT_CONFIG["adapter_batch_size"],
        help="Batch size for CLIPAdapter triplet training",
    )
    train_parser.add_argument(
        "--refinement-loops",
        type=int,
        default=DEFAULT_CONFIG["refinement_loops"],
        help="Outer refinement iterations (neutralize → SA → adapter → update)",
    )
    train_parser.add_argument(
        "--triplet-margin",
        type=float,
        default=DEFAULT_CONFIG["triplet_margin"],
        help="TripletMarginLoss margin for CLIPAdapter training",
    )
    train_parser.add_argument(
        "--proxy-energy-margin",
        type=float,
        default=DEFAULT_CONFIG["proxy_energy_margin"],
        help="Margin m in proxy energy max(d_p - d_n + m, 0) for SA",
    )

    def _add_train_from_cache_parser(sub, name: str, help_text: str):
        p = sub.add_parser(
            name,
            help=help_text,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        p.add_argument(
            "--videos",
            type=str,
            required=True,
            help="Directory of videos (same as train --videos); used for stems and FPS/frame-count metadata only",
        )
        p.add_argument(
            "--picklist-json-dir",
            type=str,
            required=True,
            help="Directory with {stem}.json picklists (same as train)",
        )
        p.add_argument(
            "--manual-labels-dir",
            type=str,
            required=True,
            help="Directory with {stem}.csv manual state labels (same as train)",
        )
        p.add_argument(
            "--compact-frame-indexing",
            type=str,
            default="opencv0",
            choices=["opencv0", "pipeline1"],
            help="Frame numbering in compact state CSVs (must match the run that wrote the cache)",
        )
        p.add_argument(
            "--output-dir",
            type=str,
            default="../models/classifier",
            help="Where to save the new model (and default cache root parent)",
        )
        p.add_argument(
            "--cache-dir",
            type=str,
            default=None,
            help="Root of per-video caches (default: <output-dir>/.cache/{stem}/). Use if cache lives elsewhere",
        )
        p.add_argument(
            "--frame-skip",
            type=int,
            default=4,
            help="Must match the frame_skip used when embeddings were written to cache",
        )
        p.add_argument(
            "--ilr-epochs",
            type=int,
            default=DEFAULT_CONFIG["ilr_epochs"],
            help="ILR epochs (default 500)",
        )
        p.add_argument(
            "--no-annealing",
            action="store_true",
            help="Skip ILR swap/annealing; keep initial labels only (cluster voting if --cluster-voting, else random)",
        )
        p.add_argument(
            "--cluster-voting",
            action="store_true",
            help="Use global K-means cluster voting for initial labels; default is random bijection per picklist multiset",
        )
        p.add_argument(
            "--ilr-allow-cross-round-swaps",
            action="store_true",
            help="ILR may swap labels between carry segments from different picklist rounds "
            "in the same video, if each post-swap label is still in that segment's candidate multiset. "
            "Default: same picklist round only.",
        )
        p.add_argument(
            "--initial-temp",
            type=float,
            default=DEFAULT_CONFIG["initial_temp"],
            help="Initial temperature for simulated annealing (default 1.0)",
        )
        p.add_argument(
            "--temp-decay",
            type=str,
            default=DEFAULT_CONFIG["temp_decay"],
            choices=["exponential", "linear", "cosine"],
            help="Temperature decay schedule (default exponential)",
        )
        p.add_argument(
            "--verbose",
            action="store_true",
            default=True,
            help="Show detailed progress",
        )
        p.add_argument(
            "--variance-eps",
            type=float,
            default=DEFAULT_CONFIG["variance_eps"],
            help="Epsilon for variance-weighted ILR denominator",
        )
        p.add_argument(
            "--bad-swap-cool-divisor",
            type=float,
            default=DEFAULT_CONFIG["bad_swap_cool_divisor"],
            help="Epoch divisor for bad-swap cooling (HSV-style ILR)",
        )
        p.add_argument(
            "--min-frames-per-cluster",
            type=int,
            default=DEFAULT_CONFIG["min_frames_per_cluster"],
            help="Minimum frames per item/empty cluster when splitting carry segments",
        )
        p.add_argument(
            "--iterated-model",
            action="store_true",
            help="Replace standard ILR with iterated neutralize → proxy-triplet SA → CLIP adapter loop",
        )
        p.add_argument(
            "--hand-embeddings-dir",
            type=str,
            default=None,
            help="Directory of .npy empty-hand embeddings for PCA neutralizer",
        )
        p.add_argument(
            "--apply-hand-pca",
            action="store_true",
            help="Apply HandNeutralizer PCA preprocessing (20 components removed) to all embeddings BEFORE annealing. Requires --hand-embeddings-dir",
        )
        p.add_argument(
            "--n-components",
            type=int,
            default=DEFAULT_CONFIG["n_components"],
            help="Number of PCA components to remove (used with --apply-hand-pca)",
        )
        p.add_argument(
            "--sa-iters",
            type=int,
            default=DEFAULT_CONFIG["sa_iters"],
            help="SA epochs per refinement loop (iterated model)",
        )
        p.add_argument(
            "--adapter-epochs",
            type=int,
            default=DEFAULT_CONFIG["adapter_epochs"],
            help="CLIPAdapter training epochs per refinement loop",
        )
        p.add_argument(
            "--adapter-lr",
            type=float,
            default=DEFAULT_CONFIG["adapter_lr"],
            help="CLIPAdapter Adam learning rate",
        )
        p.add_argument(
            "--adapter-batch-size",
            type=int,
            default=DEFAULT_CONFIG["adapter_batch_size"],
            help="Batch size for CLIPAdapter triplet training",
        )
        p.add_argument(
            "--refinement-loops",
            type=int,
            default=DEFAULT_CONFIG["refinement_loops"],
            help="Outer refinement iterations",
        )
        p.add_argument(
            "--triplet-margin",
            type=float,
            default=DEFAULT_CONFIG["triplet_margin"],
            help="TripletMarginLoss margin",
        )
        p.add_argument(
            "--proxy-energy-margin",
            type=float,
            default=DEFAULT_CONFIG["proxy_energy_margin"],
            help="Margin in proxy-triplet SA energy max(d_p - d_n + m, 0)",
        )
        p.add_argument(
            "--min-temp",
            type=float,
            default=DEFAULT_CONFIG.get("min_temp", 0.05),
            help="Floor temperature for cosine / annealing schedules (default: 0.05)",
        )
        return p

    _add_train_from_cache_parser(
        subparsers,
        "train-from-cache",
        "Multi-video weak supervision using only cached .npy embeddings (no CLIP / no frame decode for embedding)",
    )
    _add_train_from_cache_parser(
        subparsers,
        "train_no_cache",
        "Same as train-from-cache (retrain from disk cache only; no CLIP pass)",
    )

    inc_parser = subparsers.add_parser(
        "incremental",
        help="Update saved centroids from a new video (fit_iterative + EWMA)",
    )
    inc_parser.add_argument("--video", type=str, required=True, help="Path to new video")
    inc_parser.add_argument(
        "--label",
        type=str,
        default=None,
        help='Optional. Picklist JSON array (one string per carry segment). Omit when using '
        "--video-config-path or --picklist-json-dir",
    )
    inc_parser.add_argument(
        "--video-config-path",
        "--video-config",
        type=str,
        default=None,
        help="JSON with picklists (same as train). Overrides --picklist-json-dir if both set.",
    )
    inc_parser.add_argument(
        "--picklist-json-dir",
        type=str,
        default=None,
        help="Directory containing picklist_{video_stem}.json (e.g. hmm-testing/picklist_jsons). "
        "Used when --label is omitted.",
    )
    inc_parser.add_argument(
        "--force-reembed",
        action="store_true",
        help="Allow incremental on a video stem already listed in model embedded_video_stems",
    )
    inc_parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory with centroids.npy, model_metadata.json from prior train",
    )
    inc_parser.add_argument(
        "--beta",
        type=float,
        default=0.9,
        help="EWMA weight on previous centroid (default 0.9; new observation weight is 1-beta); ignored with --equal-video-weight",
    )
    inc_parser.add_argument(
        "--equal-video-weight",
        action="store_true",
        help="After the best permutation, store one spherical mean per (label, video) and set each centroid to the spherical mean across videos (beta ignored)",
    )
    inc_parser.add_argument("--threshold", type=float, default=100.0)
    inc_parser.add_argument("--frame-skip", type=int, default=4)
    inc_parser.add_argument("--verbose", action="store_true", default=True)
    inc_parser.add_argument("--htk-model-dir", type=str, default=None)
    inc_parser.add_argument("--aruco-config", type=str, default=None)
    inc_parser.add_argument("--manual-labels-dir", type=str, default=None)
    inc_parser.add_argument(
        "--compact-frame-indexing",
        type=str,
        default="opencv0",
        choices=["opencv0", "pipeline1"],
    )
    inc_parser.add_argument(
        "--ilr-epochs",
        type=int,
        default=None,
        help="Override ILR epochs from saved model config (useful for re-embedding with --force-reembed)",
    )
    inc_parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Override random seed from saved model config (useful for re-embedding with --force-reembed)",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict on a single image")
    predict_parser.add_argument(
        "--model-dir",
        type=str,
        default="../models/classifier",
        help="Directory containing trained model"
    )
    predict_parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to image to classify"
    )
    predict_parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of top predictions to show"
    )
    
    infer_parser = subparsers.add_parser("infer", help="Run inference on video and output CSV")
    infer_parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file to process"
    )
    infer_parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing trained model"
    )
    infer_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output CSV file"
    )
    infer_parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Blur detection threshold (Laplacian variance, default 50.0; same as train)",
    )
    infer_parser.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="Process every Nth frame (default 4; same as train)",
    )
    infer_parser.add_argument(
        "--picklist-json",
        type=str,
        default=None,
        help="Path to picklist JSON; restrict predictions to labels in picklists (all rounds)",
    )
    infer_parser.add_argument(
        "--manual-labels-dir",
        type=str,
        default=None,
        help="Directory with {video_stem}.csv manual state labels (same as train). When set, only CARRY_WITH frames are classified.",
    )
    infer_parser.add_argument(
        "--compact-frame-indexing",
        type=str,
        default="opencv0",
        help="Frame indexing for manual CSV (default opencv0; must match training)",
    )
    infer_parser.add_argument(
        "--apply-iterated-postprocess",
        action="store_true",
        default=False,
        help=(
            "Apply iterated-model embedding postprocess during video inference "
            "(hand neutralizer + adapter if saved in model dir)."
        ),
    )
    infer_parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Show detailed progress"
    )

    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Hyperparameter sweep: train-from-cache per config; rank by final_assignments.csv hit rate (needs --ground-truth-csv)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sweep_parser.add_argument(
        "--videos",
        type=str,
        required=True,
        help="Directory of training videos (same as train-from-cache)",
    )
    sweep_parser.add_argument(
        "--picklist-json-dir",
        type=str,
        required=True,
        help="Directory with {stem}.json picklists",
    )
    sweep_parser.add_argument(
        "--manual-labels-dir",
        type=str,
        required=True,
        help="Directory with {stem}.csv manual state labels",
    )
    sweep_parser.add_argument(
        "--ground-truth-csv",
        type=str,
        required=True,
        help="Wide CSV: columns = video stems (e.g. picklist_061), rows = ordered true labels",
    )
    sweep_parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Root of per-video embedding caches (default: <output-dir>/.cache)",
    )
    sweep_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Sweep output root (default: ../experiments/sweeps/sweep_<timestamp>)",
    )
    sweep_parser.add_argument(
        "--search-type",
        type=str,
        default="random",
        choices=["grid", "random"],
        help="Exhaustive grid or random sampling over swept axes (see sweep_config.PARAM_GRID; override with --sweep-*)",
    )
    sweep_parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Random search sample count (ignored for grid)",
    )
    sweep_parser.add_argument(
        "--sweep-random-seed",
        type=int,
        default=42,
        help="RNG seed for random search config generation",
    )
    sweep_parser.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="Frame skip for train-from-cache (must match cache)",
    )
    sweep_parser.add_argument(
        "--compact-frame-indexing",
        type=str,
        default="opencv0",
        choices=["opencv0", "pipeline1"],
        help="Must match the run that wrote caches / manual CSVs",
    )
    sweep_parser.add_argument(
        "--sweep-ilr-epochs",
        type=str,
        default=None,
        help="Comma-separated ilr_epochs (default: sweep_config.PARAM_GRID)",
    )
    sweep_parser.add_argument(
        "--sweep-n-components",
        type=str,
        default=None,
        help="Comma-separated n_components (PCA dims; default: PARAM_GRID)",
    )
    sweep_parser.add_argument(
        "--sweep-triplet-margin",
        type=str,
        default=None,
        help="Comma-separated triplet_margin floats (default: PARAM_GRID)",
    )
    sweep_parser.add_argument(
        "--sweep-refinement-loops",
        type=str,
        default=None,
        help="Comma-separated refinement_loops (default: PARAM_GRID)",
    )
    sweep_parser.add_argument(
        "--sweep-adapter-epochs",
        type=str,
        default=None,
        help="Comma-separated adapter_epochs (default: PARAM_GRID)",
    )
    sweep_parser.add_argument(
        "--sweep-adapter-lr",
        type=str,
        default=None,
        help="Comma-separated adapter_lr floats (default: PARAM_GRID)",
    )
    sweep_parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose training logs per run",
    )

    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Resolve relative paths from the repo root (symbiotic-ai/), not from cli/
    script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    if args.command == "train":
        base_output_dir = os.path.normpath(os.path.join(script_dir, args.output_dir))

        config = DEFAULT_CONFIG.copy()
        config["ilr_epochs"] = args.ilr_epochs
        config["initial_temp"] = args.initial_temp
        config["temp_decay"] = args.temp_decay
        config["min_temp"] = float(getattr(args, "min_temp", DEFAULT_CONFIG.get("min_temp", 0.05)))
        config["variance_eps"] = args.variance_eps
        config["bad_swap_cool_divisor"] = args.bad_swap_cool_divisor
        config["detect_empty"] = False
        config["min_frames_per_cluster"] = args.min_frames_per_cluster
        config["skip_ilr"] = bool(getattr(args, "no_annealing", False))
        config["use_cluster_voting"] = bool(getattr(args, "cluster_voting", False))
        config["ilr_allow_cross_round_swaps"] = bool(
            getattr(args, "ilr_allow_cross_round_swaps", False)
        )
        _merge_iterated_model_config(script_dir, args, config)
        if config.get("skip_ilr"):
            config["use_iterated_model"] = False

        manual_labels_dir = None
        if args.manual_labels_dir:
            manual_labels_dir = args.manual_labels_dir
            if not os.path.isabs(manual_labels_dir):
                manual_labels_dir = os.path.normpath(os.path.join(script_dir, manual_labels_dir))

        htk_model_dir = None
        if args.htk_model_dir:
            htk_model_dir = args.htk_model_dir
            if not os.path.isabs(htk_model_dir):
                htk_model_dir = os.path.normpath(os.path.join(script_dir, htk_model_dir))

        aruco_config = None
        if args.aruco_config:
            aruco_config = args.aruco_config
            if not os.path.isabs(aruco_config):
                aruco_config = os.path.normpath(os.path.join(script_dir, aruco_config))

        if getattr(args, "videos", None):
            if not args.picklist_json_dir:
                print("Error: --picklist-json-dir is required with --videos")
                sys.exit(1)
            if not args.manual_labels_dir:
                print("Error: --manual-labels-dir is required with --videos (one CSV per video stem)")
                sys.exit(1)
            if args.video_config_path:
                print("Error: do not combine --videos with --video-config-path; use per-video JSONs in --picklist-json-dir")
                sys.exit(1)
            if args.label is not None and str(args.label).strip() != "":
                print("Error: do not combine --videos with --label; picklists come from JSON files")
                sys.exit(1)

            videos_dir = args.videos
            if not os.path.isabs(videos_dir):
                videos_dir = os.path.normpath(os.path.join(script_dir, videos_dir))
            picklist_json_dir = args.picklist_json_dir
            if not os.path.isabs(picklist_json_dir):
                picklist_json_dir = os.path.normpath(os.path.join(script_dir, picklist_json_dir))

            run_multi_video_training(
                videos_dir=videos_dir,
                picklist_json_dir=picklist_json_dir,
                manual_labels_dir=manual_labels_dir,
                base_output_dir=base_output_dir,
                config=config,
                threshold=args.threshold,
                frame_skip=args.frame_skip,
                verbose=args.verbose,
                htk_model_dir=htk_model_dir,
                aruco_config_path=aruco_config,
                compact_frame_indexing=args.compact_frame_indexing,
            )
        else:
            video_path = args.video
            if not os.path.isabs(video_path):
                video_path = os.path.normpath(os.path.join(script_dir, video_path))

            if not os.path.exists(video_path):
                print(f"Error: Video file not found: {args.video}")
                sys.exit(1)

            picklist: list = []
            if args.label is not None and str(args.label).strip() != "":
                try:
                    picklist = json.loads(args.label)
                    if not isinstance(picklist, list):
                        raise ValueError("Label must be a JSON array")
                    for item in picklist:
                        if not isinstance(item, str):
                            raise ValueError("All picklist items must be strings")
                except json.JSONDecodeError as e:
                    print(f"Error: Invalid JSON for --label: {e}")
                    print('Expected format: \'["apple", "banana", "apple"]\' or [] with --video-config')
                    sys.exit(1)
                except ValueError as e:
                    print(f"Error: {e}")
                    sys.exit(1)

            if not args.video_config_path and len(picklist) == 0:
                print("Error: provide --label with at least one SKU, or use --video-config-path with picklists")
                sys.exit(1)

            vcfg = args.video_config_path
            if vcfg and not os.path.isabs(vcfg):
                vcfg = os.path.normpath(os.path.join(script_dir, vcfg))

            run_video_training(
                video_path=video_path,
                picklist=picklist,
                base_output_dir=base_output_dir,
                config=config,
                threshold=args.threshold,
                frame_skip=args.frame_skip,
                verbose=args.verbose,
                htk_model_dir=htk_model_dir,
                aruco_config_path=aruco_config,
                manual_labels_dir=manual_labels_dir,
                video_config_path=vcfg,
                compact_frame_indexing=args.compact_frame_indexing,
            )

    elif args.command in ("train-from-cache", "train_no_cache"):
        base_output_dir = os.path.normpath(os.path.join(script_dir, args.output_dir))

        config = DEFAULT_CONFIG.copy()
        config["ilr_epochs"] = args.ilr_epochs
        config["initial_temp"] = args.initial_temp
        config["temp_decay"] = args.temp_decay
        config["min_temp"] = float(getattr(args, "min_temp", DEFAULT_CONFIG.get("min_temp", 0.05)))
        config["variance_eps"] = args.variance_eps
        config["bad_swap_cool_divisor"] = args.bad_swap_cool_divisor
        config["detect_empty"] = False
        config["min_frames_per_cluster"] = args.min_frames_per_cluster
        config["skip_ilr"] = bool(getattr(args, "no_annealing", False))
        config["use_cluster_voting"] = bool(getattr(args, "cluster_voting", False))
        config["ilr_allow_cross_round_swaps"] = bool(
            getattr(args, "ilr_allow_cross_round_swaps", False)
        )
        config["apply_hand_pca"] = bool(getattr(args, "apply_hand_pca", False))
        config["n_components"] = int(getattr(args, "n_components", DEFAULT_CONFIG["n_components"]))
        _merge_iterated_model_config(script_dir, args, config)
        if config.get("skip_ilr"):
            config["use_iterated_model"] = False
        if config.get("apply_hand_pca") and not config.get("hand_embeddings_dir"):
            print("Error: --apply-hand-pca requires --hand-embeddings-dir")
            sys.exit(1)

        manual_labels_dir = args.manual_labels_dir
        if not os.path.isabs(manual_labels_dir):
            manual_labels_dir = os.path.normpath(os.path.join(script_dir, manual_labels_dir))

        videos_dir = args.videos
        if not os.path.isabs(videos_dir):
            videos_dir = os.path.normpath(os.path.join(script_dir, videos_dir))

        picklist_json_dir = args.picklist_json_dir
        if not os.path.isabs(picklist_json_dir):
            picklist_json_dir = os.path.normpath(os.path.join(script_dir, picklist_json_dir))

        cache_dir = getattr(args, "cache_dir", None)
        if cache_dir:
            if not os.path.isabs(cache_dir):
                cache_dir = os.path.normpath(os.path.join(script_dir, cache_dir))

        run_multi_video_training_from_cache(
            videos_dir=videos_dir,
            picklist_json_dir=picklist_json_dir,
            manual_labels_dir=manual_labels_dir,
            base_output_dir=base_output_dir,
            config=config,
            cache_dir=cache_dir,
            frame_skip=args.frame_skip,
            verbose=args.verbose,
            compact_frame_indexing=args.compact_frame_indexing,
        )

    elif args.command == "sweep":
        import csv

        from ..experiments.assignment_score import score_final_assignments_hit_rate
        from ..experiments.results_manager import (
            aggregate_results,
            find_best_config,
            save_results_csv,
            save_summary_report,
        )
        from ..experiments.sweep_config import (
            SWEEP_PARAM_KEYS,
            build_param_grid_from_cli,
            describe_param_grid,
            generate_configs,
            grid_size,
        )
        from ..experiments.sweep_runner import run_sweep

        param_grid = build_param_grid_from_cli(
            {
                "ilr_epochs": args.sweep_ilr_epochs,
                "n_components": args.sweep_n_components,
                "triplet_margin": args.sweep_triplet_margin,
                "refinement_loops": args.sweep_refinement_loops,
                "adapter_epochs": args.sweep_adapter_epochs,
                "adapter_lr": args.sweep_adapter_lr,
            }
        )

        if args.search_type == "grid":
            gs = grid_size(param_grid)
            if gs > 5000:
                print(
                    f"Warning: full grid has {gs} runs; this may take a very long time. "
                    "Consider --search-type random --num-samples N instead."
                )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.output_dir:
            base_sweep_dir = os.path.normpath(os.path.join(script_dir, args.output_dir))
        else:
            base_sweep_dir = os.path.normpath(
                os.path.join(script_dir, "..", "experiments", "sweeps", f"sweep_{ts}")
            )

        videos_dir = args.videos
        if not os.path.isabs(videos_dir):
            videos_dir = os.path.normpath(os.path.join(script_dir, videos_dir))

        picklist_json_dir = args.picklist_json_dir
        if not os.path.isabs(picklist_json_dir):
            picklist_json_dir = os.path.normpath(os.path.join(script_dir, picklist_json_dir))

        manual_labels_dir = args.manual_labels_dir
        if not os.path.isabs(manual_labels_dir):
            manual_labels_dir = os.path.normpath(os.path.join(script_dir, manual_labels_dir))

        ground_truth_csv = args.ground_truth_csv
        if not os.path.isabs(ground_truth_csv):
            ground_truth_csv = os.path.normpath(os.path.join(script_dir, ground_truth_csv))

        cache_dir = getattr(args, "cache_dir", None) or None
        if cache_dir:
            if not os.path.isabs(cache_dir):
                cache_dir = os.path.normpath(os.path.join(script_dir, cache_dir))

        for p, label in (
            (videos_dir, "--videos"),
            (picklist_json_dir, "--picklist-json-dir"),
            (manual_labels_dir, "--manual-labels-dir"),
            (ground_truth_csv, "--ground-truth-csv"),
        ):
            if not os.path.exists(p):
                print(f"Error: {label} not found: {p}")
                sys.exit(1)

        training_stems = [
            os.path.splitext(os.path.basename(p))[0] for p in _list_videos_in_folder(videos_dir)
        ]
        with open(ground_truth_csv, newline="", encoding="utf-8-sig") as f:
            gt_reader = csv.DictReader(f)
            gt_cols = {
                h.strip()
                for h in (gt_reader.fieldnames or [])
                if h and str(h).strip()
            }
        overlap = [s for s in training_stems if s in gt_cols]
        if not overlap:
            print(
                "Error: no overlap between video stems in --videos and columns in --ground-truth-csv. "
                f"Training stems: {training_stems!r}. GT columns (sample): {sorted(gt_cols)[:20]!r}..."
            )
            sys.exit(1)
        missing_gt = [s for s in training_stems if s not in gt_cols]
        if missing_gt:
            print(
                f"Warning: no ground-truth column for {len(missing_gt)} training stem(s); "
                f"they are skipped in scoring: {missing_gt!r}"
            )

        combos = generate_configs(
            search_type=args.search_type,
            num_samples=args.num_samples,
            random_state=args.sweep_random_seed,
            param_grid=param_grid,
        )
        for c in combos:
            c["use_iterated_model"] = True
            c["ground_truth_csv"] = ground_truth_csv

        meta = {
            "search_type": args.search_type,
            "num_samples": args.num_samples,
            "sweep_random_seed": args.sweep_random_seed,
            "ground_truth_csv": ground_truth_csv,
            "scoring_mode": "final_assignments_csv_hit_rate",
            "sweep_param_keys": list(SWEEP_PARAM_KEYS),
            "param_grid": {k: list(v) for k, v in param_grid.items()},
            "training_video_stems": training_stems,
            "ground_truth_overlap_stems": overlap,
            "n_configs": len(combos),
        }
        os.makedirs(base_sweep_dir, exist_ok=True)
        with open(os.path.join(base_sweep_dir, "sweep_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"Sweep output directory: {base_sweep_dir}")
        print(describe_param_grid(param_grid))
        print(
            f"Training runs: {len(combos)}; scoring = hit rate from each run's final_assignments.csv "
            f"(ground truth columns for {len(overlap)} training stem(s))"
        )

        results = run_sweep(
            videos_dir=videos_dir,
            picklist_json_dir=picklist_json_dir,
            manual_labels_dir=manual_labels_dir,
            base_output_dir=base_sweep_dir,
            param_combinations=combos,
            cache_dir=cache_dir,
            frame_skip=args.frame_skip,
            verbose=args.verbose,
            compact_frame_indexing=args.compact_frame_indexing,
            exclude_stems=None,
        )

        eval_by_run_dir: dict = {}
        for rec in results:
            rec.pop("trainer", None)
            if not rec.get("success"):
                print(f"Run {rec.get('run_index')} failed: {rec.get('error', '')[:200]}")
                continue
            run_dir = rec["run_dir"]
            try:
                ev = score_final_assignments_hit_rate(run_dir)
                with open(os.path.join(run_dir, "eval_results.json"), "w", encoding="utf-8") as f:
                    json.dump(ev, f, indent=2)
                if ev.get("metrics") is None:
                    err = ev.get("error", "unknown scoring error")
                    rec["eval_error"] = err
                    print(f"Scoring failed for {run_dir}: {err}")
                else:
                    eval_by_run_dir[run_dir] = ev
            except Exception as e:
                rec["eval_error"] = str(e)
                print(f"Scoring failed for {run_dir}: {e}")

        df = aggregate_results(results, eval_by_run_dir=eval_by_run_dir)
        if "assignment_hit_rate" in df.columns:
            df = df.sort_values(
                "assignment_hit_rate", ascending=False, na_position="last"
            ).reset_index(drop=True)
        results_csv = os.path.join(base_sweep_dir, "results.csv")
        save_results_csv(df, results_csv)
        best = find_best_config(df)
        summary_md = os.path.join(base_sweep_dir, "summary_report.md")
        save_summary_report(df, summary_md, best=best)

        print(f"\nWrote {results_csv}")
        print(f"Wrote {summary_md}")
        if best:
            print("\nBest run (by assignment_hit_rate from final_assignments.csv):")
            print(f"  run_dir: {best.get('run_dir')}")
            print(f"  assignment_hit_rate: {best.get('assignment_hit_rate')}")
        else:
            print("\nNo successful scored runs to rank.")

    elif args.command == "incremental":
        video_path = args.video
        if not os.path.isabs(video_path):
            video_path = os.path.normpath(os.path.join(script_dir, video_path))
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {args.video}")
            sys.exit(1)

        picklist = None
        label_raw = getattr(args, "label", None)
        if label_raw is not None and str(label_raw).strip() != "":
            try:
                picklist = json.loads(label_raw)
                if not isinstance(picklist, list) or len(picklist) == 0:
                    raise ValueError("Label must be a non-empty JSON array")
                for item in picklist:
                    if not isinstance(item, str):
                        raise ValueError("All picklist items must be strings")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error: {e}")
                sys.exit(1)

        vcfg_inc = getattr(args, "video_config_path", None)
        if vcfg_inc and not os.path.isabs(vcfg_inc):
            vcfg_inc = os.path.normpath(os.path.join(script_dir, vcfg_inc))

        pjdir = getattr(args, "picklist_json_dir", None)
        if pjdir and not os.path.isabs(pjdir):
            pjdir = os.path.normpath(os.path.join(script_dir, pjdir))

        if picklist is None and not vcfg_inc and not pjdir:
            print(
                "Error: provide --label with a JSON picklist array, or --video-config-path, "
                "or --picklist-json-dir (with picklist_{video_stem}.json matching the video)."
            )
            sys.exit(1)

        model_dir = args.model_dir
        if not os.path.isabs(model_dir):
            model_dir = os.path.normpath(os.path.join(script_dir, model_dir))
        if not os.path.isdir(model_dir):
            print(f"Error: Model directory not found: {model_dir}")
            sys.exit(1)

        mdir = args.manual_labels_dir
        if mdir and not os.path.isabs(mdir):
            mdir = os.path.normpath(os.path.join(script_dir, mdir))
        htk = args.htk_model_dir
        if htk and not os.path.isabs(htk):
            htk = os.path.normpath(os.path.join(script_dir, htk))
        aruco = args.aruco_config
        if aruco and not os.path.isabs(aruco):
            aruco = os.path.normpath(os.path.join(script_dir, aruco))

        run_incremental_training(
            video_path=video_path,
            picklist=picklist,
            model_dir=model_dir,
            beta=args.beta,
            threshold=args.threshold,
            frame_skip=args.frame_skip,
            verbose=args.verbose,
            htk_model_dir=htk,
            aruco_config_path=aruco,
            manual_labels_dir=mdir,
            compact_frame_indexing=args.compact_frame_indexing,
            video_config_path=vcfg_inc,
            picklist_json_dir=pjdir,
            force_reembed=getattr(args, "force_reembed", False),
            equal_video_weight=getattr(args, "equal_video_weight", False),
            ilr_epochs_override=getattr(args, "ilr_epochs", None),
            random_seed_override=getattr(args, "random_seed", None),
        )

    elif args.command == "predict":
        model_dir = os.path.normpath(os.path.join(script_dir, args.model_dir))
        image_path = args.image
        
        if not os.path.exists(image_path):
            image_path = os.path.normpath(os.path.join(script_dir, args.image))
        
        if not os.path.exists(image_path):
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)
        
        recognizer = ObjectRecognizer(model_dir)
        
        if args.top_k == 1:
            result = recognizer.predict(image_path)
            if result is None:
                print("Error: Could not process image (hand detection may have failed)")
                sys.exit(1)
            
            print(f"\nPrediction for: {os.path.basename(image_path)}")
            print(f"  Label: {result['label']}")
            print(f"  Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        else:
            results = recognizer.predict_top_k(image_path, k=args.top_k)
            if results is None:
                print("Error: Could not process image (hand detection may have failed)")
                sys.exit(1)
            
            print(f"\nTop-{args.top_k} predictions for: {os.path.basename(image_path)}")
            for rank, (label, conf) in enumerate(results, 1):
                print(f"  {rank}. {label}: {conf:.4f} ({conf*100:.2f}%)")
    
    elif args.command == "infer":
        video_path = args.video
        if not os.path.isabs(video_path):
            video_path = os.path.normpath(os.path.join(script_dir, video_path))
        
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {args.video}")
            sys.exit(1)
        
        model_dir = args.model_dir
        if not os.path.isabs(model_dir):
            model_dir = os.path.normpath(os.path.join(script_dir, model_dir))
        
        if not os.path.exists(model_dir):
            print(f"Error: Model directory not found: {args.model_dir}")
            sys.exit(1)
        
        output_csv = args.output
        if not os.path.isabs(output_csv):
            output_csv = os.path.normpath(os.path.join(script_dir, output_csv))

        picklist_json = getattr(args, "picklist_json", None)
        if picklist_json and not os.path.isabs(picklist_json):
            picklist_json = os.path.normpath(os.path.join(script_dir, picklist_json))

        manual_labels_dir = getattr(args, "manual_labels_dir", None)
        if manual_labels_dir and not os.path.isabs(manual_labels_dir):
            manual_labels_dir = os.path.normpath(os.path.join(script_dir, manual_labels_dir))

        try:
            result_path = run_video_inference(
                video_path=video_path,
                model_dir=model_dir,
                output_csv=output_csv,
                threshold=args.threshold,
                frame_skip=args.frame_skip,
                verbose=args.verbose,
                picklist_json=picklist_json,
                manual_labels_dir=manual_labels_dir,
                compact_frame_indexing=getattr(
                    args, "compact_frame_indexing", "opencv0"
                ),
                apply_iterated_postprocess=getattr(
                    args, "apply_iterated_postprocess", False
                ),
            )
            print(f"\nInference complete! Results saved to: {result_path}")
        except Exception as e:
            print(f"\nError during inference: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()


__all__ = ['main']
