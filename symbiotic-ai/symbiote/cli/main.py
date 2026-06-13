"""Command-line interface for the video-to-classification pipeline."""

import argparse
import sys
import os

from ..core.config import DEFAULT_CONFIG
from ..pipelines.video_training import run_video_training
from ..pipelines.video_inference import run_video_inference
from ..inference.recognizer import ObjectRecognizer
from ..state_detection.training import train_state_detector
from ..state_detection.detector import detect_states_from_video, HandState


def main():
    parser = argparse.ArgumentParser(
        description="Video-to-Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from video (basic)
  python -m symbiote.cli.main train --video ../videos/object1.mp4 --label "object1"
  
  # Train with custom parameters
  python -m symbiote.cli.main train --video ../videos/object1.mp4 --label "object1" --threshold 150.0 --frame-skip 6
  
  # Predict on a single image
  python -m symbiote.cli.main predict --model-dir ../models/classifier/video_name --image ../images/test.jpg
  
  # Get top-3 predictions
  python -m symbiote.cli.main predict --model-dir ../models/classifier/video_name --image ../images/test.jpg --top-k 3
  
  # Run inference on video and output CSV
  python -m symbiote.cli.main infer --video ../videos/test.mp4 --model-dir ../models/classifier/video_name --output results.csv
  
  # Inference with custom frame skip and blur threshold
  python -m symbiote.cli.main infer --video ../videos/test.mp4 --model-dir ../models/classifier/video_name --output results.csv --frame-skip 10 --threshold 120.0
  
  # Train HTK HMM state detector
  python -m symbiote.cli.main train-hmm --videos video1.mp4 video2.mp4 --annotations ann1.csv ann2.csv --output-dir ../models/htk --aruco-config config/aruco_bins.json
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a classifier from video")
    train_parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file to process"
    )
    train_parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Class label for frames from this video"
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
        default=100.0,
        help="Blur detection threshold (Laplacian variance, default 100.0)"
    )
    train_parser.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="Process every Nth frame (default 4)"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_CONFIG["max_epochs"],
        help="Maximum number of training epochs"
    )
    train_parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_CONFIG["early_stopping_patience"],
        help="Early stopping patience"
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_CONFIG["learning_rate"],
        help="Learning rate"
    )
    train_parser.add_argument(
        "--hidden-dim",
        type=int,
        default=DEFAULT_CONFIG["hidden_dim"],
        help="Hidden layer dimension"
    )
    train_parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Show detailed progress"
    )
    train_parser.add_argument(
        "--image-dir",
        type=str,
        default="../images/image-testing",
        help="Directory with image folders (for loading old cache files from classifier_pipeline)"
    )
    train_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache (re-embed all images every run)"
    )
    train_parser.add_argument(
        "--htk-model-dir",
        type=str,
        default=None,
        help="Path to trained HTK HMM model directory (for state detection during training)"
    )
    train_parser.add_argument(
        "--aruco-config",
        type=str,
        default=None,
        help="Path to ARUCO marker configuration JSON"
    )
    
    # Train-HMM command
    hmm_parser = subparsers.add_parser(
        "train-hmm",
        help="Train HTK HMM state detector from annotated videos"
    )
    hmm_parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="Paths to training video files"
    )
    hmm_parser.add_argument(
        "--annotations",
        nargs="+",
        required=True,
        help="Paths to CSV annotation files (one per video, same order)"
    )
    hmm_parser.add_argument(
        "--output-dir",
        type=str,
        default="../models/htk",
        help="Directory to save trained HTK HMM model"
    )
    hmm_parser.add_argument(
        "--aruco-config",
        type=str,
        default=None,
        help="Path to ARUCO marker configuration JSON"
    )
    hmm_parser.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="Process every Nth frame (default 4)"
    )
    hmm_parser.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help="Blur detection threshold (Laplacian variance, default 100.0)"
    )
    hmm_parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Show detailed progress"
    )
    
    # Predict command
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
    
    # Infer command
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
        default=100.0,
        help="Blur detection threshold (Laplacian variance, default 100.0)"
    )
    infer_parser.add_argument(
        "--frame-skip",
        type=int,
        default=5,
        help="Process every Nth frame (default 5)"
    )
    infer_parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Show detailed progress"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Resolve paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.command == "train":
        video_path = args.video
        if not os.path.isabs(video_path):
            video_path = os.path.normpath(os.path.join(script_dir, video_path))
        
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {args.video}")
            sys.exit(1)
        
        base_output_dir = os.path.normpath(os.path.join(script_dir, args.output_dir))
        
        # Resolve image_dir path
        image_dir = None
        if args.image_dir:
            image_dir = args.image_dir
            if not os.path.isabs(image_dir):
                image_dir = os.path.normpath(os.path.join(script_dir, image_dir))
        
        # Build config
        config = DEFAULT_CONFIG.copy()
        config["max_epochs"] = args.epochs
        config["early_stopping_patience"] = args.patience
        config["learning_rate"] = args.lr
        config["hidden_dim"] = args.hidden_dim
        
        # Resolve optional HTK paths
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
        
        run_video_training(
            video_path=video_path,
            label=args.label,
            base_output_dir=base_output_dir,
            config=config,
            threshold=args.threshold,
            frame_skip=args.frame_skip,
            image_dir=image_dir,
            verbose=args.verbose,
            htk_model_dir=htk_model_dir,
            aruco_config_path=aruco_config,
        )
        
    elif args.command == "predict":
        model_dir = os.path.normpath(os.path.join(script_dir, args.model_dir))
        image_path = args.image
        
        if not os.path.exists(image_path):
            # Try relative to script dir
            image_path = os.path.normpath(os.path.join(script_dir, args.image))
        
        if not os.path.exists(image_path):
            print(f"Error: Image not found: {args.image}")
            sys.exit(1)
        
        # Load recognizer
        recognizer = ObjectRecognizer(model_dir)
        
        # Run prediction
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
        
        # Run video inference
        try:
            result_path = run_video_inference(
                video_path=video_path,
                model_dir=model_dir,
                output_csv=output_csv,
                threshold=args.threshold,
                frame_skip=args.frame_skip,
                verbose=args.verbose
            )
            print(f"\nInference complete! Results saved to: {result_path}")
        except Exception as e:
            print(f"\nError during inference: {e}")
            sys.exit(1)
    
    elif args.command == "train-hmm":
        # Resolve video paths
        video_paths = []
        for vp in args.videos:
            if not os.path.isabs(vp):
                vp = os.path.normpath(os.path.join(script_dir, vp))
            if not os.path.exists(vp):
                print(f"Error: Video file not found: {vp}")
                sys.exit(1)
            video_paths.append(vp)
        
        # Resolve annotation paths
        annotation_paths = []
        for ap in args.annotations:
            if not os.path.isabs(ap):
                ap = os.path.normpath(os.path.join(script_dir, ap))
            if not os.path.exists(ap):
                print(f"Error: Annotation file not found: {ap}")
                sys.exit(1)
            annotation_paths.append(ap)
        
        if len(video_paths) != len(annotation_paths):
            print("Error: Number of --videos must match number of --annotations")
            sys.exit(1)
        
        output_dir = os.path.normpath(os.path.join(script_dir, args.output_dir))
        
        aruco_config = None
        if args.aruco_config:
            aruco_config = args.aruco_config
            if not os.path.isabs(aruco_config):
                aruco_config = os.path.normpath(os.path.join(script_dir, aruco_config))
        
        try:
            final_model_dir = train_state_detector(
                video_paths=video_paths,
                annotation_paths=annotation_paths,
                output_dir=output_dir,
                aruco_config_path=aruco_config,
                frame_skip=args.frame_skip,
                blur_threshold=args.threshold,
                verbose=args.verbose,
            )
            print(f"\nHTK HMM training complete! Model saved to: {final_model_dir}")
        except Exception as e:
            print(f"\nError during HTK HMM training: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()


__all__ = ['main']
