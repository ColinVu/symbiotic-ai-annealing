from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from .clip_embedder import ClipEmbedder, normalize_embeddings
from .process_video import _collect_frames, _prepare_embeddings, _align_embeddings


def generate_embeddings_for_videos(
    videos_dir: Path,
    output_dir: Path,
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 16,
    skip_no_hand: bool = False,
) -> None:
    """Generate CLIP embeddings for all videos in a directory.
    
    Args:
        videos_dir: Directory containing video files (.mp4 or .MP4)
        output_dir: Directory to save embeddings (creates if doesn't exist)
        model_name: CLIP model name
        batch_size: Batch size for embedding
        skip_no_hand: Skip frames without detected hands
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video files (case-insensitive, remove duplicates on Windows)
    video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.MP4"))
    # Remove duplicates (Windows file system is case-insensitive)
    video_files = sorted(list(dict.fromkeys(video_files)))
    if not video_files:
        raise ValueError(f"No .mp4 files found in {videos_dir}")
    
    print(f"Found {len(video_files)} videos to process")
    print(f"Output directory: {output_dir.absolute()}")
    
    embedder = ClipEmbedder(model_name=model_name)
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        print(f"\nProcessing: {video_path.name}")
        
        # Collect frames and extract embeddings
        try:
            records, frame_duration = _collect_frames(video_path, skip_no_hand=skip_no_hand)
            if not records:
                print(f"  Warning: No frames extracted, skipping")
                continue
            
            embeddings = _prepare_embeddings(records, embedder, batch_size=batch_size)
            if embeddings.size == 0:
                print(f"  Warning: No embeddings generated, skipping")
                continue
            
            usable_embeddings, timestamps = _align_embeddings(records, embeddings)
            
            # Normalize embeddings
            normalized_embeddings = normalize_embeddings(usable_embeddings)
            
            # Save to file
            output_file = output_dir / f"{video_path.stem}.npz"
            print(f"  Saving to: {output_file.absolute()}")
            
            np.savez_compressed(
                output_file,
                embeddings=normalized_embeddings,
                timestamps=np.array(timestamps),
                frame_duration=frame_duration,
                video_name=video_path.name,
            )
            
            # Verify file was created
            if output_file.exists():
                file_size = output_file.stat().st_size / 1024  # KB
                print(f"  ✓ Saved: {output_file.name} ({file_size:.1f} KB)")
                print(f"  Embeddings shape: {normalized_embeddings.shape}")
                print(f"  Duration: {timestamps[-1]:.2f}s ({len(timestamps)} frames)")
            else:
                print(f"  ✗ ERROR: File was not created at {output_file.absolute()}")
            
        except Exception as e:
            print(f"  Error processing {video_path.name}: {e}")
            continue
    
    print(f"\nCompleted! Embeddings saved to {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CLIP embeddings for video files."
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        required=True,
        help="Directory containing training videos (.mp4 files)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("embeddings"),
        help="Directory to save embeddings (default: ./embeddings)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP model id",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of frames to embed per batch",
    )
    parser.add_argument(
        "--skip-no-hand",
        action="store_true",
        help="Drop frames without detected hands",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_embeddings_for_videos(
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        skip_no_hand=args.skip_no_hand,
    )


if __name__ == "__main__":
    main()
