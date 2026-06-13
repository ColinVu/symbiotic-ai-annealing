"""
Frame similarity visualizer: Extract middle frames from segments and create visual
comparison PNGs showing cosine similarities between same-item and different-item frames.

Usage:
    python3 frame_similarity_visualizer.py \\
        --models-root models/classifier \\
        --manual-labels symbiotic-ai/hmm-testing/picklist_labels \\
        --video-dir symbiotic-ai/hmm-testing/picklist_videos \\
        --output-dir frame_similarity_out \\
        --hand-neutralize 50 \\
        --n-samples 20

Dependencies:
    - embedding_analysis (local module)
    - symbiotic-ai (on PYTHONPATH)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Import from embedding_analysis
sys.path.insert(0, str(Path(__file__).parent))
from embedding_analysis.io_cache import load_cache_dir
from embedding_analysis.io_ground_truth import (
    list_stems_in_ground_truth,
    load_ground_truth_column,
)
from embedding_analysis.pipeline import (
    _ensure_symbiote_path,
    get_carry_intervals,
    _video_frame_count,
)
from embedding_analysis.segments import build_segments, middle_index_sorted
from embedding_analysis.geometry import l2_normalize_rows


def _default_symbiotic_root() -> Path:
    return Path(__file__).resolve().parent / "symbiotic-ai"


def _resolve(p: str, base: Path) -> Path:
    return Path(p) if os.path.isabs(p) else (base / p).resolve()


def _get_frame_cache_path(cache_dir: Path, video_stem: str, frame_number: int) -> Path:
    """Get the cache file path for a specific frame."""
    return cache_dir / f"{video_stem}_frame{frame_number:06d}.pkl"


def _load_cached_frame(cache_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load a cached frame if it exists."""
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
            return (data['frame_rgb'], data['frame_cropped'])
    except Exception as e:
        print(f"Warning: Failed to load cached frame {cache_path}: {e}", file=sys.stderr)
        return None


def _save_cached_frame(cache_path: Path, frame_rgb: np.ndarray, frame_cropped: np.ndarray):
    """Save a frame to cache."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'frame_rgb': frame_rgb,
                'frame_cropped': frame_cropped
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"Warning: Failed to save frame to cache {cache_path}: {e}", file=sys.stderr)


def extract_frame_from_video(
    video_path: Path, frame_number_1based: int, hands_detector=None, cache_dir: Optional[Path] = None
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract a specific frame from video and apply hand segmentation.
    
    Args:
        video_path: Path to video file
        frame_number_1based: Frame number (1-based indexing)
        hands_detector: Optional pre-initialized hands detector
        cache_dir: Optional directory to cache extracted frames
    
    Returns:
        Tuple of (original_rgb_frame, hand_cropped_rgb_frame) or None if failed
    """
    # Check cache first
    if cache_dir is not None:
        video_stem = video_path.stem
        cache_path = _get_frame_cache_path(cache_dir, video_stem, frame_number_1based)
        cached = _load_cached_frame(cache_path)
        if cached is not None:
            return cached
    
    # Import inside function to avoid module-level mediapipe import
    try:
        from symbiote_weak_generalized.lib.hand_detection import segment_hand
    except ImportError as e:
        print(f"Error: Missing dependency: {e}", file=sys.stderr)
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    # OpenCV uses 0-based indexing
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number_1based - 1)
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret or frame_bgr is None:
        return None

    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Apply hand segmentation
    import mediapipe as mp
    
    should_close = False
    if hands_detector is None:
        mp_hands = mp.solutions.hands
        hands_detector = mp_hands.Hands(
            min_detection_confidence=0.7, min_tracking_confidence=0.3, max_num_hands=2
        )
        should_close = True

    try:
        segmented = segment_hand(frame_rgb, hands_detector)
        if segmented is None or segmented.size == 0:
            return None
        
        # Save to cache before returning
        if cache_dir is not None:
            _save_cached_frame(cache_path, frame_rgb, segmented)
        
        return (frame_rgb, segmented)
    finally:
        if should_close:
            hands_detector.close()


class MiddleFrameExtractor:
    """Extract middle frames from all segments across all videos."""

    def __init__(
        self,
        models_root: Path,
        manual_labels_dir: Path,
        video_dir: Path,
        symbiotic_ai_root: Path,
        ground_truth_csv: Path,
        frame_skip: int = 4,
        frame_indexing: str = "opencv0",
    ):
        self.models_root = models_root
        self.manual_labels_dir = manual_labels_dir
        self.video_dir = video_dir
        self.symbiotic_ai_root = symbiotic_ai_root
        self.ground_truth_csv = ground_truth_csv
        self.frame_skip = frame_skip
        self.frame_indexing = frame_indexing
        self.cache_base = models_root / ".cache"
        self.frame_cache_dir = models_root / ".frame_cache"
        self.frame_cache_dir.mkdir(parents=True, exist_ok=True)

    def extract_all_middle_frames(
        self, hand_neutralize_components: int = 0, hand_embeddings_dir: Optional[Path] = None
    ) -> List[Dict]:
        """
        Extract middle frames from all segments across all videos.
        
        Returns list of dicts with:
            - video_stem: str
            - segment_idx: int
            - true_label: str
            - frame_number: int (1-based)
            - frame_rgb: np.ndarray (original frame)
            - frame_cropped: np.ndarray (hand-cropped frame)
            - embedding: np.ndarray (neutralized if requested)
        """
        _ensure_symbiote_path(self.symbiotic_ai_root)
        
        # Load ground truth to get video stems
        stems = sorted(x for x in list_stems_in_ground_truth(self.ground_truth_csv) if x)

        # Setup hand neutralizer if requested
        neutralizer = None
        if hand_neutralize_components > 0 and hand_embeddings_dir:
            try:
                from symbiote_weak_generalized.training.hand_neutralizer import HandNeutralizer

                neutralizer = HandNeutralizer(
                    str(hand_embeddings_dir),
                    n_components=hand_neutralize_components,
                    verbose=True,
                )
                if not neutralizer.enabled:
                    print("Warning: Hand neutralizer could not be enabled")
                    neutralizer = None
            except Exception as e:
                print(f"Warning: Could not initialize hand neutralizer: {e}")
                neutralizer = None

        all_frames = []
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        hands_detector = mp_hands.Hands(
            min_detection_confidence=0.7, min_tracking_confidence=0.3, max_num_hands=2
        )

        try:
            for stem in stems:
                print(f"Processing {stem}...", file=sys.stderr)
                
                # Check for video file
                vpath = None
                for ext in [".MP4", ".mp4"]:
                    candidate = self.video_dir / f"{stem}{ext}"
                    if candidate.is_file():
                        vpath = candidate
                        break
                
                if vpath is None:
                    print(f"  [skip] No video file found for {stem}", file=sys.stderr)
                    continue

                # Check for cache dir
                cache_dir = self.cache_base / stem
                if not cache_dir.is_dir():
                    print(f"  [skip] No cache dir for {stem}", file=sys.stderr)
                    continue

                # Load embeddings
                try:
                    emb_by_frame = load_cache_dir(cache_dir)
                    if not emb_by_frame:
                        print(f"  [skip] No embeddings in cache for {stem}", file=sys.stderr)
                        continue
                except Exception as e:
                    print(f"  [skip] Error loading cache for {stem}: {e}", file=sys.stderr)
                    continue

                # Apply hand neutralization to embeddings
                if neutralizer and neutralizer.enabled:
                    emb_by_frame = {
                        frame_num: neutralizer.neutralize(emb)
                        for frame_num, emb in emb_by_frame.items()
                    }

                # Get ground truth labels
                try:
                    expected = load_ground_truth_column(self.ground_truth_csv, stem)
                except Exception as e:
                    print(f"  [skip] Could not load ground truth for {stem}: {e}", file=sys.stderr)
                    continue

                # Get total frames
                total_frames = _video_frame_count(vpath)
                if total_frames <= 0:
                    print(f"  [skip] Could not read frame count for {stem}", file=sys.stderr)
                    continue

                # Get CARRY intervals
                man_csv = self.manual_labels_dir / f"{stem}.csv"
                if not man_csv.is_file():
                    print(f"  [skip] No manual labels for {stem}", file=sys.stderr)
                    continue

                try:
                    intervals = get_carry_intervals(
                        man_csv, total_frames, self.frame_indexing, self.symbiotic_ai_root
                    )
                except Exception as e:
                    print(f"  [skip] Error getting intervals for {stem}: {e}", file=sys.stderr)
                    continue

                # Build segments
                n_use = min(len(intervals), len(expected))
                if n_use < len(intervals):
                    intervals = intervals[:n_use]
                if n_use < len(expected):
                    expected = expected[:n_use]

                segments = build_segments(intervals, expected, emb_by_frame, self.frame_skip)

                # Extract middle frame from each segment
                for seg in segments:
                    if seg.is_placeholder or len(seg.frame_indices_1based) == 0:
                        continue

                    # Get middle frame index
                    mid_idx = middle_index_sorted(seg.frame_indices_1based)
                    
                    # Try to find a usable frame, starting from middle
                    frame_extracted = None
                    embedding = None
                    frame_num_used = None
                    
                    for offset in range(len(seg.frame_indices_1based)):
                        try_idx = mid_idx + offset
                        if try_idx >= len(seg.frame_indices_1based):
                            try_idx = mid_idx - offset
                            if try_idx < 0:
                                continue
                        
                        frame_num = seg.frame_indices_1based[try_idx]
                        
                        # Extract frame from video (with caching)
                        result = extract_frame_from_video(vpath, frame_num, hands_detector, cache_dir=self.frame_cache_dir)
                        if result is not None:
                            frame_rgb, frame_cropped = result
                            frame_extracted = (frame_rgb, frame_cropped)
                            embedding = seg.embeddings[try_idx]
                            frame_num_used = frame_num
                            break
                    
                    if frame_extracted is None:
                        print(f"  [skip] Could not extract frame for {stem} seg {seg.segment_idx}", file=sys.stderr)
                        continue

                    frame_rgb, frame_cropped = frame_extracted
                    
                    all_frames.append({
                        "video_stem": stem,
                        "segment_idx": seg.segment_idx,
                        "true_label": seg.true_label,
                        "frame_number": frame_num_used,
                        "frame_rgb": frame_rgb,
                        "frame_cropped": frame_cropped,
                        "embedding": embedding,
                    })

                print(f"  Extracted {len([f for f in all_frames if f['video_stem'] == stem])} frames from {stem}")

        finally:
            hands_detector.close()

        return all_frames


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    e1 = l2_normalize_rows(emb1.reshape(1, -1))[0]
    e2 = l2_normalize_rows(emb2.reshape(1, -1))[0]
    return float(np.dot(e1, e2))


def create_comparison_png(
    query_frame: Dict,
    comparison_frames: List[Tuple[Dict, float, bool]],
    output_path: Path,
    max_comparisons: int = 20,
):
    """
    Create a PNG showing query frame and comparison frames with cosine similarities.
    
    Args:
        query_frame: Dict with frame_cropped, true_label, video_stem, segment_idx
        comparison_frames: List of (frame_dict, cosine_sim, is_same_item)
        output_path: Where to save the PNG
        max_comparisons: Maximum number of comparison frames to show
    """
    # Thumbnail sizes
    query_size = 300
    comp_size = 150
    padding = 20
    text_height = 80
    
    # Sort by cosine similarity (descending) and take top N
    comparison_frames = sorted(comparison_frames, key=lambda x: x[1], reverse=True)[:max_comparisons]
    
    # Calculate grid layout for comparison frames
    cols = 4
    rows = (len(comparison_frames) + cols - 1) // cols
    
    # Calculate canvas size
    canvas_width = max(query_size + padding * 2, cols * (comp_size + padding) + padding)
    canvas_height = query_size + text_height + padding * 3 + rows * (comp_size + text_height + padding)
    
    # Create canvas
    canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')
    draw = ImageDraw.Draw(canvas)
    
    # Try to load a font
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
    except:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    # Draw query frame
    query_img = Image.fromarray(query_frame['frame_cropped'])
    query_img.thumbnail((query_size, query_size), Image.Resampling.LANCZOS)
    query_x = padding
    query_y = padding
    canvas.paste(query_img, (query_x, query_y))
    
    # Draw query label
    query_text = f"Query: {query_frame['true_label']}"
    query_detail = f"{query_frame['video_stem']}#seg{query_frame['segment_idx']} frame{query_frame['frame_number']}"
    draw.text((query_x, query_y + query_size + 5), query_text, fill='black', font=title_font)
    draw.text((query_x, query_y + query_size + 25), query_detail, fill='gray', font=small_font)
    
    # Draw comparison frames in grid
    start_y = query_y + query_size + text_height + padding
    
    for idx, (comp_frame, cosine_sim, is_same_item) in enumerate(comparison_frames):
        row = idx // cols
        col = idx % cols
        
        x = padding + col * (comp_size + padding)
        y = start_y + row * (comp_size + text_height + padding)
        
        # Draw comparison frame
        comp_img = Image.fromarray(comp_frame['frame_cropped'])
        comp_img.thumbnail((comp_size, comp_size), Image.Resampling.LANCZOS)
        canvas.paste(comp_img, (x, y))
        
        # Draw border (green for same item, red for different)
        border_color = 'green' if is_same_item else 'red'
        draw.rectangle([x-2, y-2, x+comp_img.width+2, y+comp_img.height+2], outline=border_color, width=2)
        
        # Draw label and similarity
        label_text = f"{comp_frame['true_label']}"
        sim_text = f"cos={cosine_sim:.3f}"
        detail_text = f"{comp_frame['video_stem']}#s{comp_frame['segment_idx']}"
        
        draw.text((x, y + comp_img.height + 2), label_text, fill=border_color, font=label_font)
        draw.text((x, y + comp_img.height + 18), sim_text, fill='black', font=small_font)
        draw.text((x, y + comp_img.height + 32), detail_text, fill='gray', font=small_font)
    
    # Save
    canvas.save(output_path)
    print(f"Saved: {output_path}")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Extract middle frames and create similarity comparison PNGs"
    )
    ap.add_argument(
        "--models-root",
        type=str,
        default="models/classifier",
        help="Directory with ground_truth.csv and .cache/{stem}/",
    )
    ap.add_argument(
        "--ground-truth",
        type=str,
        default=None,
        help="ground_truth.csv path (default: <models-root>/ground_truth.csv)",
    )
    ap.add_argument(
        "--manual-labels",
        type=str,
        required=True,
        help="Directory of {stem}.csv compact state labels",
    )
    ap.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Directory containing {stem}.MP4 / .mp4 (default: symbiotic-ai/hmm-testing/picklist_videos)",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="frame_similarity_out",
        help="Where to write comparison PNGs",
    )
    ap.add_argument(
        "--symbiotic-ai",
        type=str,
        default=None,
        dest="symbiotic_ai",
        help="Path to symbiotic-ai/ (default: sibling of this script)",
    )
    ap.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="Must match the cache generation",
    )
    ap.add_argument(
        "--compact-frame-indexing",
        type=str,
        default="opencv0",
        choices=["opencv0", "pipeline1"],
        dest="compact_frame_indexing",
        help="Must match training",
    )
    ap.add_argument(
        "--hand-neutralize",
        type=int,
        default=0,
        dest="hand_neutralize_components",
        help="Apply hand PCA neutralization with n_components (0=disabled)",
    )
    ap.add_argument(
        "--hand-embeddings-dir",
        type=str,
        default=None,
        help="Directory with empty-hand .npy files for PCA fitting",
    )
    ap.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Number of random query frames to visualize",
    )
    ap.add_argument(
        "--comparisons-per-query",
        type=int,
        default=20,
        help="Number of comparison frames to show per query (25%% same item, 75%% different)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )

    args = ap.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)

    here = Path.cwd()
    root = _resolve(args.models_root, here)
    gt = Path(args.ground_truth) if args.ground_truth else (root / "ground_truth.csv")
    manual = _resolve(args.manual_labels, here)
    out = _resolve(args.output_dir, here)
    out.mkdir(parents=True, exist_ok=True)

    sroot = Path(args.symbiotic_ai or _default_symbiotic_root()).resolve()
    if not sroot.is_dir():
        ap.error(f"symbiotic-ai not found at {sroot}")

    vid_dir = _resolve(args.video_dir, here) if args.video_dir else (sroot / "hmm-testing" / "picklist_videos")
    
    hand_neutralize_n = int(args.hand_neutralize_components)
    hand_emb_dir = None
    if hand_neutralize_n > 0:
        if not args.hand_embeddings_dir:
            hand_emb_dir = sroot / "hmm-testing" / "hand_embeddings"
        else:
            hand_emb_dir = _resolve(args.hand_embeddings_dir, here)
        if not hand_emb_dir.is_dir():
            ap.error(
                f"--hand-neutralize={hand_neutralize_n} requires --hand-embeddings-dir; "
                f"expected at {hand_emb_dir}"
            )

    print("Extracting middle frames from all segments...")
    extractor = MiddleFrameExtractor(
        models_root=root,
        manual_labels_dir=manual,
        video_dir=vid_dir,
        symbiotic_ai_root=sroot,
        ground_truth_csv=gt,
        frame_skip=args.frame_skip,
        frame_indexing=args.compact_frame_indexing,
    )

    all_frames = extractor.extract_all_middle_frames(
        hand_neutralize_components=hand_neutralize_n,
        hand_embeddings_dir=hand_emb_dir,
    )

    if not all_frames:
        print("Error: No frames extracted!", file=sys.stderr)
        return 1

    print(f"Extracted {len(all_frames)} total frames")

    # Index frames by item label
    frames_by_item: Dict[str, List[Dict]] = defaultdict(list)
    for frame in all_frames:
        frames_by_item[frame['true_label']].append(frame)

    print(f"Found {len(frames_by_item)} unique items")

    # Sample query frames
    n_samples = min(args.n_samples, len(all_frames))
    query_frames = random.sample(all_frames, n_samples)

    print(f"Creating {n_samples} comparison PNGs...")

    for idx, query_frame in enumerate(query_frames):
        query_label = query_frame['true_label']
        query_emb = query_frame['embedding']
        
        # Get same-item frames (excluding the query itself)
        same_item_frames = [
            f for f in frames_by_item[query_label]
            if not (f['video_stem'] == query_frame['video_stem'] and 
                   f['segment_idx'] == query_frame['segment_idx'])
        ]
        
        # Get different-item frames
        different_item_frames = [
            f for f in all_frames
            if f['true_label'] != query_label
        ]
        
        # Calculate how many of each to sample (25% same item, 75% different)
        n_comp = args.comparisons_per_query
        n_same = min(len(same_item_frames), n_comp // 4)
        n_diff = min(len(different_item_frames), n_comp - n_same)
        
        # If we don't have enough same-item frames, get more different-item frames
        if n_same < n_comp // 4:
            n_diff = min(len(different_item_frames), n_comp - n_same)
        
        # Sample comparison frames
        sampled_same = random.sample(same_item_frames, n_same) if same_item_frames else []
        sampled_diff = random.sample(different_item_frames, n_diff) if different_item_frames else []
        
        # Compute similarities
        comparisons = []
        for frame in sampled_same:
            sim = compute_cosine_similarity(query_emb, frame['embedding'])
            comparisons.append((frame, sim, True))
        
        for frame in sampled_diff:
            sim = compute_cosine_similarity(query_emb, frame['embedding'])
            comparisons.append((frame, sim, False))
        
        # Create PNG
        output_path = out / f"similarity_{idx:03d}_{query_frame['video_stem']}_seg{query_frame['segment_idx']}_{query_label}.png"
        create_comparison_png(query_frame, comparisons, output_path, max_comparisons=n_comp)

    # Create summary JSON
    summary = {
        "total_frames_extracted": len(all_frames),
        "unique_items": len(frames_by_item),
        "n_visualizations": n_samples,
        "hand_neutralize_components": hand_neutralize_n,
        "comparisons_per_query": args.comparisons_per_query,
        "frames_per_item": {label: len(frames) for label, frames in frames_by_item.items()},
    }
    
    summary_path = out / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary written to: {summary_path}")
    print(f"Created {n_samples} comparison PNGs in: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
