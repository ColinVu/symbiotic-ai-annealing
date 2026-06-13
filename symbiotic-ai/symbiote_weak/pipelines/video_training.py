"""Video-based weakly supervised training pipeline."""

import os
import json
import random
import re
from typing import Dict, Any, Optional, List, Tuple, AbstractSet
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import cv2
import pandas as pd
from transformers import AutoModel, AutoProcessor

from ..core.config import MODEL
from ..embeddings.cache_manager import load_frame_from_cache
from ..preprocessing.video_processor import process_video_frames
from ..training.weak_supervision import WeakSupervisedTrainer, Segment
from ..persistence.model_io import save_model, load_weak_trainer
from ..state_detection.detector import detect_states_from_video, HandState
from ..state_detection.compact_timeline import (
    load_state_labels_auto,
    carry_with_pipeline_frame_intervals_1based,
)


def _load_flat_picklist_from_video_config_json(json_path: str) -> List[str]:
    """Same flattening as training: one SKU string per carry segment, temporal order."""
    with open(json_path, "r", encoding="utf-8") as f:
        video_config = json.load(f)
    picklists_nested = video_config.get("picklists", [])
    if not picklists_nested:
        raise ValueError(f"No 'picklists' in {json_path}")
    return [str(item) for sublist in picklists_nested for item in sublist]


def _resolve_incremental_picklist(
    video_path: str,
    picklist: Optional[List[str]],
    video_config_path: Optional[str],
    picklist_json_dir: Optional[str],
) -> Tuple[List[str], Optional[str]]:
    """
    Resolve flat picklist for incremental training.

    Returns:
        (flat_picklist, json_path_used_or_none if loaded from file)
    """
    if picklist is not None and len(picklist) > 0:
        return picklist, None
    json_path: Optional[str] = None
    if video_config_path:
        json_path = video_config_path
    elif picklist_json_dir:
        stem = Path(video_path).stem
        json_path = os.path.join(picklist_json_dir, f"{stem}.json")
    else:
        raise ValueError(
            "Provide a non-empty --label JSON array, or --video-config-path, "
            "or --picklist-json-dir (expects picklist_{video_stem}.json)."
        )
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Picklist JSON not found: {json_path}")
    return _load_flat_picklist_from_video_config_json(json_path), json_path


def _flatten_candidate_assignments(picklists_nested: List[List[str]]) -> List[Tuple[str, ...]]:
    """One multiset tuple per carry segment in temporal order."""
    out: List[Tuple[str, ...]] = []
    for block in picklists_nested:
        tup = tuple(str(x) for x in block)
        for _ in block:
            out.append(tup)
    return out


def _group_frames_into_segments(
    embeddings: List[np.ndarray],
    frame_numbers: List[int],
    state_results,
    fps: float,
    video_id: str,
    per_segment_candidates: Optional[List[Tuple[str, ...]]] = None,
    clip_dim_fallback: int = 512,
) -> List[Segment]:
    """
    Group frame embeddings into segments based on state detection boundaries.
    
    Creates one segment per CARRY_WITH interval. If an interval has zero valid frames,
    a placeholder segment with a zero embedding vector is created to preserve temporal
    alignment with the picklist.
    
    Args:
        embeddings: List of CLIP embeddings for each frame
        frame_numbers: Frame numbers corresponding to embeddings
        state_results: DataFrame with state detection results (timestamp_start, timestamp_end, state)
        fps: Video frames per second
        video_id: Identifier for this video
        per_segment_candidates: Optional candidate labels per segment
        clip_dim_fallback: CLIP dim for zero placeholder vectors when no embeddings exist
        
    Returns:
        List of Segment objects, one per CARRY_WITH segment (includes placeholders)
    """
    if state_results.empty:
        segment = Segment(
            segment_id=0,
            embeddings=np.array(embeddings),
            video_id=video_id,
        )
        return [segment]
    
    carry_with_rows = state_results[state_results['state'] == HandState.CARRY_WITH.value]
    
    if carry_with_rows.empty:
        segment = Segment(
            segment_id=0,
            embeddings=np.array(embeddings),
            video_id=video_id,
        )
        return [segment]
    
    segments = []
    segment_id_counter = 0
    
    for _, row in carry_with_rows.iterrows():
        t_start = row['timestamp_start']
        t_end = row['timestamp_end']
        
        segment_embeddings = []
        for emb, frame_num in zip(embeddings, frame_numbers):
            frame_time = frame_num / fps if fps > 0 else 0.0
            if t_start <= frame_time <= t_end:
                segment_embeddings.append(emb)
        
        cand: Optional[Tuple[str, ...]] = None
        if per_segment_candidates and segment_id_counter < len(per_segment_candidates):
            cand = per_segment_candidates[segment_id_counter]
        
        if len(segment_embeddings) > 0:
            segment = Segment(
                segment_id=segment_id_counter,
                embeddings=np.array(segment_embeddings),
                video_id=video_id,
                candidate_labels=cand,
                is_placeholder=False,
            )
        else:
            # Placeholder: no valid frames in this interval
            inferred_dim = embeddings[0].shape[0] if embeddings else clip_dim_fallback
            segment = Segment(
                segment_id=segment_id_counter,
                embeddings=np.zeros((1, inferred_dim), dtype=np.float64),
                video_id=video_id,
                candidate_labels=cand,
                is_placeholder=True,
            )
        
        segments.append(segment)
        segment_id_counter += 1
    
    if len(segments) == 0:
        segment = Segment(
            segment_id=0,
            embeddings=np.array(embeddings),
            video_id=video_id,
        )
        return [segment]
    
    return segments


def _load_picklists_nested_from_json(json_path: str) -> List[List[str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        video_config = json.load(f)
    picklists_nested = video_config.get("picklists", [])
    if not picklists_nested:
        raise ValueError(f"No 'picklists' in {json_path}")
    return picklists_nested


def _process_single_video_to_segments(
    video_path: str,
    picklists_nested: List[List[str]],
    clip_model,
    processor,
    cache_dir: str,
    manual_labels_dir: Optional[str],
    require_manual_label_csv: bool,
    compact_frame_indexing: str,
    threshold: float,
    frame_skip: int,
    htk_model_dir: Optional[str],
    aruco_config_path: Optional[str],
    verbose: bool,
) -> Tuple[List[Segment], List[str], str]:
    """
    Embed one video, run state detection, and build carry segments aligned to the flat picklist.

    Returns:
        (segments, flat_picklist, video_stem)
    """
    if not Path(video_path).exists():
        raise SystemExit(f"Error: Video file not found: {video_path}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    annotation_path: Optional[str] = None
    if manual_labels_dir is not None:
        annotation_path = os.path.join(manual_labels_dir, f"{video_name}.csv")
        if require_manual_label_csv:
            if not os.path.isfile(annotation_path):
                raise SystemExit(
                    f"Error: Multi-video train requires manual label CSV for each video. "
                    f"Missing: {annotation_path}"
                )

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    carry_intervals = None
    if annotation_path is not None and os.path.isfile(annotation_path):
        cap_i = cv2.VideoCapture(video_path)
        n_total = int(cap_i.get(cv2.CAP_PROP_FRAME_COUNT)) if cap_i.isOpened() else 0
        cap_i.release()
        carry_intervals = carry_with_pipeline_frame_intervals_1based(
            annotation_path,
            n_total,
            frame_indexing=compact_frame_indexing,
        )
        if verbose and carry_intervals:
            print(
                f"  [{video_name}] Manual labels: embedding only inside "
                f"{len(carry_intervals)} CARRY_WITH span(s) from {annotation_path}"
            )

    def _state_detect(video_path, embeddings, frame_numbers, fps):
        # First arg must be named video_path: process_video_frames calls with keyword video_path=...
        if annotation_path is not None and os.path.exists(annotation_path):
            cap_d = cv2.VideoCapture(video_path)
            nfr = int(cap_d.get(cv2.CAP_PROP_FRAME_COUNT)) if cap_d.isOpened() else 0
            cap_d.release()
            dur = (nfr / fps) if fps and nfr else None
            return load_state_labels_auto(
                annotation_path,
                fps=float(fps),
                frame_indexing=compact_frame_indexing,
                video_duration_sec=dur,
            )
        return detect_states_from_video(
            video_path,
            embeddings,
            frame_numbers,
            fps,
            htk_model_dir=htk_model_dir,
            aruco_config_path=aruco_config_path,
            frame_skip=frame_skip,
            blur_threshold=threshold,
            clip_model=clip_model,
            clip_processor=processor,
            verbose=verbose,
        )

    flat_picklist = [item for sublist in picklists_nested for item in sublist]
    first_label = flat_picklist[0] if flat_picklist else "unknown"

    video_embeddings, _, _, state_results, embedding_frame_indices = process_video_frames(
        video_path,
        first_label,
        clip_model,
        processor,
        cache_dir,
        threshold=threshold,
        frame_skip=frame_skip,
        state_detection_func=_state_detect,
        verbose=verbose,
        allowed_frame_intervals_1based=carry_intervals,
    )

    if len(video_embeddings) == 0:
        raise SystemExit(f"Error: No valid frames extracted from video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
    else:
        fps = 30.0

    per_segment_candidates = _flatten_candidate_assignments(picklists_nested)
    segments = _group_frames_into_segments(
        video_embeddings,
        embedding_frame_indices,
        state_results,
        fps,
        video_name,
        per_segment_candidates=per_segment_candidates,
    )

    if verbose:
        print(f"  [{video_name}] Extracted {len(segments)} carry segments")
        n_placeholders = sum(1 for seg in segments if seg.is_placeholder)
        if n_placeholders:
            print(
                f"  [{video_name}] WARNING: {n_placeholders} segment(s) are placeholders "
                "(zero valid frames in their CARRY_WITH interval)"
            )

    if len(segments) != len(flat_picklist):
        if verbose:
            print(
                f"  WARNING [{video_name}]: segments ({len(segments)}) != picklist "
                f"({len(flat_picklist)}); alignment error."
            )
        if len(segments) > len(flat_picklist):
            segments = segments[: len(flat_picklist)]
        else:
            # Pad with placeholders to match picklist length
            inferred_dim = segments[0].embeddings.shape[1] if segments else 512
            while len(segments) < len(flat_picklist):
                idx = len(segments)
                cand = per_segment_candidates[idx] if idx < len(per_segment_candidates) else None
                segments.append(
                    Segment(
                        segment_id=idx,
                        embeddings=np.zeros((1, inferred_dim), dtype=np.float64),
                        video_id=video_name,
                        candidate_labels=cand,
                        is_placeholder=True,
                    )
                )

    return segments, flat_picklist, video_name


def _glob_cached_frame_indices_first_label(cache_dir: str, first_label: str) -> List[int]:
    """
    List 1-based frame indices present on disk for ``{first_label}_frame_<n>_<md5>.npy``.
    """
    if not os.path.isdir(cache_dir):
        return []
    pat = re.compile(
        rf"^{re.escape(first_label)}_frame_(\d+)_[a-f0-9]{{32}}\.npy$"
    )
    out: List[int] = []
    for name in os.listdir(cache_dir):
        if name.endswith("_seg.npy"):
            continue
        m = pat.match(name)
        if m:
            out.append(int(m.group(1)))
    out.sort()
    return out


def _candidate_frame_indices_carry_and_skip(
    carry_intervals: Optional[List[Tuple[int, int]]],
    n_total: int,
    frame_skip: int,
) -> List[int]:
    """
    1-based frame indices that ``process_video_frames`` would consider after
    ``frame_skip`` filtering and optional CARRY_WITH windowing.
    """
    if n_total <= 0:
        return []
    out: List[int] = []
    if carry_intervals:
        for lo, hi in carry_intervals:
            lo2 = max(1, int(lo))
            hi2 = min(int(hi), n_total)
            for t in range(lo2, hi2 + 1):
                if t % frame_skip == 0:
                    out.append(t)
    else:
        for t in range(1, n_total + 1):
            if t % frame_skip == 0:
                out.append(t)
    return out


def _process_single_video_from_cache(
    video_path: str,
    picklists_nested: List[List[str]],
    cache_dir: str,
    manual_labels_dir: Optional[str],
    require_manual_label_csv: bool,
    compact_frame_indexing: str,
    frame_skip: int,
    verbose: bool,
) -> Tuple[List[Segment], List[str], str]:
    """
    Rebuild carry segments from disk cache only (no CLIP / no full-video decode).

    Uses the same cache key scheme as ``save_frame_to_cache``: first flat picklist
    label + 1-based frame index. Loads **only frames that exist on disk** whose
    indices lie in CARRY_WITH spans and satisfy ``frame_skip`` (sparse cache:
    training skips frames that fail hand detection or blur, so not every candidate
    index has a file).

    Opens the video once to read FPS and frame count only (container metadata).
    """
    if not Path(video_path).exists():
        raise SystemExit(f"Error: Video file not found: {video_path}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    annotation_path: Optional[str] = None
    if manual_labels_dir is not None:
        annotation_path = os.path.join(manual_labels_dir, f"{video_name}.csv")
        if require_manual_label_csv:
            if not os.path.isfile(annotation_path):
                raise SystemExit(
                    f"Error: Multi-video cache train requires manual label CSV for each video. "
                    f"Missing: {annotation_path}"
                )

    cap_meta = cv2.VideoCapture(video_path)
    if not cap_meta.isOpened():
        raise SystemExit(f"Error: Cannot open video for metadata: {video_path}")
    fps = float(cap_meta.get(cv2.CAP_PROP_FPS) or 30.0)
    n_total = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap_meta.release()

    carry_intervals = None
    if annotation_path is not None and os.path.isfile(annotation_path):
        carry_intervals = carry_with_pipeline_frame_intervals_1based(
            annotation_path,
            n_total,
            frame_indexing=compact_frame_indexing,
        )
        if verbose and carry_intervals:
            print(
                f"  [{video_name}] Manual labels: cache load inside "
                f"{len(carry_intervals)} CARRY_WITH span(s) from {annotation_path}"
            )

    flat_picklist = [item for sublist in picklists_nested for item in sublist]
    first_label = flat_picklist[0] if flat_picklist else "unknown"
    per_segment_candidates = _flatten_candidate_assignments(picklists_nested)

    candidate_frames = _candidate_frame_indices_carry_and_skip(
        carry_intervals, n_total, frame_skip
    )
    candidate_set = set(candidate_frames)
    glob_frames = _glob_cached_frame_indices_first_label(cache_dir, first_label)
    usable_frames = [
        t for t in glob_frames if 1 <= t <= n_total and t in candidate_set
    ]

    if not candidate_frames:
        raise SystemExit(
            f"Error: No candidate frames for cache load (video {video_name!r}, "
            f"n_total={n_total}, frame_skip={frame_skip}). Check manual CSV / CARRY_WITH spans."
        )

    if not usable_frames:
        raise SystemExit(
            f"Error: No usable cached embeddings under {cache_dir!r} for label={first_label!r} "
            f"after CARRY_WITH + frame_skip filter (glob found {len(glob_frames)} files). "
            f"Run full ``train`` once to populate cache, or fix --cache-dir / manual CSV alignment."
        )

    video_embeddings: List[np.ndarray] = []
    embedding_frame_indices: List[int] = []
    for frame_num in usable_frames:
        emb = load_frame_from_cache(first_label, frame_num, cache_dir)
        if emb is not None:
            video_embeddings.append(np.asarray(emb, dtype=np.float64))
            embedding_frame_indices.append(frame_num)

    if annotation_path is not None and os.path.isfile(annotation_path):
        dur = (n_total / fps) if fps and n_total else None
        state_results = load_state_labels_auto(
            annotation_path,
            fps=float(fps),
            frame_indexing=compact_frame_indexing,
            video_duration_sec=dur,
        )
    else:
        state_results = pd.DataFrame()

    segments = _group_frames_into_segments(
        video_embeddings,
        embedding_frame_indices,
        state_results,
        fps,
        video_name,
        per_segment_candidates=per_segment_candidates,
    )

    if verbose:
        print(f"  [{video_name}] From cache: {len(video_embeddings)} frames -> {len(segments)} carry segments")

    if len(segments) != len(flat_picklist):
        if verbose:
            print(
                f"  WARNING [{video_name}]: segments ({len(segments)}) != picklist "
                f"({len(flat_picklist)}); trimming/padding like full train."
            )
        if len(segments) > len(flat_picklist):
            segments = segments[: len(flat_picklist)]
        else:
            while len(segments) < len(flat_picklist):
                all_embs = np.vstack([seg.embeddings for seg in segments])
                n_per_seg = max(1, len(all_embs) // len(flat_picklist))
                segments = []
                for i in range(len(flat_picklist)):
                    start_idx = i * n_per_seg
                    end_idx = start_idx + n_per_seg if i < len(flat_picklist) - 1 else len(all_embs)
                    cand = per_segment_candidates[i] if i < len(per_segment_candidates) else None
                    segments.append(
                        Segment(
                            segment_id=i,
                            embeddings=all_embs[start_idx:end_idx],
                            video_id=video_name,
                            candidate_labels=cand,
                        )
                    )
                break

    return segments, flat_picklist, video_name


def run_multi_video_training_from_cache(
    videos_dir: str,
    picklist_json_dir: str,
    manual_labels_dir: str,
    base_output_dir: str,
    config: Dict[str, Any],
    cache_dir: Optional[str] = None,
    frame_skip: int = 4,
    verbose: bool = True,
    compact_frame_indexing: str = "opencv0",
    exclude_stems: Optional[AbstractSet[str]] = None,
) -> WeakSupervisedTrainer:
    """
    Same joint weak supervision as ``run_multi_video_training``, but segments are
    built only from ``.npy`` files under ``cache_dir`` (default ``output/.cache``).

    Does not load CLIP or decode video frames for embedding. Still opens each video
    briefly to read FPS and frame count for state timeline alignment.
    """
    random.seed(config.get("random_seed", 42))
    np.random.seed(config.get("random_seed", 42))
    torch.manual_seed(config.get("random_seed", 42))

    if not os.path.isdir(picklist_json_dir):
        raise SystemExit(f"Error: picklist JSON directory not found: {picklist_json_dir}")
    if not os.path.isdir(manual_labels_dir):
        raise SystemExit(f"Error: manual labels directory not found: {manual_labels_dir}")

    video_paths = _list_videos_in_folder(videos_dir)
    if exclude_stems:
        before = len(video_paths)
        video_paths = [
            vp for vp in video_paths if Path(vp).stem not in exclude_stems
        ]
        if not video_paths:
            raise SystemExit(
                f"Error: After exclude_stems={sorted(exclude_stems)!r}, no videos left in {videos_dir!r} "
                f"(had {before} before filter)."
            )
        if verbose:
            print(
                f"Excluded stems {sorted(exclude_stems)!r} from training; "
                f"{len(video_paths)} video(s) remain."
            )
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    cache_root = cache_dir if cache_dir else os.path.join(base_output_dir, ".cache")

    if verbose:
        print("=" * 60)
        print("MULTI-VIDEO TRAINING FROM DISK CACHE (no CLIP embedding pass)")
        print("=" * 60)
        print(f"Videos directory (for stems + metadata): {videos_dir}")
        print(f"Picklist JSON dir: {picklist_json_dir}")
        print(f"Manual labels dir: {manual_labels_dir}")
        print(f"Cache root: {cache_root}")
        print(f"Videos ({len(video_paths)}): {[os.path.basename(v) for v in video_paths]}")
        print(f"Output: {base_output_dir}")
        print(f"frame_skip: {frame_skip} (must match the cache that was written)")

    video_segments: Dict[str, Tuple[List[Segment], List[str]]] = {}

    for video_path in video_paths:
        stem = Path(video_path).stem
        json_path = os.path.join(picklist_json_dir, f"{stem}.json")
        if not os.path.isfile(json_path):
            raise SystemExit(
                f"Error: Missing picklist JSON for video {stem!r}: expected {json_path}"
            )
        picklists_nested = _load_picklists_nested_from_json(json_path)
        per_video_cache = os.path.join(cache_root, stem)
        if not os.path.isdir(per_video_cache):
            raise SystemExit(
                f"Error: No cache directory for {stem!r}: expected {per_video_cache}"
            )
        segments, flat_picklist, video_name = _process_single_video_from_cache(
            video_path,
            picklists_nested,
            per_video_cache,
            manual_labels_dir,
            require_manual_label_csv=True,
            compact_frame_indexing=compact_frame_indexing,
            frame_skip=frame_skip,
            verbose=verbose,
        )
        video_segments[video_name] = (segments, flat_picklist)

    trainer = WeakSupervisedTrainer(
        ilr_epochs=config.get("ilr_epochs", 500),
        initial_temp=config.get("initial_temp", 1.0),
        temp_decay=config.get("temp_decay", "exponential"),
        decay_rate=config.get("decay_rate", 0.99),
        random_seed=config.get("random_seed", 42),
        variance_eps=config.get("variance_eps", 1e-6),
        bad_swap_cool_divisor=config.get("bad_swap_cool_divisor", 50.0),
        detect_empty=config.get("detect_empty", False),
        min_frames_per_cluster=config.get("min_frames_per_cluster", 3),
        ilr_allow_cross_round_swaps=bool(config.get("ilr_allow_cross_round_swaps", False)),
    )
    use_cv = bool(config.get("use_cluster_voting", False))
    _init_vote_csv = (
        os.path.join(base_output_dir, "initial_cluster_voting.csv") if use_cv else None
    )
    trainer.fit(
        video_segments,
        verbose=verbose,
        skip_ilr=bool(config.get("skip_ilr", False)),
        initial_cluster_voting_csv=_init_vote_csv,
        use_cluster_voting=use_cv,
    )

    stem_order = [os.path.splitext(os.path.basename(vp))[0] for vp in video_paths]
    save_model(
        trainer,
        config,
        base_output_dir,
        embedded_video_stems_override=stem_order,
    )
    if verbose:
        print(f"\n✓ Model saved to {base_output_dir}")
        print(f"  embedded_video_stems: {stem_order}")

    return trainer


def _list_videos_in_folder(videos_dir: str) -> List[str]:
    """Return sorted paths to .mp4 / .MP4 files in *videos_dir* (non-recursive)."""
    p = Path(videos_dir)
    if not p.is_dir():
        raise SystemExit(f"Error: --videos is not a directory: {videos_dir}")
    paths: List[str] = []
    for pattern in ("*.mp4", "*.MP4", "*.m4v", "*.M4V"):
        paths.extend(str(x) for x in p.glob(pattern))
    paths = sorted(set(paths), key=lambda x: os.path.basename(x).lower())
    if not paths:
        raise SystemExit(f"Error: No .mp4/.MP4 videos found in {videos_dir}")
    return paths


def run_multi_video_training(
    videos_dir: str,
    picklist_json_dir: str,
    manual_labels_dir: str,
    base_output_dir: str,
    config: Dict[str, Any],
    threshold: float = 50.0,
    frame_skip: int = 4,
    verbose: bool = True,
    htk_model_dir: Optional[str] = None,
    aruco_config_path: Optional[str] = None,
    compact_frame_indexing: str = "opencv0",
) -> WeakSupervisedTrainer:
    """
    Joint weak supervision training on every video in *videos_dir*.

    For each ``picklist_<stem>.MP4``, requires ``picklist_<stem>.json`` under
    *picklist_json_dir* and ``picklist_<stem>.csv`` under *manual_labels_dir*.
    """
    random.seed(config.get("random_seed", 42))
    np.random.seed(config.get("random_seed", 42))
    torch.manual_seed(config.get("random_seed", 42))

    if not os.path.isdir(picklist_json_dir):
        raise SystemExit(f"Error: picklist JSON directory not found: {picklist_json_dir}")
    if not os.path.isdir(manual_labels_dir):
        raise SystemExit(f"Error: manual labels directory not found: {manual_labels_dir}")

    video_paths = _list_videos_in_folder(videos_dir)
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    base_cache = os.path.join(base_output_dir, ".cache")

    if verbose:
        print("=" * 60)
        print("MULTI-VIDEO WEAKLY SUPERVISED TRAINING")
        print("=" * 60)
        print(f"Videos directory: {videos_dir}")
        print(f"Picklist JSON dir: {picklist_json_dir}")
        print(f"Manual labels dir: {manual_labels_dir}")
        print(f"Videos ({len(video_paths)}): {[os.path.basename(v) for v in video_paths]}")
        print(f"Output: {base_output_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = AutoModel.from_pretrained(MODEL)
    clip_model.eval()
    if device == "cuda":
        clip_model = clip_model.to(device)
    processor = AutoProcessor.from_pretrained(MODEL)
    if verbose:
        print(f"✓ CLIP model loaded (device: {device})")

    video_segments: Dict[str, Tuple[List[Segment], List[str]]] = {}

    for video_path in video_paths:
        stem = Path(video_path).stem
        json_path = os.path.join(picklist_json_dir, f"{stem}.json")
        if not os.path.isfile(json_path):
            raise SystemExit(
                f"Error: Missing picklist JSON for video {stem!r}: expected {json_path}"
            )
        picklists_nested = _load_picklists_nested_from_json(json_path)
        per_video_cache = os.path.join(base_cache, stem)
        segments, flat_picklist, video_name = _process_single_video_to_segments(
            video_path,
            picklists_nested,
            clip_model,
            processor,
            per_video_cache,
            manual_labels_dir,
            require_manual_label_csv=True,
            compact_frame_indexing=compact_frame_indexing,
            threshold=threshold,
            frame_skip=frame_skip,
            htk_model_dir=htk_model_dir,
            aruco_config_path=aruco_config_path,
            verbose=verbose,
        )
        video_segments[video_name] = (segments, flat_picklist)

    trainer = WeakSupervisedTrainer(
        ilr_epochs=config.get("ilr_epochs", 500),
        initial_temp=config.get("initial_temp", 1.0),
        temp_decay=config.get("temp_decay", "exponential"),
        decay_rate=config.get("decay_rate", 0.99),
        random_seed=config.get("random_seed", 42),
        variance_eps=config.get("variance_eps", 1e-6),
        bad_swap_cool_divisor=config.get("bad_swap_cool_divisor", 50.0),
        detect_empty=config.get("detect_empty", False),
        min_frames_per_cluster=config.get("min_frames_per_cluster", 3),
        ilr_allow_cross_round_swaps=bool(config.get("ilr_allow_cross_round_swaps", False)),
    )
    use_cv = bool(config.get("use_cluster_voting", False))
    _init_vote_csv = (
        os.path.join(base_output_dir, "initial_cluster_voting.csv") if use_cv else None
    )
    trainer.fit(
        video_segments,
        verbose=verbose,
        skip_ilr=bool(config.get("skip_ilr", False)),
        initial_cluster_voting_csv=_init_vote_csv,
        use_cluster_voting=use_cv,
    )

    stem_order = [os.path.splitext(os.path.basename(vp))[0] for vp in video_paths]
    save_model(
        trainer,
        config,
        base_output_dir,
        embedded_video_stems_override=stem_order,
    )
    if verbose:
        print(f"\n✓ Model saved to {base_output_dir}")
        print(f"  embedded_video_stems: {stem_order}")

    return trainer


def run_video_training(
    video_path: str,
    picklist: List[str],
    base_output_dir: str,
    config: Dict[str, Any],
    threshold: float = 50.0,
    frame_skip: int = 4,
    verbose: bool = True,
    htk_model_dir: Optional[str] = None,
    aruco_config_path: Optional[str] = None,
    manual_labels_dir: Optional[str] = None,
    video_config_path: Optional[str] = None,
    compact_frame_indexing: str = "opencv0",
):
    """
    Run weakly supervised training from video.
    
    If video_config_path is provided, loads picklists from JSON config.
    Otherwise uses the provided picklist parameter.
    """
    # Validate video path
    if not Path(video_path).exists():
        raise SystemExit(f"Error: Video file not found: {video_path}")
    
    # Load video config if provided
    if video_config_path:
        if not os.path.exists(video_config_path):
            raise SystemExit(f"Error: Video config file not found: {video_config_path}")
        with open(video_config_path, "r") as f:
            video_config = json.load(f)
        picklists_nested = video_config.get("picklists", [picklist] if picklist else [[]])
    else:
        picklists_nested = [picklist] if picklist else [[]]
    
    # Set random seeds
    random.seed(config.get("random_seed", 42))
    np.random.seed(config.get("random_seed", 42))
    torch.manual_seed(config.get("random_seed", 42))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create output directory
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 60)
        print("WEAKLY SUPERVISED TRAINING")
        print("=" * 60)
        print(f"Video: {video_path}")
        print(f"Picklists: {picklists_nested}")
        print(f"Output: {base_output_dir}")
    
    # Load CLIP model
    clip_model = AutoModel.from_pretrained(MODEL)
    clip_model.eval()
    if device == "cuda":
        clip_model = clip_model.to(device)
    processor = AutoProcessor.from_pretrained(MODEL)
    
    if verbose:
        print(f"✓ CLIP model loaded (device: {device})")
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cache_dir = os.path.join(base_output_dir, ".cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    segments, flat_picklist, _ = _process_single_video_to_segments(
        video_path,
        picklists_nested,
        clip_model,
        processor,
        cache_dir,
        manual_labels_dir,
        require_manual_label_csv=False,
        compact_frame_indexing=compact_frame_indexing,
        threshold=threshold,
        frame_skip=frame_skip,
        htk_model_dir=htk_model_dir,
        aruco_config_path=aruco_config_path,
        verbose=verbose,
    )
    
    # Train weak supervision model
    trainer = WeakSupervisedTrainer(
        ilr_epochs=config.get("ilr_epochs", 500),
        initial_temp=config.get("initial_temp", 1.0),
        temp_decay=config.get("temp_decay", "exponential"),
        decay_rate=config.get("decay_rate", 0.99),
        random_seed=config.get("random_seed", 42),
        variance_eps=config.get("variance_eps", 1e-6),
        bad_swap_cool_divisor=config.get("bad_swap_cool_divisor", 50.0),
        detect_empty=config.get("detect_empty", False),
        min_frames_per_cluster=config.get("min_frames_per_cluster", 3),
        ilr_allow_cross_round_swaps=bool(config.get("ilr_allow_cross_round_swaps", False)),
    )
    use_cv = bool(config.get("use_cluster_voting", False))
    _init_vote_csv = (
        os.path.join(base_output_dir, "initial_cluster_voting.csv") if use_cv else None
    )
    trainer.fit(
        {video_name: (segments, flat_picklist)},
        verbose=verbose,
        skip_ilr=bool(config.get("skip_ilr", False)),
        initial_cluster_voting_csv=_init_vote_csv,
        use_cluster_voting=use_cv,
    )
    
    # Save model (record this video stem so incremental runs can skip duplicates)
    save_model(trainer, config, base_output_dir, append_embedded_video_stem=video_name)
    
    if verbose:
        print(f"\n✓ Model saved to {base_output_dir}")
    
    return trainer


def run_incremental_training(
    video_path: str,
    picklist: Optional[List[str]],
    model_dir: str,
    beta: float = 0.9,
    threshold: float = 100.0,
    frame_skip: int = 4,
    verbose: bool = True,
    htk_model_dir: Optional[str] = None,
    aruco_config_path: Optional[str] = None,
    manual_labels_dir: Optional[str] = None,
    compact_frame_indexing: str = "opencv0",
    video_config_path: Optional[str] = None,
    picklist_json_dir: Optional[str] = None,
    force_reembed: bool = False,
    equal_video_weight: bool = False,
    ilr_epochs_override: Optional[int] = None,
    random_seed_override: Optional[int] = None,
):
    """
    Update centroids with a new video using ``fit_iterative`` (permutation + EWMA
    or equal weight per video).

    Loads centroids/stds from ``model_dir``, extracts carry segments from
    ``video_path``, then runs ``WeakSupervisedTrainer.fit_iterative`` and saves
    back to the same ``model_dir``.

    Picklist can be omitted if ``video_config_path`` or ``picklist_json_dir`` is set
    (same JSON format as training). The model metadata stores ``embedded_video_stems``
    so the same video is not processed twice unless ``force_reembed`` is True.

    When ``equal_video_weight`` is True, each centroid is the spherical mean of the
    per-(label, video) means across contributing videos (``beta`` is ignored).

    Args:
        ilr_epochs_override: If set, override the ILR epochs from model config for this run.
        random_seed_override: If set, override the random seed from model config for this run.
    """
    meta_path = os.path.join(model_dir, "model_metadata.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing model metadata: {meta_path}")

    with open(meta_path, "r") as f:
        metadata = json.load(f)
    save_config = metadata.get("config", {})

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    embedded_stems: List[str] = list(metadata.get("embedded_video_stems", []))
    if video_name in embedded_stems and not force_reembed:
        raise SystemExit(
            f"Error: video stem {video_name!r} is already recorded in this model's "
            f"embedded_video_stems. Use --force-reembed to run incremental on it again, "
            f"or use a different video."
        )

    picklist_resolved, picklist_json_used = _resolve_incremental_picklist(
        video_path,
        picklist,
        video_config_path,
        picklist_json_dir,
    )
    picklist = picklist_resolved

    trainer = load_weak_trainer(
        model_dir,
        ilr_epochs_override=ilr_epochs_override,
        random_seed_override=random_seed_override,
    )

    effective_seed = random_seed_override if random_seed_override is not None else save_config.get("random_seed", 42)
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print("=" * 60)
        print("INCREMENTAL UPDATE (fit_iterative)")
        print("=" * 60)
        print(f"Model dir: {model_dir}")
        print(f"Video: {video_path}")
        if picklist_json_used:
            print(f"Picklist JSON: {picklist_json_used}")
        print(f"Picklist (flat, {len(picklist)} carries): {picklist}")
        if equal_video_weight:
            print("Update mode: equal weight per video (beta ignored)")
        else:
            print(f"Beta (weight on previous centroid): {beta}")
        if ilr_epochs_override is not None:
            print(f"ILR epochs override: {ilr_epochs_override} (model default: {save_config.get('ilr_epochs', 500)})")
        if random_seed_override is not None:
            print(f"Random seed override: {random_seed_override} (model default: {save_config.get('random_seed', 42)})")

    clip_model = AutoModel.from_pretrained(MODEL)
    clip_model.eval()
    if device == "cuda":
        clip_model = clip_model.to(device)
    processor = AutoProcessor.from_pretrained(MODEL)

    annotation_path = None
    if manual_labels_dir is not None:
        annotation_path = os.path.join(manual_labels_dir, f"{video_name}.csv")

    cache_dir = os.path.join(model_dir, ".incremental_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    carry_intervals = None
    if annotation_path is not None and os.path.isfile(annotation_path):
        cap_i = cv2.VideoCapture(video_path)
        n_total = int(cap_i.get(cv2.CAP_PROP_FRAME_COUNT)) if cap_i.isOpened() else 0
        cap_i.release()
        carry_intervals = carry_with_pipeline_frame_intervals_1based(
            annotation_path,
            n_total,
            frame_indexing=compact_frame_indexing,
        )

    def _state_detect(video_path, embeddings, frame_numbers, fps):
        if annotation_path is not None and os.path.exists(annotation_path):
            cap_d = cv2.VideoCapture(video_path)
            nfr = int(cap_d.get(cv2.CAP_PROP_FRAME_COUNT)) if cap_d.isOpened() else 0
            cap_d.release()
            dur = (nfr / fps) if fps and nfr else None
            return load_state_labels_auto(
                annotation_path,
                fps=float(fps),
                frame_indexing=compact_frame_indexing,
                video_duration_sec=dur,
            )
        return detect_states_from_video(
            video_path,
            embeddings,
            frame_numbers,
            fps,
            htk_model_dir=htk_model_dir,
            aruco_config_path=aruco_config_path,
            frame_skip=frame_skip,
            blur_threshold=threshold,
            clip_model=clip_model,
            clip_processor=processor,
            verbose=verbose,
        )

    video_embeddings, _, _, state_results, embedding_frame_indices = process_video_frames(
        video_path,
        picklist[0],
        clip_model,
        processor,
        cache_dir,
        threshold=threshold,
        frame_skip=frame_skip,
        state_detection_func=_state_detect,
        verbose=verbose,
        allowed_frame_intervals_1based=carry_intervals,
    )

    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
    else:
        fps = 30.0

    segments = _group_frames_into_segments(
        video_embeddings,
        embedding_frame_indices,
        state_results,
        fps,
        video_name,
    )

    if len(segments) != len(picklist):
        if verbose:
            print(
                f"WARNING: segments ({len(segments)}) != picklist ({len(picklist)}); "
                "trimming/padding like full train."
            )
        if len(segments) > len(picklist):
            segments = segments[: len(picklist)]
        else:
            while len(segments) < len(picklist):
                all_embs = np.vstack([seg.embeddings for seg in segments])
                n_per_seg = max(1, len(all_embs) // len(picklist))
                segments = []
                for i in range(len(picklist)):
                    start_idx = i * n_per_seg
                    end_idx = start_idx + n_per_seg if i < len(picklist) - 1 else len(all_embs)
                    segments.append(
                        Segment(
                            segment_id=i,
                            embeddings=all_embs[start_idx:end_idx],
                            video_id=video_name,
                        )
                    )
                break

    segment_raw = [seg.embeddings for seg in segments]
    _, _, chosen_perm = trainer.fit_iterative(
        segment_raw,
        picklist,
        beta=beta,
        verbose=verbose,
        video_id=video_name,
        equal_video_weight=equal_video_weight,
    )

    save_model(
        trainer,
        save_config,
        model_dir,
        append_embedded_video_stem=video_name,
    )

    inc_path = os.path.join(model_dir, "incremental_log.jsonl")
    with open(inc_path, "a") as f:
        f.write(
            json.dumps(
                {
                    "video": video_name,
                    "picklist": picklist,
                    "picklist_json": picklist_json_used,
                    "chosen_permutation": list(chosen_perm),
                    "beta": beta,
                    "equal_video_weight": equal_video_weight,
                    "ilr_epochs": trainer.ilr_epochs,
                    "random_seed": trainer.random_seed,
                }
            )
            + "\n"
        )

    if verbose:
        print(f"Incremental update saved to {model_dir}")
        print(f"  Chosen label order (aligned to segments): {chosen_perm}")

    return trainer


__all__ = [
    "run_video_training",
    "run_multi_video_training",
    "run_multi_video_training_from_cache",
    "run_incremental_training",
]
