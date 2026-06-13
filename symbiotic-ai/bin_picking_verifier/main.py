"""CLI entrypoint for industrial bin-picking verification inference."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from tqdm import tqdm

from config import VerifierConfig
from debug_overlay import draw_overlay, write_debug_frame
from depth_tracker import DepthEstimator
from evaluate import load_ground_truth
from grid_tracker import GridTracker
from hand_tracker import HandTracker
from io_utils import ArucoMap, load_picklist_json, load_state_segments, pick_segments_only, video_stem_to_picklist_paths
from video_reader import VideoReader
from visualizer import FrameVisualizer
from voting_logic import PickPrediction, predict_for_segments

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Industrial bin-picking verification from egocentric video")
    p.add_argument("--videos", type=Path, required=True, help="Directory containing picklist videos (.mp4/.MP4)")
    p.add_argument("--state-csvs", type=Path, required=True, help="Directory of state CSV files")
    p.add_argument("--picklist-jsons", type=Path, required=True, help="Directory of picklist JSON files")
    p.add_argument("--aruco-map", type=Path, required=True, help="JSON map marker_id -> SKU + row/col")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory for prediction JSON files")
    p.add_argument("--debug", action="store_true", help="Write per-pick debug annotated frame")
    p.add_argument(
        "--visualized",
        action="store_true",
        help="Real-time OpenCV window: ArUcos, hand landmarks, palm gate, depth plane-break",
    )
    p.add_argument("--ground-truth", type=Path, help="Optional: ground_truth.csv for live per-video accuracy")
    p.add_argument("--workers", type=int, default=1, help="Reserved for parallel processing; currently unused")
    p.add_argument(
        "--depth-model",
        type=str,
        default=None,
        help="HuggingFace depth model id (default: depth-anything/Depth-Anything-V2-Small-hf)",
    )
    p.add_argument(
        "--depth-device",
        type=str,
        default="auto",
        help="Torch device for depth model: auto, cpu, cuda, mps",
    )
    p.add_argument(
        "--no-depth",
        action="store_true",
        help="Disable palm/depth gates; use legacy 2D point-in-polygon scoring",
    )
    p.add_argument(
        "--depth-stride",
        type=int,
        default=None,
        metavar="N",
        help="Process every Nth frame during PICK segments (same as cfg.frame_stride; default from config)",
    )
    return p.parse_args()


def _find_state_csv(stem: str, state_dir: Path) -> Optional[Path]:
    candidate = state_dir / f"{stem}.csv"
    if candidate.exists():
        return candidate
    return None


def _find_picklist_json(stem: str, json_dir: Path) -> Optional[Path]:
    for candidate in video_stem_to_picklist_paths(stem, json_dir):
        if candidate.exists():
            return candidate
    return None


def _prediction_to_json_obj(pred: PickPrediction) -> Dict[str, Any]:
    return {
        "pick_index": pred.pick_index,
        "picklist_block": pred.picklist_block,
        "segment_frames": [pred.segment_frames[0], pred.segment_frames[1]],
        "voting_frames_used": pred.voting_frames_used,
        "predicted_sku": pred.predicted_sku,
        "predicted_marker_id": pred.predicted_marker_id,
        "score": pred.score,
        "second_best": pred.second_best,
        "fallback_to_set_membership": pred.fallback_to_set_membership,
        "occluded_baseline_markers": pred.occluded_baseline_markers,
        "reason": pred.reason,
    }


def _write_debug_for_prediction(
    video_reader: VideoReader,
    grid_tracker: GridTracker,
    hand_tracker: HandTracker,
    output_dir: Path,
    video_stem: str,
    pred: PickPrediction,
) -> None:
    dbg = pred.debug or {}
    voting_frames = dbg.get("voting_frames") or []
    if not voting_frames:
        return

    frame_idx = int(voting_frames[len(voting_frames) // 2])
    frame = video_reader.read_frame(frame_idx)
    if frame is None:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = grid_tracker.detect(gray)
    polygons = grid_tracker.build_grid_polygons(detections)
    if not polygons:
        return

    ip = hand_tracker.get_interaction_point(frame)
    overlay = draw_overlay(
        frame=frame,
        polygons=polygons,
        interaction_point=ip,
        detected_marker_ids=detections.keys(),
        predicted_sku=pred.predicted_sku,
        header_text=f"pick={pred.pick_index} pred={pred.predicted_sku} score={pred.score:.2f}",
    )
    write_debug_frame(output_dir=output_dir, video_stem=video_stem, pick_index=pred.pick_index, image=overlay)


def process_video(
    video_path: Path,
    args: argparse.Namespace,
    cfg: VerifierConfig,
    aruco_map: ArucoMap,
    depth_estimator: Optional[DepthEstimator],
    gt_data: Optional[Dict[str, List[Optional[str]]]] = None,
) -> Optional[Path]:
    stem = video_path.stem
    state_csv = _find_state_csv(stem, args.state_csvs)
    if state_csv is None:
        logger.warning("No state CSV for %s", video_path.name)
        return None

    picklist_json_path = _find_picklist_json(stem, args.picklist_jsons)
    if picklist_json_path is None:
        logger.warning("No picklist JSON for %s", video_path.name)
        return None

    picklist_obj = load_picklist_json(picklist_json_path)
    picklists = picklist_obj.get("picklists", [])
    if not isinstance(picklists, list):
        logger.warning("Invalid picklists in %s", picklist_json_path)
        return None

    with VideoReader(video_path, target_height=cfg.target_height) as vr:
        assert vr.meta is not None
        segments = load_state_segments(state_csv, fps=vr.meta.fps, total_frames=vr.meta.frame_count)
        pick_segments = pick_segments_only(segments)

        if not pick_segments:
            logger.warning("No PICK segments for %s", video_path.name)

        grid_tracker = GridTracker(aruco_map=aruco_map, config=cfg)
        hand_tracker = HandTracker()

        viz: Optional[FrameVisualizer] = None
        viz_cb = None
        if args.visualized:
            viz = FrameVisualizer(cfg=cfg, subtitle=video_path.name)

            def viz_cb(frame_idx, frame_bgr, det, polygons, hand_data):  # type: ignore[no-untyped-def]
                viz.draw_vote_frame(frame_idx, frame_bgr, det, polygons, hand_data, aruco_map)
                return bool(viz.quit_requested)

        try:
            predictions = predict_for_segments(
                video_reader=vr,
                pick_segments=pick_segments,
                grid_tracker=grid_tracker,
                hand_tracker=hand_tracker,
                aruco_map=aruco_map,
                picklists=picklists,
                cfg=cfg,
                visualization_callback=viz_cb,
                depth_estimator=depth_estimator,
            )

            output = {
                "video": video_path.name,
                "fps": vr.meta.fps,
                "frame_size_processed": [vr.meta.width_proc, vr.meta.height_proc],
                "scale_to_processed": vr.meta.scale,
                "picklists": picklists,
                "predictions": [_prediction_to_json_obj(p) for p in predictions],
            }

            args.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.output_dir / f"{stem}.json"
            out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

            if args.debug:
                for pred in predictions:
                    try:
                        _write_debug_for_prediction(vr, grid_tracker, hand_tracker, args.output_dir, stem, pred)
                    except Exception as e:
                        logger.warning("Debug frame failed video=%s pick=%d: %s", video_path.name, pred.pick_index, e)

            logger.info("Wrote %s", out_path)
            
            # Immediate accuracy check if ground truth available
            if gt_data and stem in gt_data:
                gt_picks = gt_data[stem]
                correct = 0
                total = 0
                for pick_idx, pred in enumerate(predictions):
                    if pick_idx < len(gt_picks) and gt_picks[pick_idx] is not None:
                        total += 1
                        if pred.predicted_sku == gt_picks[pick_idx]:
                            correct += 1
                
                accuracy = correct / total if total > 0 else 0.0
                print(f"  ✓ {stem}: {correct}/{total} correct = {accuracy:.1%}")
            
            return out_path
        finally:
            if viz is not None:
                viz.close()
            hand_tracker.close()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = VerifierConfig()
    if args.no_depth:
        cfg.use_depth_gates = False
    if args.depth_stride is not None:
        cfg.frame_stride = max(1, int(args.depth_stride))
    if args.depth_model:
        cfg.depth_model_id = args.depth_model
    cfg.depth_device = args.depth_device

    aruco_map = ArucoMap.from_json_path(args.aruco_map)

    depth_estimator: Optional[DepthEstimator] = None
    if cfg.use_depth_gates:
        depth_estimator = DepthEstimator(
            model_id=cfg.depth_model_id,
            device=cfg.depth_device,
            infer_size=cfg.depth_infer_size,
        )
        logger.info(
            "Depth gates enabled: model=%s device=%s frame_stride=%d",
            cfg.depth_model_id,
            cfg.depth_device,
            cfg.frame_stride,
        )
    else:
        logger.info("Depth gates disabled (--no-depth); using legacy 2D scoring")
    
    # Load ground truth if provided for live accuracy feedback
    gt_data: Optional[Dict[str, List[Optional[str]]]] = None
    if args.ground_truth and args.ground_truth.exists():
        gt_data = load_ground_truth(args.ground_truth)
        logger.info("Loaded ground truth for %d videos (live accuracy enabled)", len(gt_data))

    videos = sorted([p for p in args.videos.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"])
    if not videos:
        logger.warning("No mp4 files found in %s", args.videos)
        return

    # Track cumulative stats
    cumulative_correct = 0
    cumulative_total = 0

    for video_path in tqdm(videos, desc="videos"):
        try:
            process_video(
                video_path=video_path,
                args=args,
                cfg=cfg,
                aruco_map=aruco_map,
                depth_estimator=depth_estimator,
                gt_data=gt_data,
            )
            
            # Accumulate stats if ground truth available
            if gt_data:
                stem = video_path.stem
                if stem in gt_data:
                    out_path = args.output_dir / f"{stem}.json"
                    if out_path.exists():
                        pred_data = json.loads(out_path.read_text(encoding="utf-8"))
                        gt_picks = gt_data[stem]
                        for idx, pred in enumerate(pred_data["predictions"]):
                            if idx < len(gt_picks) and gt_picks[idx] is not None:
                                cumulative_total += 1
                                if pred["predicted_sku"] == gt_picks[idx]:
                                    cumulative_correct += 1
        except Exception as e:
            logger.exception("Failed processing %s: %s", video_path, e)
    
    # Print final cumulative summary
    if gt_data and cumulative_total > 0:
        print(f"\n{'='*60}")
        print(f"CUMULATIVE: {cumulative_correct}/{cumulative_total} = {cumulative_correct/cumulative_total:.1%}")
        print(f"{'='*60}")

    if args.visualized:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()
