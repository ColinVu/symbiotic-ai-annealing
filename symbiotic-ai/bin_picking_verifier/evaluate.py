"""Evaluate predictions against ground truth CSV."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_ground_truth(csv_path: Path) -> Dict[str, List[Optional[str]]]:
    """
    Parse ground_truth.csv where:
    - Row 0 = video names (picklist_041, ...)
    - Rows 1-N = pick index, each cell is the ground truth SKU
    
    Returns dict: {video_stem: [sku_pick0, sku_pick1, ...]}
    """
    df = pd.read_csv(csv_path, header=None)
    video_names = df.iloc[0].tolist()
    
    ground_truth: Dict[str, List[Optional[str]]] = {}
    for col_idx, video_name in enumerate(video_names):
        if pd.isna(video_name):
            continue
        video_stem = str(video_name).strip()
        picks: List[Optional[str]] = []
        for row_idx in range(1, len(df)):
            cell = df.iloc[row_idx, col_idx]
            if pd.isna(cell) or str(cell).strip() == "":
                picks.append(None)
            else:
                picks.append(str(cell).strip())
        ground_truth[video_stem] = picks
    
    return ground_truth


def evaluate_predictions(
    prediction_dir: Path,
    ground_truth: Dict[str, List[Optional[str]]],
) -> Dict[str, any]:
    """Compare predictions to ground truth, return metrics."""
    
    total_picks = 0
    correct = 0
    incorrect = 0
    missing_gt = 0
    missing_pred = 0
    
    results_per_video = {}
    
    for video_stem, gt_picks in ground_truth.items():
        pred_file = prediction_dir / f"{video_stem}.json"
        if not pred_file.exists():
            logger.warning("No prediction file for %s", video_stem)
            missing_pred += len([p for p in gt_picks if p is not None])
            continue
        
        pred_data = json.loads(pred_file.read_text(encoding="utf-8"))
        predictions = pred_data.get("predictions", [])
        
        video_correct = 0
        video_incorrect = 0
        video_missing_gt = 0
        
        for pick_idx, gt_sku in enumerate(gt_picks):
            if gt_sku is None:
                continue
            
            total_picks += 1
            
            if pick_idx >= len(predictions):
                missing_pred += 1
                video_missing_gt += 1
                continue
            
            pred = predictions[pick_idx]
            pred_sku = pred.get("predicted_sku")
            
            if pred_sku == gt_sku:
                correct += 1
                video_correct += 1
            else:
                incorrect += 1
                video_incorrect += 1
                logger.debug(
                    "%s pick %d: predicted %s, GT %s",
                    video_stem,
                    pick_idx,
                    pred_sku,
                    gt_sku,
                )
        
        video_total = video_correct + video_incorrect + video_missing_gt
        video_acc = video_correct / video_total if video_total > 0 else 0.0
        results_per_video[video_stem] = {
            "correct": video_correct,
            "incorrect": video_incorrect,
            "missing": video_missing_gt,
            "total": video_total,
            "accuracy": video_acc,
        }
    
    overall_accuracy = correct / total_picks if total_picks > 0 else 0.0
    
    return {
        "total_picks": total_picks,
        "correct": correct,
        "incorrect": incorrect,
        "missing_gt": missing_gt,
        "missing_pred": missing_pred,
        "overall_accuracy": overall_accuracy,
        "per_video": results_per_video,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions vs ground truth")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Directory containing prediction JSON files",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Path to ground_truth.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional: save evaluation results to JSON",
    )
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    gt = load_ground_truth(args.ground_truth)
    logger.info("Loaded ground truth for %d videos", len(gt))
    
    results = evaluate_predictions(args.predictions, gt)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total picks:         {results['total_picks']}")
    print(f"Correct:             {results['correct']}")
    print(f"Incorrect:           {results['incorrect']}")
    print(f"Missing GT:          {results['missing_gt']}")
    print(f"Missing predictions: {results['missing_pred']}")
    print(f"Overall Accuracy:    {results['overall_accuracy']:.2%}")
    print("=" * 60)
    
    print("\nPer-video breakdown:")
    for video_stem, metrics in sorted(results["per_video"].items()):
        print(
            f"  {video_stem}: {metrics['correct']}/{metrics['total']} = {metrics['accuracy']:.1%}"
        )
    
    if args.output:
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved detailed results to {args.output}")


if __name__ == "__main__":
    main()
