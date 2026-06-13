#!/usr/bin/env python3
"""Analyze ablation test results and generate comparison report."""

import json
import re
from pathlib import Path
import pandas as pd


def parse_tune_log(log_path: Path) -> dict:
    """Extract metrics from hmm_tune log file."""
    if not log_path.exists():
        return {
            "boundary_rmse": None,
            "macro_f1": None,
            "word_penalty": None,
            "grammar_scale": None,
            "recall_pick": None,
            "recall_carry_with": None,
            "recall_place": None,
            "recall_carry_empty": None,
            "coarse_f1": None,
            "interact_f1": None,
            "carry_f1": None,
            "samples": None,
        }
    
    text = log_path.read_text()
    
    # Extract best parameters line
    best_match = re.search(
        r"\[hmm_tune\] Best: p=([\d\.-]+), s=([\d\.-]+), boundary_rmse=([\d\.]+)s, macro_f1=([\d\.]+)",
        text
    )
    
    # Extract last grid line with detailed metrics
    grid_matches = list(re.finditer(
        r"\[hmm_tune\] p=\s*([\d\.-]+) s=\s*([\d\.]+) "
        r"boundary_rmse=([\d\.]+)s macro_f1=([\d\.]+) n=(\d+) "
        r"coarse/interact/carry=([\d\.]+)/([\d\.]+)/([\d\.]+) "
        r"R\(P/CW/PL/CE\)=([\d\.]+)/([\d\.]+)/([\d\.]+)/([\d\.]+)",
        text
    ))
    
    if not best_match:
        return {"error": "Could not parse best parameters"}
    
    result = {
        "word_penalty": float(best_match.group(1)),
        "grammar_scale": float(best_match.group(2)),
        "boundary_rmse": float(best_match.group(3)),
        "macro_f1": float(best_match.group(4)),
    }
    
    # Find the grid line that matches the best parameters
    for match in grid_matches:
        p_val = float(match.group(1))
        s_val = float(match.group(2))
        if abs(p_val - result["word_penalty"]) < 0.01 and abs(s_val - result["grammar_scale"]) < 0.01:
            result.update({
                "samples": int(match.group(5)),
                "coarse_f1": float(match.group(6)),
                "interact_f1": float(match.group(7)),
                "carry_f1": float(match.group(8)),
                "recall_pick": float(match.group(9)),
                "recall_carry_with": float(match.group(10)),
                "recall_place": float(match.group(11)),
                "recall_carry_empty": float(match.group(12)),
            })
            break
    
    return result


def load_tuning_grid(model_dir: str) -> pd.DataFrame:
    """Load the full tuning grid CSV if available."""
    grid_path = Path(model_dir) / "models" / "hmm_final" / "tuning_grid.csv"
    if grid_path.exists():
        return pd.read_csv(grid_path)
    return pd.DataFrame()


def analyze_results():
    """Generate comparison report from all ablation tests."""
    results_dir = Path("ablation_results")
    
    if not results_dir.exists():
        print("ERROR: ablation_results/ directory not found.")
        print("Run test_ablations.sh first.")
        return
    
    tests = [
        {
            "name": "1. No Constraints (29D)",
            "log": results_dir / "test1_no_constraints.log",
            "model_dir": "/models/htk_weak_test1_no_constraints",
            "description": "Full 29D features, no sequence constraints",
        },
        {
            "name": "2. With Constraints (29D)",
            "log": results_dir / "test2_with_constraints.log",
            "model_dir": "/models/htk_weak_test2_with_constraints",
            "description": "Full 29D features + sequence constraints (current)",
        },
        {
            "name": "3. No Color (17D)",
            "log": results_dir / "test3_no_color.log",
            "model_dir": "/models/htk_weak_test3_no_color",
            "description": "17D: no color histograms, with constraints",
        },
        {
            "name": "4. Single ARUCO (27D)",
            "log": results_dir / "test4_single_aruco.log",
            "model_dir": "/models/htk_weak_test4_single_aruco",
            "description": "27D: single ARUCO channel, with constraints",
        },
    ]
    
    rows = []
    for test in tests:
        metrics = parse_tune_log(test["log"])
        metrics["test_name"] = test["name"]
        metrics["description"] = test["description"]
        rows.append(metrics)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns for readability
    cols = [
        "test_name",
        "boundary_rmse",
        "macro_f1",
        "recall_pick",
        "recall_carry_with",
        "recall_place",
        "recall_carry_empty",
        "coarse_f1",
        "interact_f1",
        "carry_f1",
        "word_penalty",
        "grammar_scale",
        "samples",
    ]
    df = df[[c for c in cols if c in df.columns]]
    
    # Save full results
    csv_path = results_dir / "ablation_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved full comparison to: {csv_path}")
    print()
    
    # Print summary table
    print("=" * 120)
    print("ABLATION TEST RESULTS SUMMARY")
    print("=" * 120)
    print()
    
    # Key metrics table
    summary_cols = [
        "test_name",
        "boundary_rmse",
        "macro_f1",
        "recall_pick",
        "recall_carry_with",
        "recall_place",
        "recall_carry_empty",
    ]
    summary_df = df[summary_cols].copy()
    
    print(summary_df.to_string(index=False))
    print()
    print("=" * 120)
    
    # Analysis
    print("\nANALYSIS:")
    print("-" * 80)
    
    if df["macro_f1"].isna().all():
        print("⚠ No valid results found. Check log files for errors.")
        return
    
    best_f1_idx = df["macro_f1"].idxmax()
    best_boundary_idx = df["boundary_rmse"].idxmin()
    
    print(f"\n✓ Best Macro-F1: {df.loc[best_f1_idx, 'test_name']}")
    print(f"  F1={df.loc[best_f1_idx, 'macro_f1']:.4f}, "
          f"boundary={df.loc[best_f1_idx, 'boundary_rmse']:.2f}s")
    
    print(f"\n✓ Best Boundary RMSE: {df.loc[best_boundary_idx, 'test_name']}")
    print(f"  boundary={df.loc[best_boundary_idx, 'boundary_rmse']:.2f}s, "
          f"F1={df.loc[best_boundary_idx, 'macro_f1']:.4f}")
    
    # Check for class imbalance issues
    print("\n⚠ Class Balance Issues:")
    for idx, row in df.iterrows():
        recalls = [
            row.get("recall_pick", 0),
            row.get("recall_carry_with", 0),
            row.get("recall_place", 0),
            row.get("recall_carry_empty", 0),
        ]
        if any(pd.notna(recalls)):
            max_recall = max([r for r in recalls if pd.notna(r)])
            min_recall = min([r for r in recalls if pd.notna(r)])
            imbalance = max_recall - min_recall
            
            if imbalance > 0.5:
                print(f"  {row['test_name']}: High imbalance (Δ={imbalance:.3f})")
                print(f"    PICK={recalls[0]:.3f}, CW={recalls[1]:.3f}, "
                      f"PL={recalls[2]:.3f}, CE={recalls[3]:.3f}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("-" * 80)
    
    # Compare test 1 vs 2 (constraints effect)
    if not df.loc[0, "macro_f1"] is None and not df.loc[1, "macro_f1"] is None:
        f1_delta = df.loc[1, "macro_f1"] - df.loc[0, "macro_f1"]
        if f1_delta < -0.05:
            print("\n⚠ Sequence constraints HURT performance (F1 dropped by "
                  f"{abs(f1_delta):.3f})")
            print("  → Consider disabling constraints or checking label CSV quality")
        elif f1_delta > 0.05:
            print("\n✓ Sequence constraints HELP performance (F1 improved by "
                  f"{f1_delta:.3f})")
    
    # Check if color helps
    if len(df) >= 3 and not df.loc[1, "macro_f1"] is None and not df.loc[2, "macro_f1"] is None:
        f1_delta_color = df.loc[1, "macro_f1"] - df.loc[2, "macro_f1"]
        if f1_delta_color < -0.05:
            print("\n⚠ Color descriptors HURT performance (removing them improved "
                  f"F1 by {abs(f1_delta_color):.3f})")
            print("  → Color may be too noisy or causing overfitting")
        elif f1_delta_color > 0.05:
            print("\n✓ Color descriptors HELP performance (adding them improved "
                  f"F1 by {f1_delta_color:.3f})")
    
    # Check if 3-channel ARUCO helps
    if len(df) >= 4 and not df.loc[1, "macro_f1"] is None and not df.loc[3, "macro_f1"] is None:
        f1_delta_aruco = df.loc[1, "macro_f1"] - df.loc[3, "macro_f1"]
        if f1_delta_aruco < -0.05:
            print("\n⚠ 3-channel ARUCO HURT performance (single channel better by "
                  f"{abs(f1_delta_aruco):.3f})")
            print("  → Revert to single signed ARUCO channel")
        elif f1_delta_aruco > 0.05:
            print("\n✓ 3-channel ARUCO HELPS performance (improved F1 by "
                  f"{f1_delta_aruco:.3f})")
    
    # General recommendations
    print("\nGENERAL OBSERVATIONS:")
    avg_f1 = df["macro_f1"].mean()
    if avg_f1 < 0.3:
        print("  ⚠ All configurations show very low macro-F1 (<0.3)")
        print("    → This suggests a deeper issue:")
        print("      - Insufficient training data")
        print("      - Train/test distribution mismatch")
        print("      - Feature/label quality issues")
        print("      - Check: How many train videos? How many dev videos?")
    
    max_recall = df[["recall_pick", "recall_carry_with", "recall_place", 
                     "recall_carry_empty"]].max().max()
    min_recall = df[["recall_pick", "recall_carry_with", "recall_place", 
                     "recall_carry_empty"]].min().min()
    
    if max_recall - min_recall > 0.6:
        print(f"  ⚠ Severe class imbalance across all tests (Δ={max_recall - min_recall:.3f})")
        print("    → Check if training labels are balanced")
        print("    → Consider per-class weighting or data augmentation")
    
    print("\n" + "=" * 80)
    print()
    print("For detailed analysis:")
    print(f"  - Read logs in {results_dir}/")
    print("  - Check tuning_grid.csv in each model's hmm_final/ directory")
    print("  - Run inference on a sample video to inspect predictions")
    print()


if __name__ == "__main__":
    analyze_results()
