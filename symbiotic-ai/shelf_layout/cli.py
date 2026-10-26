"""CLI harness for the simulator-facing shelf decision API."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .engine import DecisionEngine, SnapshotError, load_model


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, fields: List[str], rows: List[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _layout_shelves(candidate: Mapping[str, Any]) -> Dict[str, str]:
    return {sku: value["shelf"] for sku, value in candidate["layout"].items()}


def _retrospective(engine: DecisionEngine, truth: Mapping[str, str], recommendation: Mapping[str, Any]) -> Dict[str, Any]:
    base = engine.retrospective(truth, engine.baseline)
    changed = engine.retrospective(truth, _layout_shelves(recommendation))
    fixed = new = 0
    for oid, sku in truth.items():
        old_ok = base["predictions"][oid] == sku
        new_ok = changed["predictions"][oid] == sku
        fixed += not old_ok and new_ok
        new += old_ok and not new_ok
    return {"baseline_errors": base["errors"], "candidate_errors": changed["errors"],
            "count": base["count"], "fixed": fixed, "new_errors": new}


def _report(engine: DecisionEngine, command: str, output: Path, result: Optional[Dict[str, Any]],
            truth: Optional[Mapping[str, str]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    diag = engine.diagnostics()
    rec = result["recommendation"] if result else None
    summary: Dict[str, Any] = {"mode": command, "step": engine.data["step"],
                               "snapshot_hash": engine.data["snapshot_hash"],
                               "model_hash": engine.model.fingerprint,
                               "observations": len(engine.observations),
                               "baseline": diag["baseline"], "recommendation": rec,
                               "settings": result["settings"] if result else None,
                               "search": result["search"] if result else None,
                               "interpretation": "In-simulation estimate on revealed observations; no new-video accuracy claim."}
    if engine.data.get("previous_decision") is not None:
        summary["previous_decision"] = engine.data["previous_decision"]
    if rec and truth is not None:
        summary["retrospective"] = _retrospective(engine, truth, rec)
    _write_json(output / "summary.json", summary)
    baseline_predictions = engine.score_current()["predictions"]
    proposed_layout = _layout_shelves(rec) if rec else engine.baseline
    proposed_predictions = (engine.score_current() if proposed_layout == engine.baseline
                            else engine.score(proposed_layout))["predictions"]
    sku_rows = []
    for sku, metrics in sorted(diag["per_sku"].items()):
        rows = [r for r in engine.observations if r["sku"] == sku]
        fixed = sum(baseline_predictions[r["id"]] != sku and proposed_predictions[r["id"]] == sku for r in rows)
        new = sum(baseline_predictions[r["id"]] == sku and proposed_predictions[r["id"]] != sku for r in rows)
        sku_rows.append({"sku": sku, "current_shelf": engine.baseline[sku],
                         "proposed_shelf": proposed_layout[sku], **metrics,
                         "proposed_proxy_errors": metrics["proxy_errors"] - fixed + new,
                         "fixed": fixed, "new_errors": new})
    _write_csv(output / "per_sku_metrics.csv",
               ["sku", "current_shelf", "proposed_shelf", "segments", "proxy_errors",
                "proposed_proxy_errors", "fixed", "new_errors", "own_distance_median", "own_distance_p90"],
               sku_rows)
    _write_csv(output / "confusing_pairs.csv",
               ["source", "target", "confusions", "source_segments", "directed_overlap_segments",
                "competitor_margin_p10", "centroid_cosine_distance", "same_shelf"],
               diag["confusing_pairs"])
    _write_csv(output / "confusion_evidence.csv",
               ["a", "b", "a_to_b", "b_to_a", "a_segments", "b_segments", "pair_count",
                "pair_rate", "same_shelf", "mean_assignment_confidence", "confidence_segments",
                "eligible", "exclusion_reasons"],
               [{**row, "exclusion_reasons": ";".join(row["exclusion_reasons"])}
                for row in diag["confusion_evidence"]])
    per_shelf = []
    for shelf in sorted(engine.data["capacities"]):
        rows = [r for r in engine.observations if proposed_layout[r["sku"]] == shelf]
        per_shelf.append({"shelf": shelf, "segments": len(rows),
                          "current_proxy_errors_for_destination_items": sum(baseline_predictions[r["id"]] != r["sku"] for r in rows),
                          "proposed_proxy_errors": sum(proposed_predictions[r["id"]] != r["sku"] for r in rows),
                          "fixed": sum(baseline_predictions[r["id"]] != r["sku"] and proposed_predictions[r["id"]] == r["sku"] for r in rows),
                          "new_errors": sum(baseline_predictions[r["id"]] == r["sku"] and proposed_predictions[r["id"]] != r["sku"] for r in rows)})
    _write_csv(output / "per_shelf_metrics.csv",
               ["shelf", "segments", "current_proxy_errors_for_destination_items", "proposed_proxy_errors", "fixed", "new_errors"], per_shelf)

    if result:
        candidate_rows = []
        for rank, candidate in enumerate(result["candidates"], 1):
            for sku, location in candidate["layout"].items():
                candidate_rows.append({"rank": rank, "objective": candidate["objective"],
                                       "proxy_errors": candidate["proxy_errors"], "moved_skus": candidate["moved_skus"],
                                       "sku": sku, "shelf": location["shelf"], "slot": location["slot"]})
        _write_csv(output / "candidate_layouts.csv",
                   ["rank", "objective", "proxy_errors", "moved_skus", "sku", "shelf", "slot"], candidate_rows)
        _write_csv(output / "move_transactions.csv",
                   ["sku", "from_shelf", "from_slot", "to_shelf", "to_slot"], result["transactions"])
        _write_csv(output / "decisions.csv",
                   ["step", "snapshot_hash", "baseline_proxy_errors", "proposed_proxy_errors", "moved_skus", "fixed", "new_errors"],
                   [{"step": result["step"], "snapshot_hash": result["snapshot_hash"],
                     "baseline_proxy_errors": result["baseline"]["errors"],
                     "proposed_proxy_errors": rec["proxy_errors"], "moved_skus": rec["moved_skus"],
                     "fixed": rec.get("fixed", 0), "new_errors": rec.get("new_errors", 0)}])

    lines = ["# Shelf decision simulation report", "", f"Step: {engine.data['step']}",
             f"Revealed segments: {len(engine.observations)}", "",
             f"Current proxy errors: {diag['baseline']['errors']}"]
    if rec:
        lines.extend([f"Proposed proxy errors: {rec['proxy_errors']}",
                      f"Items moved: {rec['moved_skus']}",
                      f"Proxy errors fixed: {rec.get('fixed', 0)}",
                      f"New proxy errors: {rec.get('new_errors', 0)}"])
        search = result["search"]
        lines.extend([f"Swap candidates generated: {search['total_generated_swaps']}",
                      f"Exact swap replays: {search['total_exact_swap_replays']}",
                      f"Fallback passes: {search['fallback_passes']}"])
        if rec["moved_skus"] == 0:
            lines.append(f"No-move reason: {search['no_move_reason']}")
    if "retrospective" in summary:
        ret = summary["retrospective"]
        lines.extend(["", "## Retrospective true-label analysis", "",
                      f"Current errors: {ret['baseline_errors']}; proposed errors: {ret['candidate_errors']}.",
                      f"Fixed: {ret['fixed']}; newly introduced: {ret['new_errors']}."])
    lines.extend(["", "## Interpretation", "",
                  "These are in-simulation estimates from existing observations. The decision used inferred SKU assignments, not hidden true labels. Moving bins may change future images; that effect belongs to the bin-change simulator.",
                  "", "## Reproducibility", "",
                  f"Snapshot SHA-256: `{engine.data['snapshot_hash']}`",
                  f"Model SHA-256: `{engine.model.fingerprint}`",
                  f"Settings: `{json.dumps(result['settings'], sort_keys=True) if result else 'analysis only'}`", ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _add_search_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--search-mode", choices=("confusion", "all_pairs"), default="confusion")
    parser.add_argument("--min-pair-segments", type=int, default=5)
    parser.add_argument("--min-pair-confusions", type=int, default=2)
    parser.add_argument("--min-pair-rate", type=float, default=0.20)
    parser.add_argument("--max-confusion-pairs", type=int, default=10)
    parser.add_argument("--partners-per-shelf", type=int, default=5)
    parser.add_argument("--fallback-swap-budget", type=int, default=32)


def _search_options(args: argparse.Namespace) -> Dict[str, Any]:
    return {"search_mode": args.search_mode, "min_pair_segments": args.min_pair_segments,
            "min_pair_confusions": args.min_pair_confusions, "min_pair_rate": args.min_pair_rate,
            "max_confusion_pairs": args.max_confusion_pairs,
            "partners_per_shelf": args.partners_per_shelf,
            "fallback_swap_budget": args.fallback_swap_budget}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Simulated shelf-layout decision engine")
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "analyze", "recommend"):
        sub = subs.add_parser(name)
        sub.add_argument("--snapshot", required=True, help="JSON snapshot with items and revealed observations")
        sub.add_argument("--model-dir", required=True, help="Saved centroid model directory")
        sub.add_argument("--output-dir", required=True)
        if name != "preflight":
            sub.add_argument("--truth-json", help="Optional ID-to-true-SKU mapping, used only after decisions")
        if name == "recommend":
            sub.add_argument("--move-cost", type=float, default=0.25)
            sub.add_argument("--max-moved", type=int, default=12)
            sub.add_argument("--cycle-pool-size", type=int, default=12)
            sub.add_argument("--swap-replay-limit", type=int, default=200)
            _add_search_options(sub)
    sim = subs.add_parser("simulate", help="Run decisions on simulator-provided snapshots in step order")
    sim.add_argument("--snapshots-dir", required=True)
    sim.add_argument("--model-dir", required=True)
    sim.add_argument("--output-dir", required=True)
    sim.add_argument("--move-cost", type=float, default=0.25)
    sim.add_argument("--max-moved", type=int, default=12)
    sim.add_argument("--cycle-pool-size", type=int, default=12)
    sim.add_argument("--swap-replay-limit", type=int, default=200)
    _add_search_options(sim)
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    try:
        model = load_model(args.model_dir)
        if args.command == "simulate":
            paths = sorted(Path(args.snapshots_dir).glob("*.json"))
            if not paths:
                raise SnapshotError("No snapshot JSON files found")
            snapshots = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
            if any(not isinstance(x, dict) or not isinstance(x.get("step"), int) for x in snapshots):
                raise SnapshotError("Every simulation snapshot needs an integer step")
            snapshots.sort(key=lambda x: x.get("step", -1))
            if len({x.get("step") for x in snapshots}) != len(snapshots):
                raise SnapshotError("Simulation snapshots have duplicate steps")
            decisions = []
            for snapshot in snapshots:
                engine = DecisionEngine(snapshot, model)
                result = engine.recommend(move_cost=args.move_cost, max_moved=args.max_moved,
                                          cycle_pool_size=args.cycle_pool_size,
                                          swap_replay_limit=args.swap_replay_limit,
                                          **_search_options(args))
                step_dir = output / f"step_{engine.data['step']:06d}"
                _report(engine, "recommend", step_dir, result, None)
                rec = result["recommendation"]
                decisions.append({"step": engine.data["step"], "snapshot_hash": result["snapshot_hash"],
                                  "baseline_proxy_errors": result["baseline"]["errors"],
                                  "proposed_proxy_errors": rec["proxy_errors"],
                                  "moved_skus": rec["moved_skus"],
                                  "previous_decision": json.dumps(engine.data.get("previous_decision"), sort_keys=True)})
            output.mkdir(parents=True, exist_ok=True)
            _write_csv(output / "decisions.csv", ["step", "snapshot_hash", "baseline_proxy_errors",
                                                  "proposed_proxy_errors", "moved_skus", "previous_decision"], decisions)
            _write_json(output / "summary.json", {"mode": "simulate", "steps": len(decisions),
                                                  "decisions": decisions,
                                                  "interpretation": "Existing-video simulation; each decision used its own revealed snapshot."})
            return 0
        snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
        engine = DecisionEngine(snapshot, model)
        if args.command == "preflight":
            output.mkdir(parents=True, exist_ok=True)
            _write_json(output / "preflight.json", {"status": "ok", "step": engine.data["step"],
                                                     "skus": len(engine.labels), "observations": len(engine.observations),
                                                     "snapshot_hash": engine.data["snapshot_hash"],
                                                     "model_hash": model.fingerprint})
            return 0
        result = (engine.recommend(move_cost=args.move_cost, max_moved=args.max_moved,
                                   cycle_pool_size=args.cycle_pool_size,
                                   swap_replay_limit=args.swap_replay_limit,
                                   **_search_options(args))
                  if args.command == "recommend" else None)
        truth = json.loads(Path(args.truth_json).read_text(encoding="utf-8")) if args.truth_json else None
        if truth is not None and result is None:
            raise SnapshotError("--truth-json is supported only for recommend")
        _report(engine, args.command, output, result, truth)
        return 0
    except (SnapshotError, ValueError, OSError, json.JSONDecodeError) as exc:
        output.mkdir(parents=True, exist_ok=True)
        _write_json(output / "missing_inputs.json", {"status": "error", "message": str(exc)})
        print(f"shelf-layout: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
