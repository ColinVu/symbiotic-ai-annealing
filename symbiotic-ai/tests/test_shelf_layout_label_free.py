"""Decision-time label independence, ambiguity gates, legal search, and reports."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from shelf_layout import DecisionEngine, SnapshotError
from shelf_layout.label_free import AmbiguitySettings
from shelf_layout.cli import main
from test_shelf_layout import fixture


class LabelFreeTests(unittest.TestCase):
    def engine(self, **settings):
        model, snapshot = fixture()
        for row in snapshot["observations"]:
            row.pop("inferred_sku")
        return DecisionEngine(snapshot, model, ambiguity=AmbiguitySettings(**settings))

    def test_unlabeled_default_separates_repeated_pair_without_replay(self):
        engine = self.engine()
        with patch.object(engine, "_predict", side_effect=AssertionError("label replay")):
            result = engine.recommend(max_moved=2)
        rec = result["recommendation"]
        self.assertEqual(result["metric_type"], "ambiguity_risk")
        self.assertNotIn("proxy_errors", rec)
        self.assertNotEqual(rec["layout"]["a"]["shelf"], rec["layout"]["b"]["shelf"])
        self.assertGreater(rec["predicted_risk_reduction"], 0)
        self.assertEqual(rec["moved_skus"], 2)
        self.assertTrue(any(c["moved_skus"] == 0 for c in result["candidates"]))
        for shelf in engine.data["capacities"]:
            slots = [loc["slot"] for loc in rec["layout"].values() if loc["shelf"] == shelf]
            self.assertEqual(len(slots), 3)
            self.assertEqual(len(set(slots)), 3)
        self.assertEqual(result, engine.recommend(max_moved=2))

    def test_inferred_labels_and_confidence_cannot_change_result(self):
        model, snapshot = fixture()
        unlabeled = copy.deepcopy(snapshot)
        changed = copy.deepcopy(snapshot)
        for row in unlabeled["observations"]:
            row.pop("inferred_sku")
        for row in changed["observations"]:
            row["inferred_sku"] = "not-an-inventory-sku"
            row["assignment_confidence"] = "ignored"
        results = [DecisionEngine(s, model).recommend(max_moved=2) for s in (snapshot, unlabeled, changed)]
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], results[2])
        # Retrospective labels remain a separate API and cannot enter a snapshot.
        changed["observations"][0]["true_label"] = "a"
        with self.assertRaises(SnapshotError):
            DecisionEngine(changed, model)

    def test_plausibility_gap_and_each_segment_votes_once(self):
        engine = self.engine()
        pair = engine.ambiguity_pairs[0]
        self.assertEqual(pair["support"], 3)
        self.assertTrue(pair["eligible"])
        model, snapshot = fixture()
        # Repeat frames without changing segment support.
        snapshot["observations"][0]["frames"] *= 20
        self.assertEqual(DecisionEngine(snapshot, model).ambiguity_pairs[0]["support"], 3)
        dominant = self.engine(max_score_gap=0.001)
        self.assertFalse(any(r["ambiguous"] for r in dominant.segment_evidence))
        # An orthogonal frame has two equally bad candidates (distance=1).
        for row in snapshot["observations"]:
            row["frames"] = [model.centroids[3].tolist()]
        poor = DecisionEngine(snapshot, model)
        self.assertFalse(any(r["ambiguous"] for r in poor.segment_evidence))
        self.assertEqual(poor.ambiguity_pairs, [])

    def test_single_ambiguity_does_not_trigger_with_fallback_disabled(self):
        model, snapshot = fixture()
        snapshot["observations"] = snapshot["observations"][:1]
        engine = DecisionEngine(snapshot, model)
        self.assertFalse(engine.ambiguity_pairs[0]["eligible"])
        self.assertEqual(engine.recommend(fallback_swap_budget=0)["transactions"], [])

    def test_geometry_fallback_is_bounded_and_only_for_sparse_history(self):
        model, snapshot = fixture()
        snapshot["observations"] = []
        result = DecisionEngine(snapshot, model).recommend(max_moved=2, fallback_swap_budget=1)
        self.assertEqual(result["recommendation"]["moved_skus"], 2)
        self.assertLessEqual(len(result["search"]["passes"][0]["fallback_pairs"]), 1)
        # Plenty of dominant observations do not invoke geometry fallback.
        snapshot["observations"] = [{"id": str(i), "revealed_at": 1, "observed_shelf": "s1",
                                     "frames": [model.centroids[2].tolist()]} for i in range(3)]
        result = DecisionEngine(snapshot, model).recommend(max_moved=2)
        self.assertEqual(result["transactions"], [])

    def test_destination_overlap_and_delta_match_full_risk(self):
        engine = self.engine()
        # Every candidate must pay for risks newly introduced on both shelves.
        for a in engine.labels:
            for b in engine.labels:
                if engine.baseline[a] == engine.baseline[b]:
                    continue
                layout = dict(engine.baseline)
                layout[a], layout[b] = layout[b], layout[a]
                delta = engine.score(layout)["ambiguity_risk"] - engine.score_current()["ambiguity_risk"]
                self.assertAlmostEqual(delta, engine._swap_proxy_delta(engine.baseline, a, b))
        # Equal pair risk everywhere: separating one pair cannot improve any layout.
        engine.pair_risk[:] = 1
        np.fill_diagonal(engine.pair_risk, 0)
        result = engine.recommend(move_cost=0, search_mode="all_pairs")
        self.assertEqual(result["transactions"], [])
        self.assertEqual(result["search"]["no_move_reason"], "no_surrogate_objective_improvement")

    def test_empty_future_and_invalid_settings(self):
        model, snapshot = fixture()
        for row in snapshot["observations"]:
            row["frames"] = []
        engine = DecisionEngine(snapshot, model)
        self.assertEqual(len(engine.diagnostics()["unscorable_segments"]), 3)
        self.assertEqual(engine.ambiguity_pairs, [])
        snapshot["observations"][0]["revealed_at"] = 3
        with self.assertRaises(SnapshotError):
            DecisionEngine(snapshot, model)
        with self.assertRaises(ValueError):
            self.engine(min_support=1)
        with self.assertRaises(ValueError):
            self.engine(geometry_scale=float("nan"))

    def test_transforms_applied_and_input_immutable(self):
        model, snapshot = fixture()
        original = copy.deepcopy(snapshot)
        from shelf_layout.engine import ModelData
        transformed = ModelData(model.labels, model.centroids, lambda x: np.tile(model.centroids[2], (len(x), 1)), "transform")
        engine = DecisionEngine(snapshot, transformed)
        self.assertFalse(any(r["ambiguous"] for r in engine.segment_evidence))
        engine.recommend()
        self.assertEqual(snapshot, original)

    def test_cli_default_reports_and_truth_are_separate(self):
        model, snapshot = fixture()
        for row in snapshot["observations"]:
            row.pop("inferred_sku")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "model_metadata.json").write_text(json.dumps({"centroid_labels": model.labels}))
            np.save(model_dir / "centroids.npy", model.centroids)
            sp = root / "snapshot.json"
            sp.write_text(json.dumps(snapshot))
            out = root / "out"
            args = ["recommend", "--snapshot", str(sp), "--model-dir", str(model_dir), "--output-dir", str(out)]
            self.assertEqual(main(args), 0)
            before = json.loads((out / "summary.json").read_text())
            truth = root / "truth.json"
            truth.write_text(json.dumps({r["id"]: "b" for r in snapshot["observations"]}))
            self.assertEqual(main(args + ["--truth-json", str(truth)]), 0)
            after = json.loads((out / "summary.json").read_text())
            self.assertEqual(before["recommendation"], after["recommendation"])
            self.assertIsNone(after["retrospective"]["candidate_accuracy"])
            self.assertNotIn("proxy_errors", (out / "summary.md").read_text())
            self.assertTrue((out / "ambiguity_evidence.csv").exists())
            snaps = root / "snapshots"
            snaps.mkdir()
            (snaps / "step.json").write_text(json.dumps(snapshot))
            self.assertEqual(main(["simulate", "--snapshots-dir", str(snaps), "--model-dir", str(model_dir),
                                   "--output-dir", str(out)]), 0)
            timeline = json.loads((out / "summary.json").read_text())
            self.assertEqual(timeline["decisions"][0]["metric_type"], "ambiguity_risk")

    def test_three_shelf_cycle_can_improve_when_no_swap_can(self):
        engine = self.engine()
        engine.pair_risk = np.array([
            [0, 0, 6, 8, 5, 9, 0, 8, 1], [0, 0, 1, 4, 4, 9, 3, 7, 4],
            [6, 1, 0, 1, 9, 8, 7, 9, 5], [8, 4, 1, 0, 3, 4, 0, 8, 4],
            [5, 4, 9, 3, 0, 1, 8, 5, 5], [9, 9, 8, 4, 1, 0, 0, 6, 4],
            [0, 3, 7, 0, 8, 0, 0, 0, 3], [8, 7, 9, 8, 5, 6, 0, 0, 5],
            [1, 4, 5, 4, 5, 4, 3, 5, 0]], dtype=float)
        options = dict(move_cost=0, max_moved=3, search_mode="all_pairs", fallback_swap_budget=0)
        self.assertEqual(engine.recommend(cycle_pool_size=0, **options)["transactions"], [])
        result = engine.recommend(cycle_pool_size=9, **options)
        self.assertEqual(result["recommendation"]["moved_skus"], 3)
        self.assertAlmostEqual(result["recommendation"]["objective_change"], -1)
        self.assertEqual(self.engine().recommend(max_moved=0)["transactions"], [])
