from __future__ import annotations

import unittest

import numpy as np

from symbiote_weak_generalized.experiments.evaluator import (
    _evaluate_segments,
    _extended_metrics,
    _predict_rows_from_embeddings,
    shelf_for_video_stem,
)


class _FakeModel:
    def __init__(self):
        self.centroids = {
            "c11": np.array([1.0, 0.0]),
            "c12": np.array([-1.0, 0.0]),
            "d21": np.array([0.0, 1.0]),
        }

    def _l2_normalize(self, vector):
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def cosine_distance(self, left, right):
        return 1.0 - float(
            np.dot(self._l2_normalize(left), self._l2_normalize(right))
        )

    def predict_top_k(self, embedding, k=3):
        if float(embedding[0]) > 0.5:
            labels = [("c11", 0.8), ("d21", 0.15), ("c12", 0.05)]
        else:
            labels = [("d21", 0.8), ("c11", 0.15), ("c12", 0.05)]
        return labels[:k]


class _FakeRecognizer:
    def __init__(self):
        self.model = _FakeModel()

    def _postprocess_embedding(self, embedding):
        return embedding


class TestEvaluatorMetrics(unittest.TestCase):
    def test_shelf_mapping(self):
        self.assertEqual(shelf_for_video_stem("picklist_101"), "c")
        self.assertEqual(shelf_for_video_stem("picklist_205"), "g")
        self.assertIsNone(shelf_for_video_stem("picklist_200"))

    def test_frame_segment_and_shelf_metrics(self):
        recognizer = _FakeRecognizer()
        rows = _predict_rows_from_embeddings(
            recognizer,
            [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
            [1, 2],
            shelf_prefix="c",
        )
        segments = _evaluate_segments([(1, 2)], ["c11"], rows)
        metrics = _extended_metrics(
            video_stem="picklist_101",
            intervals=[(1, 2)],
            expected_labels=["c11"],
            inference_rows=rows,
            segment_evals=segments,
        )

        self.assertEqual(metrics["frame"], {"hits": 1, "count": 2, "accuracy": 0.5})
        self.assertEqual(
            metrics["segment_top1"],
            {"hits": 1, "count": 1, "accuracy": 1.0},
        )
        self.assertEqual(
            metrics["shelf_constrained_segment_top1"],
            {"hits": 1, "count": 1, "accuracy": 1.0},
        )
        self.assertIn("c", metrics["by_shelf"])


if __name__ == "__main__":
    unittest.main()
