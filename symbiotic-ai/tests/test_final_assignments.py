"""Ground-truth columns in saved annealing assignments."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from symbiote_weak_generalized.core.config import DEFAULT_CONFIG
from symbiote_weak_generalized.experiments.assignment_score import write_final_assignments_csv


class TestFinalAssignments(unittest.TestCase):
    def test_default_ground_truth_points_to_project_data(self):
        project_root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            Path(DEFAULT_CONFIG["ground_truth_csv"]),
            project_root / "ground_truth.csv",
        )
        self.assertTrue(Path(DEFAULT_CONFIG["ground_truth_csv"]).is_file())

    def test_writer_includes_ordered_ground_truth_and_hits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth = root / "ground_truth.csv"
            with truth.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["picklist_001", "picklist_002"])
                writer.writerow(["c11", "d11"])
                writer.writerow(["c12", ""])

            trainer = SimpleNamespace(last_refined_labels={
                ("picklist_001", 0): "c11",
                ("picklist_001", 1): "c11",
                ("picklist_002", 0): "d11",
            })
            output = root / "final_assignments.csv"
            count = write_final_assignments_csv(
                str(output), trainer, ground_truth_csv=str(truth),
                video_stems_order=["picklist_001", "picklist_002"],
            )
            with output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

            self.assertEqual(count, 3)
            self.assertEqual(
                [(row["ground_truth"], row["hit"]) for row in rows],
                [("c11", "1"), ("c12", "0"), ("d11", "1")],
            )

    def test_missing_ground_truth_fails_instead_of_writing_empty_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trainer = SimpleNamespace(last_refined_labels={("picklist_001", 0): "c11"})
            output = root / "final_assignments.csv"
            with self.assertRaises(FileNotFoundError):
                write_final_assignments_csv(
                    str(output), trainer,
                    ground_truth_csv=str(root / "missing.csv"),
                )
            self.assertFalse(output.exists())

    def test_missing_video_column_fails_instead_of_writing_empty_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth = root / "ground_truth.csv"
            truth.write_text("picklist_002\nd11\n", encoding="utf-8")
            trainer = SimpleNamespace(last_refined_labels={("picklist_001", 0): "c11"})
            output = root / "final_assignments.csv"
            with self.assertRaisesRegex(ValueError, "picklist_001"):
                write_final_assignments_csv(
                    str(output), trainer, ground_truth_csv=str(truth),
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
