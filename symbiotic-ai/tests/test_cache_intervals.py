"""Cache interval filtering and cache-only skip accounting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.cv_fixtures import (
    FRAME_SKIP,
    N_FRAMES,
    build_synthetic_dataset,
    write_cache_for_video,
    write_compact_labels,
    write_dummy_video,
    write_picklist_json,
)
from symbiote_weak_generalized.pipelines.video_training import (
    _candidate_frame_indices_carry_and_skip,
    _glob_cached_frame_indices_first_label,
    _process_single_video_from_cache,
)
from symbiote_weak_generalized.state_detection.compact_timeline import (
    carry_with_pipeline_frame_intervals_1based,
)


class TestCacheIntervals(unittest.TestCase):
    def test_out_of_interval_cached_frames_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "picklist_001.mp4"
            labels = root / "picklist_001.csv"
            cache = root / "cache"
            write_dummy_video(video)
            write_compact_labels(labels)
            intervals = carry_with_pipeline_frame_intervals_1based(
                labels, N_FRAMES, frame_indexing="opencv0"
            )
            candidates = _candidate_frame_indices_carry_and_skip(intervals, N_FRAMES, FRAME_SKIP)
            self.assertEqual(candidates, [20, 24, 28])
            write_cache_for_video(
                cache,
                "sku_001",
                in_range=[20, 24],
                extra_out_of_range=[4, 100],
            )
            globbed = _glob_cached_frame_indices_first_label(str(cache), "sku_001")
            self.assertIn(4, globbed)
            self.assertIn(100, globbed)
            usable = [t for t in globbed if t in set(candidates)]
            self.assertEqual(usable, [20, 24])
            self.assertNotIn(4, usable)
            self.assertNotIn(100, usable)

    def test_cache_only_skips_missing_in_range_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "picklist_001.mp4"
            labels_dir = root / "labels"
            labels_dir.mkdir()
            cache = root / "cache"
            write_dummy_video(video)
            write_compact_labels(labels_dir / "picklist_001.csv")
            write_picklist_json(root / "picklist_001.json", "sku_001")
            write_cache_for_video(
                cache,
                "sku_001",
                in_range=[20, 28],
                extra_out_of_range=[100],
                skip_frames=[24],
            )
            coverage = []
            segs, flat, name = _process_single_video_from_cache(
                str(video),
                [["sku_001"]],
                str(cache),
                str(labels_dir),
                True,
                "opencv0",
                FRAME_SKIP,
                False,
                coverage_out=coverage,
            )
            self.assertEqual(name, "picklist_001")
            self.assertEqual(flat, ["sku_001"])
            self.assertEqual(len(coverage), 1)
            rec = coverage[0]
            self.assertEqual(rec["candidate_frames"], 3)
            self.assertEqual(rec["skipped_missing"], 1)
            self.assertGreaterEqual(rec["skipped_out_of_interval"], 1)
            self.assertEqual(rec["usable_frames"], 2)
            self.assertTrue(segs)

    def test_no_usable_in_range_frames_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "picklist_001.mp4"
            labels_dir = root / "labels"
            labels_dir.mkdir()
            cache = root / "cache"
            write_dummy_video(video)
            write_compact_labels(labels_dir / "picklist_001.csv")
            write_cache_for_video(cache, "sku_001", in_range=[], extra_out_of_range=[4, 100])
            with self.assertRaises(SystemExit):
                _process_single_video_from_cache(
                    str(video),
                    [["sku_001"]],
                    str(cache),
                    str(labels_dir),
                    True,
                    "opencv0",
                    FRAME_SKIP,
                    False,
                )


if __name__ == "__main__":
    unittest.main()
