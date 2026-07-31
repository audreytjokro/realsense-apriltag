from __future__ import annotations

import unittest

import record_cyranose_reading_pose as recording


def make_reading(monotonic_ns: int) -> recording.CyranoseReading:
    decoded = {field: "" for field in recording.PCNOSE_SOURCE_FIELDS}
    decoded["flag"] = 2
    return recording.CyranoseReading(
        monotonic_ns=monotonic_ns,
        sample_time_estimate_utc="2026-07-22T12:00:00.000+00:00",
        serial_roundtrip_ms=40.0,
        decoded=decoded,
    )


def make_pose(monotonic_ns: int, marker: str) -> recording.PoseSample:
    return recording.PoseSample(
        monotonic_ns=monotonic_ns,
        row={field: marker for field in recording.POSE_FIELDS},
        camera_frame=None,  # Not used by alignment tests.
        observations=[],
        T_camera_cube=None,
        cube_rms_px=float("nan"),
        used_tag_ids=(),
    )


class AlignmentTests(unittest.TestCase):
    def test_selects_nearest_bracketing_pose(self) -> None:
        reading = make_reading(1_000_000_000)
        previous = make_pose(900_000_000, "previous")
        current = make_pose(1_200_000_000, "current")
        self.assertIs(
            recording.select_nearest_pose(reading, previous, current),
            previous,
        )

    def test_tie_prefers_previous_pose(self) -> None:
        reading = make_reading(1_000_000_000)
        previous = make_pose(900_000_000, "previous")
        current = make_pose(1_100_000_000, "current")
        self.assertIs(
            recording.select_nearest_pose(reading, previous, current),
            previous,
        )

    def test_valid_match_keeps_pose_fields(self) -> None:
        reading = make_reading(1_000_000_000)
        pose = make_pose(1_200_000_000, "matched-pose")
        row = recording.build_combined_row(reading, pose, max_sync_ms=250.0)
        self.assertTrue(row["pose_alignment_valid"])
        self.assertEqual(row["pose_alignment_status"], "matched")
        self.assertEqual(row["pose_minus_pcnose_ms"], 200.0)
        self.assertEqual(row["pose_elapsed_s"], "matched-pose")

    def test_over_threshold_preserves_reading_but_blanks_pose(self) -> None:
        reading = make_reading(1_000_000_000)
        pose = make_pose(1_300_000_000, "must-not-leak")
        row = recording.build_combined_row(reading, pose, max_sync_ms=250.0)
        self.assertFalse(row["pose_alignment_valid"])
        self.assertEqual(row["pose_alignment_status"], "over_threshold")
        self.assertEqual(row["abs_pose_minus_pcnose_ms"], 300.0)
        self.assertEqual(row["pcnose_flag"], 2)
        self.assertEqual(row["pose_elapsed_s"], "")

    def test_alignment_summary_counts_and_percentiles(self) -> None:
        stats = recording.AlignmentStats.create(250.0)
        stats.add(-10.0, 30.0)
        stats.add(40.0, 50.0)
        stats.add(300.0, 70.0)
        summary = stats.as_dict()
        self.assertEqual(summary["matched_readings"], 2)
        self.assertEqual(summary["rejected_readings"], 1)
        self.assertAlmostEqual(summary["match_rate_percent"], 66.667)
        self.assertEqual(
            summary["absolute_pose_minus_pcnose_ms"]["maximum"],
            300.0,
        )


if __name__ == "__main__":
    unittest.main()
