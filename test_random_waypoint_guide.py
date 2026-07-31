from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import random_waypoint_guide as guide


class RandomWaypointGuideTests(unittest.TestCase):
    def test_seed_is_reproducible_and_points_stay_in_bounds(self) -> None:
        first = guide.WaypointGenerator(73001, 26.5, 26.5, 1.5)
        second = guide.WaypointGenerator(73001, 26.5, 26.5, 1.5)
        first_points = [first.next() for _ in range(10)]
        second_points = [second.next() for _ in range(10)]

        self.assertEqual(first_points, second_points)
        for point in first_points:
            self.assertGreaterEqual(point.x_cm, 1.5)
            self.assertLessEqual(point.x_cm, 25.0)
            self.assertGreaterEqual(point.y_cm, 1.5)
            self.assertLessEqual(point.y_cm, 25.0)

    def test_finds_newest_recent_session_with_matching_trial_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            session = root / "cyranose_reading_pose_session_20260730_120000"
            session.mkdir()
            (session / "session_metadata.json").write_text(
                json.dumps({"trial_id": "mint_run"}),
                encoding="utf-8",
            )

            self.assertEqual(
                guide.find_recent_session("mint_run", 3600, root),
                session,
            )
            self.assertIsNone(
                guide.find_recent_session("lavender_run", 3600, root),
            )

    def test_reads_latest_recorder_elapsed_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            csv_path = Path(temp) / "cyranose_reading_pose.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "pcnose_sample_time_estimate_utc",
                        "pcnose_elapsed_s",
                        "pcnose_S1_kohm",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "pcnose_sample_time_estimate_utc": (
                            "2026-07-30T20:00:00.000+00:00"
                        ),
                        "pcnose_elapsed_s": "0.0",
                        "pcnose_S1_kohm": "1.2",
                    }
                )
                writer.writerow(
                    {
                        "pcnose_sample_time_estimate_utc": (
                            "2026-07-30T20:00:12.400+00:00"
                        ),
                        # The legacy device-derived elapsed field advances
                        # about 1.6x faster and must not drive the guide timer.
                        "pcnose_elapsed_s": "19.84",
                        "pcnose_S1_kohm": "1.3",
                    }
                )

            self.assertEqual(
                guide.read_latest_recorder_elapsed_s(csv_path),
                12.4,
            )

    def test_duration_format(self) -> None:
        self.assertEqual(guide.format_duration(None), "--:--")
        self.assertEqual(guide.format_duration(0), "00:00")
        self.assertEqual(guide.format_duration(599.6), "09:59")
        self.assertEqual(guide.format_duration(600), "10:00")


if __name__ == "__main__":
    unittest.main()
