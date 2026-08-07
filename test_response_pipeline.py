from __future__ import annotations

import statistics
import unittest
from collections import namedtuple
from pathlib import Path

import numpy as np

from analysis.response_pipeline import (
    TrialConfig,
    compute_baseline,
    compute_baseline_array,
    find_session_by_trial_id,
    interpolate_pose,
    load_session,
    median_fingerprint,
    normalized_channels,
    percentile,
    qualifying_rows,
    sensor_response,
    smooth_grid,
    summarize_responses,
)
from analysis.response_pipeline.pose import PoseSample

REPOSITORY_ROOT = Path(__file__).resolve().parent


class ComputeBaselineTests(unittest.TestCase):
    def test_uses_median_of_back_half_of_baseline_flag_rows(self) -> None:
        rows = [
            {"pcnose_flag": "1", "pcnose_S1_kohm": "10"},
            {"pcnose_flag": "1", "pcnose_S1_kohm": "12"},
            {"pcnose_flag": "1", "pcnose_S1_kohm": "14"},
            {"pcnose_flag": "1", "pcnose_S1_kohm": "16"},
            {"pcnose_flag": "2", "pcnose_S1_kohm": "999"},
        ]
        baseline = compute_baseline(rows, ["pcnose_S1_kohm"], baseline_flag="1")
        self.assertEqual(baseline, {"pcnose_S1_kohm": 15.0})

    def test_raises_when_no_rows_match_baseline_flag(self) -> None:
        rows = [{"pcnose_flag": "2", "pcnose_S1_kohm": "10"}]
        with self.assertRaises(ValueError):
            compute_baseline(rows, ["pcnose_S1_kohm"], baseline_flag="1")


class SensorResponseTests(unittest.TestCase):
    def test_fractional_change_and_rms_combine_correctly(self) -> None:
        raw_row = {"pcnose_S1_kohm": "12", "pcnose_S2_kohm": "8"}
        baseline = {"pcnose_S1_kohm": 10.0, "pcnose_S2_kohm": 10.0}
        fractional, response = sensor_response(
            raw_row, baseline, ["pcnose_S1_kohm", "pcnose_S2_kohm"]
        )
        self.assertEqual(fractional, [20.0, -20.0])
        self.assertAlmostEqual(response, 20.0)


class ComputeBaselineArrayTests(unittest.TestCase):
    def test_matches_dict_based_baseline_on_the_same_data(self) -> None:
        rows = [
            {"pcnose_flag": "1", "pcnose_S1_kohm": "10", "pcnose_S2_kohm": "100"},
            {"pcnose_flag": "1", "pcnose_S1_kohm": "12", "pcnose_S2_kohm": "102"},
            {"pcnose_flag": "1", "pcnose_S1_kohm": "14", "pcnose_S2_kohm": "104"},
            {"pcnose_flag": "1", "pcnose_S1_kohm": "16", "pcnose_S2_kohm": "106"},
            {"pcnose_flag": "2", "pcnose_S1_kohm": "999", "pcnose_S2_kohm": "999"},
        ]
        sensor_fields = ["pcnose_S1_kohm", "pcnose_S2_kohm"]
        dict_baseline = compute_baseline(rows, sensor_fields, baseline_flag="1")

        resistances = np.array([[float(r[f]) for f in sensor_fields] for r in rows])
        flags = np.array([int(r["pcnose_flag"]) for r in rows])
        array_baseline = compute_baseline_array(resistances, flags, baseline_flag=1)

        np.testing.assert_allclose(array_baseline, [dict_baseline[f] for f in sensor_fields])

    def test_raises_when_no_rows_match_baseline_flag(self) -> None:
        resistances = np.array([[1.0, 2.0]])
        flags = np.array([2])
        with self.assertRaises(ValueError):
            compute_baseline_array(resistances, flags, baseline_flag=1)


class NormalizedChannelsTests(unittest.TestCase):
    def test_matches_sensor_response_fractional_values(self) -> None:
        raw_row = {"pcnose_S1_kohm": "12", "pcnose_S2_kohm": "8"}
        baseline_dict = {"pcnose_S1_kohm": 10.0, "pcnose_S2_kohm": 10.0}
        fractional, _ = sensor_response(raw_row, baseline_dict, ["pcnose_S1_kohm", "pcnose_S2_kohm"])

        resistances = np.array([[12.0, 8.0]])
        baseline_array = np.array([10.0, 10.0])
        result = normalized_channels(resistances, baseline_array)

        np.testing.assert_allclose(result[0], fractional)


class PercentileTests(unittest.TestCase):
    def test_nearest_rank(self) -> None:
        values = [1, 2, 3, 4, 5]
        self.assertEqual(percentile(values, 0.90), 5)
        self.assertEqual(percentile(values, 0.50), 3)
        self.assertEqual(percentile(values, 0.0), 1)


class MedianFingerprintTests(unittest.TestCase):
    def test_per_channel_median(self) -> None:
        rows = [[10.0, -5.0], [20.0, -15.0], [30.0, -25.0]]
        self.assertEqual(median_fingerprint(rows, 2), [20.0, -15.0])


class SummarizeResponsesTests(unittest.TestCase):
    def test_count_median_and_p90(self) -> None:
        summary = summarize_responses([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(summary["count"], 5)
        self.assertEqual(summary["response_median"], 3.0)
        self.assertEqual(summary["response_p90"], 5.0)

    def test_raises_on_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            summarize_responses([])


class InterpolatePoseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = [PoseSample(0.0, 0.0, 0.0, 0.0), PoseSample(2.0, 10.0, 20.0, 4.0)]
        self.times = [0.0, 2.0]

    def test_interpolates_midpoint(self) -> None:
        pose = interpolate_pose(self.samples, self.times, 1.0, max_gap_s=5.0)
        assert pose is not None
        self.assertEqual((pose.x, pose.y, pose.z), (5.0, 10.0, 2.0))

    def test_returns_none_before_first_sample(self) -> None:
        self.assertIsNone(interpolate_pose(self.samples, self.times, -1.0, max_gap_s=5.0))

    def test_returns_none_after_last_sample(self) -> None:
        self.assertIsNone(interpolate_pose(self.samples, self.times, 3.0, max_gap_s=5.0))

    def test_returns_none_when_gap_exceeds_limit(self) -> None:
        samples = [PoseSample(0.0, 0.0, 0.0, 0.0), PoseSample(10.0, 1.0, 1.0, 1.0)]
        self.assertIsNone(interpolate_pose(samples, [0.0, 10.0], 5.0, max_gap_s=5.0))


class SmoothGridTests(unittest.TestCase):
    Point = namedtuple("Point", ["x", "y", "response"])

    def test_weighted_average_at_shared_location(self) -> None:
        points = [self.Point(0.0, 0.0, 10.0), self.Point(0.0, 0.0, 20.0)]
        grid = smooth_grid(points, [0.0], [0.0], radius_cm=1.0, sigma_cm=1.0, min_count=2)
        self.assertAlmostEqual(grid[0][0], 15.0)

    def test_cell_is_none_below_min_count(self) -> None:
        points = [self.Point(0.0, 0.0, 10.0)]
        grid = smooth_grid(points, [0.0], [0.0], radius_cm=1.0, sigma_cm=1.0, min_count=2)
        self.assertIsNone(grid[0][0])

    def test_points_outside_radius_are_excluded(self) -> None:
        points = [self.Point(0.0, 0.0, 10.0), self.Point(100.0, 100.0, 999.0)]
        grid = smooth_grid(points, [0.0], [0.0], radius_cm=1.0, sigma_cm=1.0, min_count=2)
        self.assertIsNone(grid[0][0])  # only one point is within radius


class QualifyingRowsTests(unittest.TestCase):
    """A hand-traced synthetic session: pose_times land on t=0..8, one sample per row.

    Reading A (row t=4, lag 1s -> pose at t=3): in the 0-1 cm height band, inside bounds.
    Reading B (row t=6, lag 1s -> pose at t=5): outside the height band (height 9 cm).
    Reading C (row t=8, lag 1s -> pose at t=7): inside the height band but outside desk bounds.
    """

    def setUp(self) -> None:
        def row(t, flag, s1, s2, x, y, z):
            return {
                "pcnose_flag": flag,
                "pcnose_S1_kohm": str(s1),
                "pcnose_S2_kohm": str(s2),
                "pose_elapsed_s": str(t),
                "snout_desk_x_cm": str(x),
                "snout_desk_y_cm": str(y),
                "snout_desk_z_cm": str(z),
            }

        self.raw_rows = [
            row(0.0, "1", 10, 10, 0, 0, 0.5),
            row(1.0, "1", 10, 10, 1, 0, 0.5),
            row(2.0, "1", 10, 10, 2, 0, 0.5),
            row(3.0, "1", 10, 10, 3, 0, 0.5),
            row(4.0, "2", 12, 8, 4, 0, 9.0),  # reading A; own position unused by A's lag lookup
            row(5.0, "0", 10, 10, 5, 0, 9.0),
            row(6.0, "2", 10, 10, 6, 0, 9.0),  # reading B
            row(7.0, "0", 10, 10, 20, 0, 0.5),  # out of desk bounds on X
            row(8.0, "2", 10, 10, 8, 0, 0.5),  # reading C
        ]
        self.config = TrialConfig(
            trial_id="synthetic",
            lag_s=1.0,
            height_band_cm=(0.0, 1.0),
            desk_bounds_cm=(0.0, 10.0, 0.0, 10.0),
            pose_window_s=(0.0, 10.0),
            max_interpolation_gap_s=1.5,
            sensor_count=2,
        )

    def test_default_height_filter_keeps_only_the_in_band_in_bounds_reading(self) -> None:
        rows = qualifying_rows(self.raw_rows, self.config)
        self.assertEqual(len(rows), 1)
        (row,) = rows
        self.assertEqual((row.x, row.y, row.z), (3.0, 0.0, 0.5))
        self.assertAlmostEqual(row.height_cm, 0.5)
        self.assertTrue(row.height_in_band)
        self.assertAlmostEqual(row.response, 20.0)
        self.assertAlmostEqual(row.scan_elapsed_s, 3.0)
        self.assertAlmostEqual(row.scan_fraction, 0.3)

    def test_disabling_height_filter_retains_out_of_band_reading_but_flags_it(self) -> None:
        rows = qualifying_rows(self.raw_rows, self.config, apply_height_filter=False)
        # Reading C is still excluded: it fails the desk-bounds check, which always applies.
        self.assertEqual(len(rows), 2)
        in_band = [row for row in rows if row.height_in_band]
        out_of_band = [row for row in rows if not row.height_in_band]
        self.assertEqual(len(in_band), 1)
        self.assertEqual(len(out_of_band), 1)
        self.assertAlmostEqual(out_of_band[0].height_cm, 9.0)
        self.assertAlmostEqual(out_of_band[0].response, 0.0)


class Notebook04RegressionTests(unittest.TestCase):
    """Pins this module's output to notebook 04's already-published, reviewed numbers.

    These are notebook 04's own saved cell outputs (see
    analysis/notebooks/04_mint_vs_blank_raster.ipynb), reproduced here with the
    same RESPONSE_LAG_S/HEIGHT_BAND_CM/DESK_BOUNDS constants that notebook used.
    If this test ever fails after an edit to this package, the refactor changed
    a published result and must not be merged as-is.
    """

    HEIGHT_BAND_CM = (1.0, 3.5)
    DESK_BOUNDS_CM = (13.0, 38.0, 18.0, 47.0)
    LAG_S = 3.0
    MAX_INTERPOLATION_GAP_S = 1.5

    @classmethod
    def setUpClass(cls) -> None:
        blank_dir = find_session_by_trial_id(REPOSITORY_ROOT, "line_raster_blank_fresh_01")
        mint_dir = find_session_by_trial_id(REPOSITORY_ROOT, "line_raster_mint_blotter_retry_01")
        cls.blank = load_session(blank_dir)
        cls.mint = load_session(mint_dir)
        cls.blank_config = TrialConfig(
            trial_id="line_raster_blank_fresh_01",
            lag_s=cls.LAG_S,
            height_band_cm=cls.HEIGHT_BAND_CM,
            desk_bounds_cm=cls.DESK_BOUNDS_CM,
            pose_window_s=(86.0, 198.0),
            max_interpolation_gap_s=cls.MAX_INTERPOLATION_GAP_S,
        )
        cls.mint_config = TrialConfig(
            trial_id="line_raster_mint_blotter_retry_01",
            lag_s=cls.LAG_S,
            height_band_cm=cls.HEIGHT_BAND_CM,
            desk_bounds_cm=cls.DESK_BOUNDS_CM,
            pose_window_s=(24.0, 127.0),
            max_interpolation_gap_s=cls.MAX_INTERPOLATION_GAP_S,
        )

    def test_blank_trial_matches_published_counts_and_summary(self) -> None:
        rows = qualifying_rows(self.blank.raw_rows, self.blank_config)
        self.assertEqual(len(rows), 174)
        summary = summarize_responses([row.response for row in rows])
        heights = [row.height_cm for row in rows]
        self.assertAlmostEqual(summary["response_median"], 0.071, places=3)
        self.assertAlmostEqual(summary["response_p90"], 0.083, places=3)
        self.assertAlmostEqual(statistics.median(heights), 1.84, places=2)

        all_rows = qualifying_rows(self.blank.raw_rows, self.blank_config, apply_height_filter=False)
        excluded = [row for row in all_rows if not row.height_in_band]
        self.assertEqual(len(excluded), 0)

    def test_mint_trial_matches_published_counts_and_summary(self) -> None:
        rows = qualifying_rows(self.mint.raw_rows, self.mint_config)
        self.assertEqual(len(rows), 126)
        summary = summarize_responses([row.response for row in rows])
        heights = [row.height_cm for row in rows]
        self.assertAlmostEqual(summary["response_median"], 0.282, places=3)
        self.assertAlmostEqual(summary["response_p90"], 0.722, places=3)
        self.assertAlmostEqual(statistics.median(heights), 2.56, places=2)

        all_rows = qualifying_rows(self.mint.raw_rows, self.mint_config, apply_height_filter=False)
        excluded = [row for row in all_rows if not row.height_in_band]
        self.assertEqual(len(all_rows), 147)
        self.assertEqual(len(excluded), 21)

    def test_mint_to_blank_median_rms_ratio_matches_published_figure(self) -> None:
        blank_rows = qualifying_rows(self.blank.raw_rows, self.blank_config)
        mint_rows = qualifying_rows(self.mint.raw_rows, self.mint_config)
        blank_median = summarize_responses([row.response for row in blank_rows])["response_median"]
        mint_median = summarize_responses([row.response for row in mint_rows])["response_median"]
        self.assertAlmostEqual(mint_median / blank_median, 3.99, places=2)


if __name__ == "__main__":
    unittest.main()
