from __future__ import annotations

import json
import random
import subprocess
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

from h264_video import H264VideoWriter

from analysis.spatial_sequence.annotation import (
    build_calibrated_desk_reference,
    nearest_qualified_desk_row,
    undo_last_click,
)
from analysis.spatial_sequence.core import (
    GUARD_LENGTH,
    PAPER_HALF_CM,
    SENSOR_COLUMNS,
    NormalizationStats,
    PreparedData,
    SessionData,
    SessionInfo,
    SpatialAnnotation,
    SpatialAnchorDataset,
    SpatialWindowDataset,
    SplitBlock,
    WindowRecord,
    _build_session_data,
    anchor_records_for_sessions,
    assign_evaluation_blocks,
    apply_transform_2d,
    balanced_lengths,
    canonical_paper_corners,
    compute_normalization_stats,
    distances_to_polygon,
    estimate_velocity,
    fit_similarity_transform,
    inverse_regression_target,
    load_annotation,
    make_split_blocks,
    pixel_to_desk,
    paper_contains,
    points_in_polygon,
    position_bin_centers,
    position_classes,
    project_desk_points,
    transform_from_rvec_tvec,
    training_row_mask,
    windows_for_sessions,
    write_annotation,
)
from analysis.spatial_sequence.evaluation import (
    aggregate_physical_tokens,
    masked_loss,
)
from analysis.spatial_sequence.manifest import (
    LOSO_MANIFEST_PATH,
    MANIFEST_PATH,
    SEQUENCE_LENGTH_MANIFEST_PATH,
    ManifestRun,
    representative_position_runs,
    validate_run_manifest,
)
from analysis.spatial_sequence.metrics import (
    area_metrics,
    checkpoint_selection_metric,
    position_metrics,
    scalar_regression_metrics,
    summarize_metrics,
    selection_is_better,
    vector_regression_metrics,
)
from analysis.spatial_sequence.models import (
    CNN_DILATIONS,
    SpatialSequenceModel,
)
from analysis.spatial_sequence.sanity import (
    SourceEncounter,
    apply_kernel,
    balance_spatial_cells,
    build_kernel_operator,
    equal_group_weights,
    fit_equal_session_pca,
    group_source_encounters,
    normalized_wasserstein_1d,
    paired_distance_bin_differences,
    robust_zscore,
    scores_to_rgb,
    segment_cell_visits,
    spatial_cell_ids,
    symmetric_score_limits,
    validate_encounter_windows,
    weighted_quantile,
)
from analysis.spatial_sequence.reporting import (
    _median_performance_seed_index,
    _spatial_diagnostic_figure_name,
    classify_seed_convergence,
)
from analysis.spatial_sequence.training import (
    RunConfig,
    capture_rng_state,
    ensure_output_directory,
    default_run_directory,
    position_video_path,
    rewrite_checkpoint_metadata_atomic,
    restore_rng_state,
    train_run,
)
from analysis.spatial_sequence.visualization import (
    _cross_seed_parity_figure,
    nearest_video_frame_index,
    position_probability_maximum,
    render_cross_seed_diagnostics,
    render_cross_seed_parity_scatter,
    render_position_panel,
    seed_majority_classes,
    seed_probability_disagreement,
)


SESSION_IDS = (
    "20260730_152754",
    "20260730_153631",
    "20260730_154848",
    "20260730_161358",
    "20260730_163355",
)
LAYOUTS = (
    "mint-only horizontal",
    "caret mint-left lavender-right",
    "caret lavender-left mint-right",
    "lavender-only horizontal",
    "inverted caret lavender-left mint-right",
)


def _write_test_video(path: Path, frame_count: int = 5) -> None:
    writer = H264VideoWriter(path, 30.0, (320, 240))
    for index in range(frame_count):
        image = np.full((240, 320, 3), 30 + index, dtype=np.uint8)
        cv2.rectangle(image, (60, 40), (260, 210), (80, 80, 80), 2)
        writer.write(image)
    writer.release()


def _sensor_values(row_count: int) -> dict[str, np.ndarray]:
    rows = np.arange(row_count, dtype=np.float64)
    return {
        column: 20.0
        + channel
        + 0.5 * np.sin(rows / (5.0 + channel / 8.0))
        + rows * (0.001 + channel * 0.00001)
        for channel, column in enumerate(SENSOR_COLUMNS)
    }


def _full_frame(row_count: int = 500) -> pd.DataFrame:
    rows = np.arange(row_count, dtype=np.float64)
    frame = pd.DataFrame(
        {
            "pcnose_host_time_utc": (
                pd.Timestamp("2026-07-30T00:00:00Z")
                + pd.to_timedelta(rows * 0.1, unit="s")
            ).astype(str),
            "pcnose_flag": np.full(row_count, 2, dtype=int),
            "snout_position_valid": np.ones(row_count, dtype=bool),
            "snout_desk_x_cm": np.linspace(-12.0, 12.0, row_count),
            "snout_desk_y_cm": 5.0 * np.sin(rows / 31.0),
            "snout_desk_z_cm": np.full(row_count, -5.0),
            "snout_cam_x_cm": np.linspace(-12.0, 12.0, row_count),
            "snout_cam_y_cm": 5.0 * np.sin(rows / 31.0),
            "snout_cam_z_cm": np.full(row_count, 95.0),
            "pose_elapsed_s": rows * 0.1,
            "frame_number": rows.astype(int) + 1,
            "tag_6_visible": np.ones(row_count, dtype=bool),
            "tag_6_cam_x_cm": np.zeros(row_count),
            "tag_6_cam_y_cm": np.zeros(row_count),
            "tag_6_cam_z_cm": np.full(row_count, 100.0),
            "tag_6_cam_rvec_x_rad": np.zeros(row_count),
            "tag_6_cam_rvec_y_rad": np.zeros(row_count),
            "tag_6_cam_rvec_z_rad": np.zeros(row_count),
        }
    )
    for name, values in _sensor_values(row_count).items():
        frame[name] = values
    return frame


def _session_info(
    root: Path,
    session_id: str = SESSION_IDS[0],
    trial_id: str = "trial",
    layout: str = LAYOUTS[0],
) -> SessionInfo:
    run_directory = root / f"run-{session_id}"
    session_directory = run_directory / f"cyranose_reading_pose_session_{session_id}"
    session_directory.mkdir(parents=True, exist_ok=True)
    sources = tuple(name for name in ("mint", "lavender") if name in layout)
    return SessionInfo(
        session_id=session_id,
        trial_id=trial_id,
        layout=layout,
        run_directory=run_directory,
        session_directory=session_directory,
        csv_path=session_directory / "cyranose_reading_pose.csv",
        video_path=session_directory / "rectified_rgb.mp4",
        annotation_path=session_directory / "spatial_annotation.json",
        source_names=sources,
    )


def _manual_annotation(info: SessionInfo, source_polygon: np.ndarray | None = None) -> SpatialAnnotation:
    sources = {}
    if "mint" in info.source_names:
        sources["mint"] = (
            np.asarray(source_polygon, dtype=float)
            if source_polygon is not None
            else np.array([[-4, -2], [4, -2], [4, 2], [-4, 2]], dtype=float)
        )
    if "lavender" in info.source_names:
        sources["lavender"] = np.array([[-4, 4], [4, 4], [4, 7], [-4, 7]], dtype=float)
    return SpatialAnnotation(
        schema_version=1,
        session_id=info.session_id,
        trial_id=info.trial_id,
        usable_start_row=0,
        usable_end_row=499,
        desk_to_paper=np.eye(3),
        paper_to_desk=np.eye(3),
        paper_corners_desk_cm=canonical_paper_corners(),
        source_polygons_paper_cm=sources,
        source_hashes={},
        diagnostics={},
    )


def _manual_session(row_count: int = 28) -> SessionData:
    info = SessionInfo(
        "session",
        "trial",
        "mint-only",
        Path("."),
        Path("."),
        Path("data.csv"),
        Path("video.mp4"),
        Path("spatial_annotation.json"),
        ("mint",),
    )
    rows = np.arange(row_count, dtype=np.float32)
    sensors = np.column_stack(
        [rows + 0.03 * channel * rows**2 for channel in range(32)]
    ).astype(np.float32)
    positions = np.column_stack([rows, -rows]).astype(np.float32)
    distance = np.column_stack([rows + 1, rows + 2]).astype(np.float32)
    velocity = np.column_stack([2 * rows - 3, -rows + 4]).astype(np.float32)
    return SessionData(
        info=info,
        frame=pd.DataFrame(index=range(row_count)),
        raw_row_indices=np.arange(row_count),
        timestamps_s=rows.astype(float),
        sensors=sensors,
        pose_mask=np.ones(row_count, dtype=bool),
        paper_xy_cm=positions,
        position_classes=position_classes(positions),
        position_mask=paper_contains(positions),
        distance_cm=distance,
        distance_mask=np.ones((row_count, 2), dtype=bool),
        area_classes=np.zeros(row_count, dtype=np.int64),
        height_cm=(rows + 1).astype(np.float32),
        velocity_cm_s=velocity,
        velocity_mask=np.ones((row_count, 2), dtype=bool),
        blocks=[SplitBlock("train", 0, 0, row_count)],
    )


class GeometryAndAnnotationTests(unittest.TestCase):
    def test_similarity_transform_and_reprojection_round_trip(self) -> None:
        canonical = canonical_paper_corners()
        angle = np.deg2rad(23.0)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        desk = canonical @ (2.2 * rotation).T + np.array([4.0, -7.0])
        transform, inverse = fit_similarity_transform(desk, canonical)
        np.testing.assert_allclose(apply_transform_2d(transform, desk), canonical, atol=1e-10)
        np.testing.assert_allclose(transform @ inverse, np.eye(3), atol=1e-10)

        camera_matrix = np.array([[200.0, 0, 160], [0, 200.0, 120], [0, 0, 1]])
        camera_to_desk = transform_from_rvec_tvec([0, 0, 0], [0, 0, 100])
        point = np.array([[5.0, -3.0]])
        pixel = project_desk_points(point, camera_matrix, camera_to_desk)[0]
        recovered = pixel_to_desk(pixel, camera_matrix, camera_to_desk)
        np.testing.assert_allclose(recovered, point[0], atol=1e-10)

    def test_polygon_distance_membership_and_bins(self) -> None:
        polygon = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
        points = np.array([[0, 0], [2, 0], [1, 0], [2, 2]], dtype=float)
        np.testing.assert_array_equal(points_in_polygon(points, polygon), [True, False, True, False])
        np.testing.assert_allclose(distances_to_polygon(points, polygon), [0, 1, 0, np.sqrt(2)])
        values = np.array(
            [-20, -PAPER_HALF_CM, -PAPER_HALF_CM + 1e-6, PAPER_HALF_CM - 1e-6, PAPER_HALF_CM, 20]
        )
        axis_values = np.column_stack([values, np.zeros(len(values))])
        joint = position_classes(axis_values)
        self.assertEqual(joint[0], -1)
        self.assertEqual(joint[-1], -1)
        self.assertEqual(joint[1], 6 * 13)
        self.assertEqual(joint[4], 6 * 13 + 12)

    def test_area_outside_paper_precedence_height_sign_and_absent_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            info = _session_info(Path(temporary))
            frame = _full_frame()
            frame.loc[10, ["snout_desk_x_cm", "snout_cam_x_cm"]] = 13.0
            frame.loc[11, ["snout_desk_x_cm", "snout_cam_x_cm"]] = 14.0
            frame.to_csv(info.csv_path, index=False)
            annotation = _manual_annotation(
                info,
                np.array([[12, -2], [16, -2], [16, 2], [12, 2]], dtype=float),
            )
            session = _build_session_data(info, annotation)
            self.assertEqual(session.area_classes[10], 1)
            self.assertEqual(session.area_classes[11], 0)
            self.assertAlmostEqual(float(session.height_cm[10]), 5.0)
            self.assertTrue(np.all(~session.distance_mask[:, 1]))
            frame.loc[0, SENSOR_COLUMNS[0]] = np.nan
            frame.to_csv(info.csv_path, index=False)
            with self.assertRaisesRegex(ValueError, "Non-finite sensor"):
                _build_session_data(info, annotation)

    def test_velocity_equal_weight_ols_and_requirements(self) -> None:
        times = np.arange(0.0, 5.1, 0.5)
        positions = np.column_stack([2.0 * times + 1.0, -3.0 * times + 4.0])
        velocity, mask = estimate_velocity(times, positions, np.ones(len(times), dtype=bool))
        self.assertTrue(np.all(mask[3:-3]))
        np.testing.assert_allclose(
            velocity[3:-3],
            np.tile([2.0, -3.0], (len(velocity[3:-3]), 1)),
            atol=1e-10,
        )
        sparse = np.zeros(len(times), dtype=bool)
        sparse[[4, 5, 6]] = True
        _, sparse_mask = estimate_velocity(times, positions, sparse)
        self.assertFalse(np.any(sparse_mask))

    def test_single_frame_reference_tracks_camera_move_and_global_undo(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            info = _session_info(Path(temporary))
            frame = _full_frame()
            frame.loc[250, "tag_6_visible"] = False
            frame.loc[251, "tag_6_cam_x_cm"] = 10.0
            _write_test_video(info.video_path)
            camera_matrix = np.array(
                [[200.0, 0.0, 160.0], [0.0, 200.0, 120.0], [0.0, 0.0, 1.0]]
            )
            bounds = (-50.0, 50.0, -30.0, 30.0)

            self.assertEqual(nearest_qualified_desk_row(frame, 250), 249)
            before_move = build_calibrated_desk_reference(
                info,
                frame,
                249,
                camera_matrix=camera_matrix,
                bounds_cm=bounds,
                output_width=320,
            )
            after_move = build_calibrated_desk_reference(
                info,
                frame,
                251,
                camera_matrix=camera_matrix,
                bounds_cm=bounds,
                output_width=320,
            )
            fallback = build_calibrated_desk_reference(
                info,
                frame,
                250,
                camera_matrix=camera_matrix,
                bounds_cm=bounds,
                output_width=320,
            )

            self.assertEqual(before_move.source_raw_row, 249)
            self.assertEqual(after_move.source_raw_row, 251)
            self.assertEqual(fallback.source_raw_row, 249)
            self.assertEqual(after_move.raw_image_rgb.shape, (240, 320, 3))
            self.assertEqual(after_move.image_rgb.shape, (480, 320, 3))
            self.assertFalse(
                np.array_equal(before_move.image_rgb, after_move.image_rgb)
            )

        points = {"paper": [(1.0, 1.0)], "mint": [(2.0, 2.0)]}
        history = ["paper", "mint"]
        self.assertEqual(undo_last_click(points, history), "mint")
        self.assertEqual(points["mint"], [])
        self.assertEqual(undo_last_click(points, history), "paper")
        self.assertEqual(points["paper"], [])
        self.assertIsNone(undo_last_click(points, history))

    def test_annotation_schema_and_hash_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            info = _session_info(Path(temporary))
            pd.DataFrame(
                {"pcnose_host_time_utc": ["2026-07-30T00:00:00Z", "2026-07-30T00:00:01Z"]}
            ).to_csv(info.csv_path, index=False)
            info.video_path.write_bytes(b"video")
            annotation = write_annotation(
                info,
                0,
                1,
                canonical_paper_corners(),
                {"mint": np.array([[-2, -1], [2, -1], [2, 1], [-2, 1]])},
                diagnostics={"qc": True},
            )
            self.assertEqual(annotation.schema_version, 1)
            record = json.loads(info.annotation_path.read_text())
            self.assertEqual(record["coordinate_convention"]["x_positive"], "right_in_calibrated_desk_view")
            pd.DataFrame(
                {"pcnose_host_time_utc": ["2026-07-30T00:00:00Z", "2026-07-30T00:00:02Z"]}
            ).to_csv(info.csv_path, index=False)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                loaded = load_annotation(info)
            self.assertEqual(loaded.session_id, info.session_id)
            self.assertTrue(any("hash changed" in str(item.message) for item in caught))


class SplitAndNormalizationTests(unittest.TestCase):
    def test_split_ratio_remainders_guards_and_window_boundaries(self) -> None:
        blocks = make_split_blocks(1000)
        train_lengths = [block.stop - block.start for block in blocks if block.split == "train"]
        validation_lengths = [
            block.stop - block.start for block in blocks if block.split == "validation"
        ]
        self.assertLessEqual(max(train_lengths) - min(train_lengths), 1)
        self.assertLessEqual(max(validation_lengths) - min(validation_lengths), 1)
        self.assertEqual(train_lengths, sorted(train_lengths, reverse=True))
        self.assertEqual(validation_lengths, sorted(validation_lengths, reverse=True))
        for first, second in zip(blocks, blocks[1:]):
            self.assertEqual(second.start - first.stop, GUARD_LENGTH)
        retained = sum(train_lengths) + sum(validation_lengths)
        self.assertEqual(sum(validation_lengths), round(retained / 5))
        self.assertEqual(balanced_lengths(10, 4), [3, 3, 2, 2])

        session = _manual_session(1000)
        session.blocks = blocks
        for split in ("train", "validation"):
            windows = windows_for_sessions([session], split)
            for window in windows:
                containing = [
                    block
                    for block in blocks
                    if block.split == split
                    and block.start <= window.start
                    and window.start + 24 <= block.stop
                ]
                self.assertEqual(len(containing), 1)

    def test_unique_training_row_statistics_ignore_window_occurrences(self) -> None:
        session = _manual_session()
        session.sensors[27] = 1e7
        session.blocks = [
            SplitBlock("train", 0, 0, 25),
            SplitBlock("validation", 0, 25, 28),
        ]
        windows = [
            WindowRecord(0, 0, "train", 0),
            WindowRecord(0, 1, "train", 0),
        ]
        stats = compute_normalization_stats([session], windows, "velocity")
        expected_inputs = session.sensors[:25]
        np.testing.assert_allclose(stats.input_mean, expected_inputs.mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(stats.input_std, expected_inputs.std(axis=0, ddof=0), rtol=1e-6)
        expected_velocity = session.velocity_cm_s[:25]
        np.testing.assert_allclose(stats.target_mean, expected_velocity.mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(stats.target_std, expected_velocity.std(axis=0, ddof=0), rtol=1e-6)
        self.assertLess(float(stats.input_mean.max()), 1e6)
        np.testing.assert_array_equal(
            training_row_mask(session),
            np.arange(28) < 25,
        )

        masked = _manual_session()
        masked.blocks = list(session.blocks)
        masked.velocity_cm_s[5] = np.nan
        masked.velocity_mask[5] = False
        masked_stats = compute_normalization_stats([masked], windows, "velocity")
        expected_masked = masked.velocity_cm_s[:25]
        expected_mask = masked.velocity_mask[:25]
        for component in range(2):
            valid_values = expected_masked[:, component][expected_mask[:, component]]
            self.assertAlmostEqual(masked_stats.target_mean[component], valid_values.mean())
            self.assertAlmostEqual(masked_stats.target_std[component], valid_values.std(ddof=0))

        constant = _manual_session()
        constant.blocks = list(session.blocks)
        constant.sensors[:, 0] = 3.0
        with self.assertRaisesRegex(ValueError, "standard deviation"):
            compute_normalization_stats([constant], windows, "position")

    def test_masked_labels_absent_head_and_scaling(self) -> None:
        session = _manual_session()
        session.pose_mask[3] = False
        session.position_classes[3] = -1
        session.distance_mask[:, 1] = False
        session.distance_cm[:, 1] = np.nan
        stats = NormalizationStats(
            input_mean=np.zeros(32, dtype=np.float32),
            input_std=np.ones(32, dtype=np.float32),
            target_mean=np.zeros(2, dtype=np.float32),
            target_std=np.array([2.0, 3.0], dtype=np.float32),
            target_transform="std_only",
        )
        prepared = PreparedData(
            [session],
            [WindowRecord(0, 0, "train", 0)],
            [WindowRecord(0, 0, "validation", 0)],
            stats,
            "distance",
            {},
        )
        item = SpatialWindowDataset(prepared, "train")[0]
        self.assertTrue(torch.all(~item["mask"][:, 1]))
        self.assertTrue(torch.allclose(item["target"][:, 0], torch.tensor(session.distance_cm[:24, 0] / 2)))
        values = np.array([[1.5, 2.0]])
        np.testing.assert_allclose(
            inverse_regression_target("distance", values, stats),
            [[3.0, 6.0]],
        )
        height_stats = NormalizationStats(
            np.zeros(32),
            np.ones(32),
            np.zeros(1),
            np.array([2.0]),
            "log1p_std_only",
        )
        np.testing.assert_allclose(
            inverse_regression_target("height", np.array([[np.log1p(4.0) / 2]]), height_stats),
            [[4.0]],
        )

        velocity_stats = NormalizationStats(
            np.zeros(32),
            np.ones(32),
            np.array([10.0, -2.0]),
            np.array([2.0, 4.0]),
            "zscore",
        )
        np.testing.assert_allclose(
            inverse_regression_target("velocity", np.array([[1.5, -0.5]]), velocity_stats),
            [[13.0, -4.0]],
        )

class MetricAndAggregationTests(unittest.TestCase):
    def test_all_metric_definitions(self) -> None:
        truth = np.array([2 * 13 + 1, 4 * 13 + 3])
        joint = np.eye(169)[truth].reshape(-1, 13, 13)
        centers = position_bin_centers()
        xy = np.array([[centers[1], centers[2]], [centers[3], centers[4]]])
        position = position_metrics(truth, joint, xy)
        self.assertEqual(position["top_4"], 1.0)
        self.assertEqual(position["top_8"], 1.0)
        self.assertEqual(position["top_16"], 1.0)
        self.assertAlmostEqual(position["map_euclidean_error_cm"], 0.0)
        self.assertAlmostEqual(position["expected_euclidean_error_cm"], 0.0)

        area_truth = np.array([0, 1, 2, 2])
        area_prediction = np.eye(3)[[0, 1, 0, 2]]
        area = area_metrics(area_truth, area_prediction, present_classes_only=False)
        self.assertAlmostEqual(area["accuracy"], 0.75)
        self.assertEqual(area["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [1, 0, 1]])

        scalar = scalar_regression_metrics([1, 3], [2, 1])
        self.assertAlmostEqual(scalar["mae"], 1.5)
        self.assertAlmostEqual(scalar["rmse"], np.sqrt(2.5))
        vector = vector_regression_metrics(
            np.array([[0, 0], [1, 1]]),
            np.array([[3, 4], [1, 2]]),
            np.ones((2, 2), dtype=bool),
            ("vx", "vy"),
            True,
        )
        self.assertAlmostEqual(vector["mean_vector_error"], 3.0)

    def test_pooled_equal_session_and_occurrence_aggregation(self) -> None:
        identical = np.array(
            [[[0.2, 0.3, 0.5]], [[0.2, 0.3, 0.5]]], dtype=float
        )
        opposite = np.array(
            [[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]], dtype=float
        )
        np.testing.assert_allclose(seed_probability_disagreement(identical), [0.0])
        np.testing.assert_allclose(seed_probability_disagreement(opposite), [1.0])
        voting_probabilities = np.array(
            [
                [[0.51, 0.49, 0.00], [0.1, 0.1, 0.8], [0.9, 0.1, 0.0]],
                [[0.51, 0.49, 0.00], [0.1, 0.1, 0.8], [0.9, 0.1, 0.0]],
                [[0.01, 0.98, 0.01], [0.1, 0.8, 0.1], [0.1, 0.9, 0.0]],
                [[0.01, 0.98, 0.01], [0.1, 0.1, 0.8], [0.1, 0.9, 0.0]],
                [[0.00, 0.00, 1.00], [0.1, 0.1, 0.8], [0.0, 0.0, 1.0]],
            ]
        )
        np.testing.assert_array_equal(
            seed_majority_classes(voting_probabilities),
            [1, 2, 0],
        )

        truth = np.array([0, 0, 1, 1])
        probabilities = np.eye(3)[[0, 0, 0, 1]]
        summary = summarize_metrics(
            "area",
            truth,
            {"probabilities": probabilities},
            np.ones(4, dtype=bool),
            np.array([0, 0, 1, 1]),
            ["a", "b"],
        )
        self.assertIn("pooled", summary)
        self.assertIn("equal_session_macro", summary)
        self.assertEqual(set(summary["per_session"]), {"a", "b"})

        first = np.zeros((13, 13))
        second = np.zeros((13, 13))
        third = np.zeros((13, 13))
        first[2, 1] = 1.0
        second[4, 2] = 1.0
        third[4, 3] = 1.0
        occurrences = {
            "session_index": np.array([0, 0, 0]),
            "raw_row": np.array([10, 10, 11]),
            "fragment": np.array([0, 0, 0]),
            "truth": np.array([27, 27, -1]),
            "mask": np.array([True, True, False]),
            "paper_xy_cm": np.zeros((3, 2)),
            "joint_probabilities": np.stack([first, second, third]),
        }
        aggregate = aggregate_physical_tokens(occurrences, "position")
        self.assertEqual(aggregate["occurrence_count"].tolist(), [2, 1])
        np.testing.assert_allclose(aggregate["joint_probabilities"].sum(axis=(1, 2)), 1.0)
        np.testing.assert_allclose(
            aggregate["joint_probabilities"].sum(axis=1),
            aggregate["x_probabilities"],
        )
        np.testing.assert_allclose(
            aggregate["joint_probabilities"].sum(axis=2),
            aggregate["y_probabilities"],
        )
        self.assertEqual(aggregate["joint_probabilities"][0, 2, 1], 0.5)
        self.assertEqual(aggregate["joint_probabilities"][0, 4, 2], 0.5)
        self.assertEqual(aggregate["joint_probabilities"][0, 2, 2], 0.0)
        self.assertEqual(aggregate["joint_probabilities"][0, 4, 1], 0.0)
        self.assertFalse(bool(aggregate["mask"][1]))


class ModelTests(unittest.TestCase):
    def test_output_shapes_dense_supervision_and_topology(self) -> None:
        expected = {
            "position": {"position": (2, 24, 169)},
            "distance": {"distance": (2, 24, 2)},
            "area": {"area": (2, 24, 3)},
            "height": {"height": (2, 24, 1)},
            "velocity": {"velocity": (2, 24, 2)},
        }
        inputs = torch.randn(2, 24, 32)
        for architecture in ("transformer", "temporal-cnn"):
            for target_name, shapes in expected.items():
                model = SpatialSequenceModel(architecture, "causal", target_name).eval()
                outputs = model(inputs)
                self.assertEqual({name: tuple(value.shape) for name, value in outputs.items()}, shapes)
        cnn = SpatialSequenceModel("temporal-cnn", "causal", "velocity")
        self.assertIsNone(cnn.position_encoding)
        self.assertEqual(
            tuple(block.convolution.dilation[0] for block in cnn.blocks),
            CNN_DILATIONS,
        )
        outputs = SpatialSequenceModel("transformer", "causal", "position")(inputs)
        target = torch.zeros(2, 24, dtype=torch.long)
        mask = torch.ones(2, 24, dtype=torch.bool)
        _, count = masked_loss("position", outputs, target, mask)
        self.assertEqual(count, 2 * 24)

    def test_causal_future_invariance_for_both_architectures(self) -> None:
        torch.manual_seed(3)
        base = torch.randn(1, 24, 32)
        changed = base.clone()
        changed[:, 13:] += torch.randn_like(changed[:, 13:]) * 10
        for architecture in ("transformer", "temporal-cnn"):
            model = SpatialSequenceModel(architecture, "causal", "velocity").eval()
            with torch.no_grad():
                first = model(base)["velocity"]
                second = model(changed)["velocity"]
            torch.testing.assert_close(first[:, :13], second[:, :13], atol=1e-6, rtol=1e-6)

    def test_temporal_cnn_padding_boundaries(self) -> None:
        causal = SpatialSequenceModel("temporal-cnn", "causal", "position")
        bidirectional = SpatialSequenceModel(
            "temporal-cnn", "bidirectional", "position"
        )
        for dilation, causal_block, bidirectional_block in zip(
            CNN_DILATIONS, causal.blocks, bidirectional.blocks
        ):
            total = dilation * 4
            self.assertEqual(causal_block.padding, (total, 0))
            self.assertEqual(
                bidirectional_block.padding,
                (total // 2, total // 2),
            )
        parameter_shapes = {
            name: tuple(parameter.shape)
            for name, parameter in bidirectional.named_parameters()
        }
        for length in (6, 12, 18, 24):
            outputs = bidirectional(torch.randn(2, length, 32))["position"]
            self.assertEqual(tuple(outputs.shape), (2, length, 169))
            self.assertEqual(
                parameter_shapes,
                {
                    name: tuple(parameter.shape)
                    for name, parameter in bidirectional.named_parameters()
                },
            )


class ProtocolTests(unittest.TestCase):
    def _prepared_for_anchor(self, target: str = "position") -> PreparedData:
        session = _manual_session(648)
        session.blocks = make_split_blocks(648)
        stats = NormalizationStats(
            input_mean=np.zeros(32, dtype=np.float32),
            input_std=np.ones(32, dtype=np.float32),
            target_mean=np.zeros(2, dtype=np.float32),
            target_std=np.ones(2, dtype=np.float32),
            target_transform="zscore" if target == "velocity" else None,
        )
        return PreparedData(
            sessions=[session],
            train_windows=windows_for_sessions([session], "train"),
            validation_windows=windows_for_sessions([session], "validation"),
            stats=stats,
            target=target,
            source_hashes={},
        )

    def test_single_validation_split_and_shared_anchor_rows(self) -> None:
        prepared = self._prepared_for_anchor()
        blocks = prepared.sessions[0].blocks
        self.assertEqual(
            [(block.split, block.start, block.stop) for block in blocks],
            [("train", 0, 241), ("validation", 264, 384), ("train", 407, 648)],
        )
        self.assertEqual(
            sum(block.stop - block.start for block in blocks if block.split == "train"),
            482,
        )
        self.assertEqual(
            sum(
                block.stop - block.start
                for block in blocks
                if block.split == "validation"
            ),
            120,
        )
        self.assertEqual(len(prepared.train_windows), 482 - 2 * 23)
        self.assertEqual(len(prepared.validation_windows), 120 - 23)
        causal = anchor_records_for_sessions(prepared.sessions, "causal")
        bidirectional = anchor_records_for_sessions(
            prepared.sessions, "bidirectional"
        )
        self.assertEqual(len(causal), 86)
        self.assertEqual(len(bidirectional), 86)
        self.assertEqual(
            [record.target_index for record in causal],
            [record.target_index for record in bidirectional],
        )
        self.assertEqual(len({record.target_index for record in causal}), 86)
        self.assertEqual(causal[0].target_offset, 23)
        self.assertEqual(causal[0].window_start, 264)
        self.assertEqual(bidirectional[0].target_offset, 12)
        self.assertEqual(bidirectional[0].window_start, 275)

    def test_sequence_lengths_share_physical_anchors_but_use_all_windows(self) -> None:
        prepared = self._prepared_for_anchor("area")
        expected_targets = None
        expected_window_counts = {6: 472, 12: 460, 18: 448, 24: 436}
        for length in (6, 12, 18, 24):
            prepared.sequence_length = length
            prepared.train_windows = windows_for_sessions(
                prepared.sessions, "train", length
            )
            prepared.validation_windows = windows_for_sessions(
                prepared.sessions, "validation", length
            )
            records = anchor_records_for_sessions(
                prepared.sessions,
                "bidirectional",
                length,
            )
            targets = [record.target_index for record in records]
            if expected_targets is None:
                expected_targets = targets
            else:
                self.assertEqual(targets, expected_targets)
            self.assertEqual(len(records), 86)
            self.assertEqual(records[0].target_offset, length // 2)
            self.assertEqual(len(prepared.train_windows), expected_window_counts[length])
            item = SpatialWindowDataset(prepared, "train")[0]
            self.assertEqual(tuple(item["inputs"].shape), (length, 32))
            self.assertEqual(tuple(item["target"].shape), (length,))

    def test_leave_one_session_out_uses_four_full_training_sessions(self) -> None:
        sessions = []
        for index in range(5):
            current = _manual_session(40)
            current.info = SessionInfo(
                f"id-{index}",
                f"trial-{index}",
                "mint-lavender",
                Path(f"session-{index}"),
                Path(f"recording-{index}"),
                Path(f"data-{index}.csv"),
                Path(f"video-{index}.mp4"),
                Path(f"annotation-{index}.json"),
                ("mint", "lavender"),
            )
            current.sensors += index * 100.0
            sessions.append(current)
        assign_evaluation_blocks(
            sessions,
            "leave-one-session-out",
            "session-4",
        )
        self.assertTrue(
            all(session.blocks[0].split == "train" for session in sessions[:4])
        )
        self.assertEqual(sessions[4].blocks[0].split, "validation")
        train_windows = windows_for_sessions(sessions, "train", 24)
        validation_windows = windows_for_sessions(sessions, "validation", 24)
        self.assertEqual(len(train_windows), 4 * 17)
        self.assertEqual(len(validation_windows), 17)
        stats = compute_normalization_stats(sessions, train_windows, "area")
        expected = np.concatenate([session.sensors for session in sessions[:4]], axis=0)
        np.testing.assert_allclose(stats.input_mean, expected.mean(axis=0), rtol=1e-6)
        prepared = PreparedData(
            sessions,
            train_windows,
            validation_windows,
            stats,
            "area",
            {},
            sequence_length=24,
            evaluation_scheme="leave-one-session-out",
        )
        anchors = SpatialAnchorDataset(prepared, "bidirectional")
        self.assertEqual(len(anchors), 17)
        self.assertEqual(int(anchors[0]["session_index"]), 4)
        self.assertEqual(int(anchors[0]["target_offset"]), 12)
        self.assertEqual(int(anchors[0]["raw_row_index"]), 12)

    def test_dense_training_and_anchor_only_validation_coexist(self) -> None:
        prepared = self._prepared_for_anchor("position")
        dense = SpatialWindowDataset(prepared, "train")[0]
        self.assertEqual(tuple(dense["target"].shape), (24,))
        self.assertEqual(tuple(dense["mask"].shape), (24,))
        causal = SpatialAnchorDataset(prepared, "causal")
        bidirectional = SpatialAnchorDataset(prepared, "bidirectional")
        self.assertEqual(len(causal), 86)
        self.assertEqual(len(bidirectional), 86)
        self.assertEqual(int(causal[0]["target_offset"]), 23)
        self.assertEqual(int(bidirectional[0]["target_offset"]), 12)
        self.assertEqual(int(causal[0]["raw_row_index"]), 287)
        self.assertEqual(int(bidirectional[0]["raw_row_index"]), 287)
        torch.testing.assert_close(
            causal[0]["inputs"],
            torch.as_tensor(prepared.sessions[0].sensors[264:288]),
        )
        torch.testing.assert_close(
            bidirectional[0]["inputs"],
            torch.as_tensor(prepared.sessions[0].sensors[275:299]),
        )

    def test_joint_occurrence_aggregation_and_outside_masking(self) -> None:
        session = _manual_session(24)
        session.paper_xy_cm[0] = [PAPER_HALF_CM + 0.1, 0.0]
        session.position_classes[0] = -1
        session.position_mask[0] = False
        prepared = PreparedData(
            [session],
            [WindowRecord(0, 0, "train", 0)],
            [WindowRecord(0, 0, "validation", 0)],
            NormalizationStats(np.zeros(32), np.ones(32), None, None, None),
            "position",
            {},
        )
        item = SpatialWindowDataset(prepared, "train")[0]
        self.assertFalse(bool(item["mask"][0]))
        self.assertTrue(bool(item["mask"][1]))

        first = np.zeros((13, 13))
        second = np.zeros((13, 13))
        first[2, 3] = 1.0
        second[8, 9] = 1.0
        occurrences = {
            "session_index": np.array([0, 0]),
            "raw_row": np.array([10, 10]),
            "fragment": np.array([0, 0]),
            "truth": np.array([29, 29]),
            "mask": np.array([True, True]),
            "paper_xy_cm": np.array([[0.0, 0.0], [0.0, 0.0]]),
            "joint_probabilities": np.stack([first, second]),
        }
        aggregate = aggregate_physical_tokens(occurrences, "position")
        self.assertEqual(aggregate["joint_probabilities"].shape, (1, 13, 13))
        self.assertEqual(aggregate["joint_probabilities"][0, 2, 3], 0.5)
        self.assertEqual(aggregate["joint_probabilities"][0, 8, 9], 0.5)
        self.assertEqual(aggregate["joint_probabilities"][0, 2, 9], 0.0)
        np.testing.assert_allclose(
            aggregate["joint_probabilities"].sum(axis=(1, 2)), 1.0
        )
        np.testing.assert_allclose(
            aggregate["x_probabilities"],
            aggregate["joint_probabilities"].sum(axis=1),
        )

    def test_all_checkpoint_criteria_and_tie_breaking(self) -> None:
        position = {
            "pooled": {"top_8": 0.4},
            "equal_session_macro": {},
            "per_session": {},
        }
        self.assertEqual(
            checkpoint_selection_metric("position", position),
            ("anchor_top_8", 0.4, "maximize"),
        )
        area = {
            "pooled": {},
            "equal_session_macro": {"macro_f1": 0.6},
            "per_session": {},
        }
        self.assertEqual(checkpoint_selection_metric("area", area)[1:], (0.6, "maximize"))
        per_session = {
            "a": {
                "mint": {"mae": 2.0, "count": 3},
                "lavender": {"mae": 4.0, "count": 3},
                "mae": 5.0,
                "count": 3,
                "mean_vector_error": 7.0,
                "vector_count": 3,
            },
            "b": {
                "mint": {"mae": 6.0, "count": 3},
                "lavender": {"mae": 0.0, "count": 0},
                "mae": 9.0,
                "count": 3,
                "mean_vector_error": 11.0,
                "vector_count": 3,
            },
        }
        shell = {"pooled": {}, "equal_session_macro": {}, "per_session": per_session}
        self.assertAlmostEqual(checkpoint_selection_metric("distance", shell)[1], 4.5)
        self.assertAlmostEqual(checkpoint_selection_metric("height", shell)[1], 7.0)
        self.assertAlmostEqual(checkpoint_selection_metric("velocity", shell)[1], 9.0)
        self.assertTrue(selection_is_better(0.5, 1.0, 4, 0.5, 1.1, 3, "maximize"))
        self.assertTrue(selection_is_better(0.5, 1.0, 2, 0.5, 1.0, 3, "maximize"))
        self.assertFalse(selection_is_better(0.5, 1.0, 4, 0.5, 1.0, 3, "maximize"))

    def test_default_paths_and_median_seed(self) -> None:
        base = RunConfig("transformer", "causal", "position", session="sample")
        self.assertTrue(
            str(default_run_directory(base)).endswith(
                "output/spatial-sequence/runs/position/sample/transformer_causal_seed0"
            )
        )
        pooled = RunConfig("transformer", "causal", "height")
        self.assertTrue(
            str(default_run_directory(pooled)).endswith(
                "output/spatial-sequence/runs/height/transformer_causal_seed0"
            )
        )
        self.assertNotIn("/pooled/", str(default_run_directory(pooled)))
        short = RunConfig(
            "temporal-cnn",
            "bidirectional",
            "area",
            sequence_length=6,
        )
        self.assertTrue(
            str(default_run_directory(short)).endswith(
                "output/spatial-sequence/sequence-length/runs/area/length-6/temporal-cnn_bidirectional_seed0"
            )
        )
        loso = RunConfig(
            "temporal-cnn",
            "bidirectional",
            "distance",
            evaluation_scheme="leave-one-session-out",
            held_out_session="mint-only-horizontal-run01",
        )
        self.assertTrue(
            str(default_run_directory(loso)).endswith(
                "output/spatial-sequence/leave-one-session-out/runs/distance/mint-only-horizontal-run01/temporal-cnn_bidirectional_seed0"
            )
        )
        self.assertTrue(
            str(position_video_path(base)).endswith(
                "output/spatial-sequence/videos/position_sample_causal_transformer_seed0.mp4"
            )
        )

        with tempfile.TemporaryDirectory() as temporary:
            runs = []
            values = [0.1, 0.5, 0.3, 0.4, 0.2]
            for seed, value in enumerate(values):
                directory = Path(temporary) / f"seed{seed}"
                directory.mkdir()
                (directory / "metrics.json").write_text(
                    json.dumps(
                        {
                            "best_selection_value": value,
                            "equal_session_macro": {"top_8": value},
                        }
                    )
                )
                runs.append(
                    ManifestRun(
                        f"run{seed}",
                        "transformer",
                        "causal",
                        "position",
                        "session",
                        seed,
                        directory,
                    )
                )
            self.assertEqual(_median_performance_seed_index(runs), 2)
            # The helper enforces the full 20 logical groups, so duplicate this
            # controlled five-seed pattern across all groups.
            all_runs = []
            for architecture in ("transformer", "temporal-cnn"):
                for mode in ("causal", "bidirectional"):
                    for session_index in range(5):
                        for run in runs:
                            all_runs.append(
                                ManifestRun(
                                    f"{architecture}-{mode}-{session_index}-{run.seed}",
                                    architecture,
                                    mode,
                                    "position",
                                    f"session-{session_index}",
                                    run.seed,
                                    run.run_directory,
                                )
                            )
            selected = representative_position_runs(all_runs)
            self.assertEqual(len(selected), 20)
            self.assertEqual({run.seed for run in selected}, {2})

    def test_convergence_rules_and_cross_seed_diagnostic_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def convergence_run(name: str, values: np.ndarray, best_epoch: int) -> ManifestRun:
                directory = root / name
                directory.mkdir()
                pd.DataFrame(
                    {
                        "epoch": np.arange(1, 101),
                        "train_loss": np.linspace(1.0, 0.1, 100),
                        "selection_metric_value": values,
                        "selection_direction": ["maximize"] * 100,
                    }
                ).to_csv(directory / "history.csv", index=False)
                (directory / "metrics.json").write_text(
                    json.dumps({"best_epoch": best_epoch})
                )
                return ManifestRun(
                    name,
                    "transformer",
                    "causal",
                    "position",
                    "session",
                    0,
                    directory,
                )

            overfit_values = np.concatenate(
                [np.linspace(0.70, 0.80, 10), np.linspace(0.79, 0.70, 90)]
            )
            improving_values = np.linspace(0.2, 0.7, 100)
            self.assertEqual(
                classify_seed_convergence(convergence_run("overfit", overfit_values, 10)),
                "fast-overfit",
            )
            self.assertEqual(
                classify_seed_convergence(
                    convergence_run("improving", improving_values, 100)
                ),
                "still-improving",
            )

            xy = np.array([[-1.0, -1.0], [1.0, 1.0]])
            aggregates = []
            for seed in range(5):
                aggregates.append(
                    {
                        "session_index": np.array([0, 0]),
                        "raw_row": np.array([10, 11]),
                        "paper_xy_cm": xy.copy(),
                        "truth": np.array([[2.0], [4.0]]),
                        "prediction": np.array([[2.0 + seed], [4.0 + seed]]),
                    }
                )
            untouched = [aggregate["prediction"].copy() for aggregate in aggregates]
            output = root / "cross-seed-height.png"
            render_cross_seed_diagnostics(
                "height",
                aggregates,
                output,
                architecture="transformer",
                temporal_mode="causal",
            )
            self.assertTrue(output.is_file())
            for before, aggregate in zip(untouched, aggregates):
                np.testing.assert_array_equal(before, aggregate["prediction"])

            area_aggregates = []
            for seed in range(5):
                area_aggregates.append(
                    {
                        "session_index": np.array([0, 0, 1, 1]),
                        "session_slugs": np.array(["mint-layout", "lavender-layout"]),
                        "session_ids": np.array(["session-a", "session-b"]),
                        "raw_row": np.array([10, 11, 20, 21]),
                        "paper_xy_cm": np.array(
                            [[-1.0, -1.0], [1.0, 1.0], [-2.0, 2.0], [2.0, -2.0]]
                        ),
                        "truth": np.array([0, 1, 0, 2]),
                        "probabilities": np.array(
                            [
                                [0.8, 0.1, 0.1],
                                [0.1, 0.8, 0.1],
                                [0.8, 0.1, 0.1],
                                [0.1, 0.1, 0.8],
                            ]
                        ),
                    }
                )
            area_output = root / "cross-seed-area.png"
            observed_titles: list[tuple[str, list[str]]] = []
            original_savefig = Figure.savefig

            def capture_titles(figure: Figure, *args: object, **kwargs: object) -> object:
                observed_titles.append(
                    (
                        figure._suptitle.get_text() if figure._suptitle else "",
                        [axis.get_title() for axis in figure.axes],
                    )
                )
                return original_savefig(figure, *args, **kwargs)

            with mock.patch.object(Figure, "savefig", new=capture_titles):
                area_outputs = render_cross_seed_diagnostics(
                    "area",
                    area_aggregates,
                    area_output,
                    architecture="temporal-cnn",
                    temporal_mode="bidirectional",
                )
            self.assertEqual(len(area_outputs), 2)
            self.assertFalse(area_output.exists())
            self.assertTrue((root / "cross-seed-area_mint-layout.png").is_file())
            self.assertTrue((root / "cross-seed-area_lavender-layout.png").is_file())
            self.assertEqual(len(observed_titles), 2)
            for main_title, panel_titles in observed_titles:
                self.assertNotIn("\n", main_title)
                self.assertNotIn("five-seed", main_title.lower())
                self.assertIn("Prediction", panel_titles)
                self.assertNotIn("Five-Seed Majority Prediction", panel_titles)

            distance_aggregates = []
            for seed in range(5):
                distance_aggregates.append(
                    {
                        "session_index": np.array([0, 0, 1, 1]),
                        "session_slugs": np.array(["mint-layout", "lavender-layout"]),
                        "session_ids": np.array(["session-a", "session-b"]),
                        "raw_row": np.array([10, 11, 20, 21]),
                        "paper_xy_cm": np.array(
                            [[-1.0, -1.0], [1.0, 1.0], [-2.0, 2.0], [2.0, -2.0]]
                        ),
                        "truth": np.array(
                            [[1.0, np.nan], [2.0, np.nan], [np.nan, 3.0], [np.nan, 4.0]]
                        ),
                        "prediction": np.array(
                            [
                                [1.0 + seed, np.nan],
                                [2.0 + seed, np.nan],
                                [np.nan, 3.0 + seed],
                                [np.nan, 4.0 + seed],
                            ]
                        ),
                        "mask": np.array(
                            [[True, False], [True, False], [False, True], [False, True]]
                        ),
                    }
                )
            distance_output = root / "cross-seed-distance.png"
            distance_outputs = render_cross_seed_diagnostics(
                "distance",
                distance_aggregates,
                distance_output,
                architecture="temporal-cnn",
                temporal_mode="bidirectional",
            )
            self.assertEqual(len(distance_outputs), 2)
            self.assertFalse(distance_output.exists())
            self.assertTrue((root / "cross-seed-distance_mint-layout.png").is_file())
            self.assertTrue(
                (root / "cross-seed-distance_lavender-layout.png").is_file()
            )

    def test_cross_seed_parity_scatter(self) -> None:
        self.assertEqual(
            _spatial_diagnostic_figure_name(
                "distance", "pooled", "temporal-cnn", "bidirectional"
            ),
            "distance_bidirectional_tcnn.png",
        )
        self.assertEqual(
            _spatial_diagnostic_figure_name(
                "height", "pooled", "transformer", "causal"
            ),
            "height_causal_transformer.png",
        )
        identifiers = {
            "session_index": np.array([0, 0, 1, 1]),
            "raw_row": np.array([10, 11, 20, 21]),
        }
        distance_truth = np.array(
            [[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0], [5.0, np.nan]]
        )
        distance_mask = np.isfinite(distance_truth)
        distance_aggregates = []
        for seed in range(5):
            distance_aggregates.append(
                {
                    **identifiers,
                    "truth": distance_truth.copy(),
                    "mask": distance_mask.copy(),
                    "prediction": np.where(
                        distance_mask, distance_truth + 0.1 * seed, np.nan
                    ),
                }
            )
        distance_figure = _cross_seed_parity_figure(
            "distance", distance_aggregates, "temporal-cnn", "bidirectional"
        )
        self.assertEqual(len(distance_figure.axes), 2)
        self.assertEqual(
            [len(axis.collections[0].get_offsets()) for axis in distance_figure.axes],
            [15, 10],
        )
        for axis in distance_figure.axes:
            np.testing.assert_allclose(axis.get_xlim(), axis.get_ylim())
            self.assertEqual(len(axis.lines), 1)
            self.assertEqual(axis.lines[0].get_linestyle(), "--")
            self.assertIsNone(axis.get_legend())
            self.assertEqual(len(axis.collections[0].get_facecolors()), 1)
        np.testing.assert_allclose(
            distance_figure.axes[0].get_xlim(),
            distance_figure.axes[1].get_xlim(),
        )
        distance_figure.clear()

        height_truth = np.array([[1.0], [2.0], [25.0], [3.0]])
        height_aggregates = []
        for seed in range(5):
            height_aggregates.append(
                {
                    **identifiers,
                    "truth": height_truth.copy(),
                    "mask": np.ones_like(height_truth, dtype=bool),
                    "prediction": np.array(
                        [[1.0 + 0.1 * seed], [2.0], [3.0 + seed], [3.0]]
                    ),
                }
            )
        height_figure = _cross_seed_parity_figure(
            "height", height_aggregates, "transformer", "causal"
        )
        height_axis = height_figure.axes[0]
        self.assertEqual(height_axis.get_xscale(), "log")
        self.assertEqual(height_axis.get_yscale(), "log")
        self.assertEqual(len(height_axis.collections[0].get_offsets()), 20)
        self.assertGreater(height_axis.get_xlim()[1], 25.0)
        np.testing.assert_allclose(height_axis.get_xlim(), height_axis.get_ylim())
        height_figure.clear()

        velocity_truth = np.array(
            [[3.0, 4.0], [0.0, 2.0], [1.0, 1.0], [np.nan, 2.0]]
        )
        velocity_mask = np.array(
            [[True, True], [True, True], [False, True], [False, True]]
        )
        velocity_aggregates = []
        for seed in range(5):
            prediction = velocity_truth.copy()
            prediction[:2] += 0.1 * seed
            velocity_aggregates.append(
                {
                    **identifiers,
                    "truth": velocity_truth.copy(),
                    "mask": velocity_mask.copy(),
                    "prediction": prediction,
                }
            )
        velocity_figure = _cross_seed_parity_figure(
            "velocity", velocity_aggregates, "transformer", "causal"
        )
        speed_offsets = np.asarray(
            velocity_figure.axes[0].collections[0].get_offsets()
        )
        self.assertEqual(len(speed_offsets), 10)
        np.testing.assert_allclose(np.unique(speed_offsets[:, 0]), [2.0, 5.0])
        np.testing.assert_allclose(
            velocity_figure.axes[0].get_xlim(), velocity_figure.axes[0].get_ylim()
        )
        velocity_figure.clear()

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "height-parity.png"
            self.assertEqual(
                render_cross_seed_parity_scatter(
                    "height",
                    height_aggregates,
                    output,
                    architecture="transformer",
                    temporal_mode="causal",
                ),
                output,
            )
            self.assertTrue(output.is_file())

        with self.assertRaisesRegex(ValueError, "supports distance"):
            _cross_seed_parity_figure(
                "area", height_aggregates, "transformer", "causal"
            )
        with self.assertRaisesRegex(ValueError, "exactly five"):
            _cross_seed_parity_figure(
                "height", height_aggregates[:4], "transformer", "causal"
            )
        mismatched = [dict(aggregate) for aggregate in height_aggregates]
        mismatched[4] = dict(mismatched[4], raw_row=np.array([10, 11, 20, 22]))
        with self.assertRaisesRegex(ValueError, "mismatch in 'raw_row'"):
            _cross_seed_parity_figure(
                "height", mismatched, "transformer", "causal"
            )


class SensorSanityTests(unittest.TestCase):
    def test_cell_balancing_uses_one_median_per_occupied_cell(self) -> None:
        points = np.array([[-1.0, -1.0], [-0.9, -0.9], [1.0, 1.0], [1.1, 1.1]])
        values = np.array([[1.0, 10.0], [3.0, 14.0], [100.0, 20.0], [104.0, 24.0]])
        balanced = balance_spatial_cells(points, values)
        self.assertEqual(len(balanced.cell_ids), 2)
        np.testing.assert_array_equal(balanced.counts, [2, 2])
        np.testing.assert_allclose(balanced.values, [[2.0, 12.0], [102.0, 22.0]])
        np.testing.assert_allclose(balanced.cell_xy_cm, [[-0.95, -0.95], [1.05, 1.05]])

    def test_spatial_cell_ids_mask_outside_paper(self) -> None:
        ids, ix, iy = spatial_cell_ids(
            np.array([[-13.25, -13.25], [13.25, 13.25], [14.0, 0.0], [np.nan, 0.0]])
        )
        self.assertGreaterEqual(ids[0], 0)
        self.assertGreaterEqual(ids[1], 0)
        self.assertEqual(ix[1], 52)
        self.assertEqual(iy[1], 52)
        np.testing.assert_array_equal(ids[2:], [-1, -1])

    def test_robust_zscore_uses_scaled_mad_and_rejects_constants(self) -> None:
        values = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        transformed, median, scale = robust_zscore(values)
        np.testing.assert_allclose(median, [2.0, 4.0])
        np.testing.assert_allclose(scale, [1.4826, 2.9652])
        np.testing.assert_allclose(np.median(transformed, axis=0), [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "Robust scale"):
            robust_zscore(np.ones((5, 2)))

    def test_gaussian_operator_enforces_support_and_normalizes_weights(self) -> None:
        points = np.array([[-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25]])
        operator = build_kernel_operator(points)
        y_index = int(np.flatnonzero(np.isclose(operator.y_axis_cm, -0.25))[0])
        x_index = int(np.flatnonzero(np.isclose(operator.x_axis_cm, -0.25))[0])
        flat_index = y_index * len(operator.x_axis_cm) + x_index
        self.assertTrue(operator.supported[y_index, x_index])
        self.assertGreaterEqual(operator.neighbor_count[y_index, x_index], 3)
        self.assertGreaterEqual(operator.effective_sample_size[y_index, x_index], 2.0)
        self.assertAlmostEqual(operator.normalized_weights[flat_index].sum(), 1.0)
        surface = apply_kernel(operator, np.arange(1.0, 5.0))
        self.assertEqual(surface.shape, (*operator.supported.shape, 1))
        self.assertTrue(np.isnan(surface[~operator.supported]).all())
        self.assertGreater(surface[y_index, x_index, 0], 1.0)
        self.assertLess(surface[y_index, x_index, 0], 4.0)

    def test_global_pca_gives_each_session_equal_total_weight(self) -> None:
        first = np.array([[0.0, 0.0, 1.0], [2.0, 2.0, 4.0]])
        second = np.array(
            [[8.0, 1.0, 4.0], [10.0, 3.0, 8.0], [12.0, 5.0, 7.0], [14.0, 7.0, 12.0]]
        )
        fit, scores = fit_equal_session_pca([first, second])
        expected_mean = (first.mean(axis=0) + second.mean(axis=0)) / 2.0
        np.testing.assert_allclose(fit.mean, expected_mean)
        self.assertEqual([value.shape for value in scores], [(2, 3), (4, 3)])
        for component in fit.components:
            pivot = np.argmax(np.abs(component))
            self.assertGreaterEqual(component[pivot], 0.0)
        limits = symmetric_score_limits(scores, quantile=0.99, equal_session=True)
        rgb = scores_to_rgb(np.zeros((2, 3)), limits)
        np.testing.assert_allclose(rgb, 0.5)

    def test_equal_group_weights_and_weighted_quantiles(self) -> None:
        labels = np.array(["long", "long", "long", "long", "short"])
        weights = equal_group_weights(labels)
        self.assertAlmostEqual(weights[labels == "long"].sum(), 0.5)
        self.assertAlmostEqual(weights[labels == "short"].sum(), 0.5)
        quantiles = weighted_quantile(np.arange(5.0), [0.25, 0.5, 0.75], weights)
        self.assertTrue(np.all(np.diff(quantiles) >= 0))

    def test_visits_split_on_cell_change_invalid_row_and_time_gap(self) -> None:
        ids = np.array([1, 1, 1, 1, 2, 2, 2])
        timestamps = np.array([0.0, 0.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        valid = np.array([True, True, True, True, True, False, True])
        visits = segment_cell_visits(ids, timestamps, valid, maximum_gap_s=1.5)
        self.assertEqual([visit.cell_id for visit in visits], [1, 1, 2, 2])
        self.assertEqual([visit.indices.tolist() for visit in visits], [[0, 1], [2, 3], [4], [6]])

    def test_normalized_wasserstein_has_physical_scale(self) -> None:
        self.assertAlmostEqual(
            normalized_wasserstein_1d(np.array([0.0, 0.0]), np.array([2.0, 2.0]), scale=2.0),
            1.0,
        )

    def test_encounter_grouping_and_exact_ten_row_windows(self) -> None:
        timestamps = np.arange(50, dtype=float) * 0.557
        inside = np.zeros(50, dtype=bool)
        inside[[20, 21, 25, 40]] = True
        encounters = group_source_encounters(inside, timestamps, merge_gap_s=3.0)
        self.assertEqual(encounters, [SourceEncounter(20, 25), SourceEncounter(40, 40)])
        accepted, records = validate_encounter_windows(
            encounters, np.ones(50, dtype=bool), inside, np.zeros(50, dtype=bool)
        )
        self.assertEqual(len(accepted), 2)
        np.testing.assert_array_equal(accepted[0].entry_indices, np.arange(11, 21))
        np.testing.assert_array_equal(accepted[0].exit_indices, np.arange(25, 35))
        self.assertTrue(all(record["status"] == "accepted" for record in records))
        other = np.zeros(50, dtype=bool)
        other[30] = True
        accepted, records = validate_encounter_windows(
            encounters, np.ones(50, dtype=bool), inside, other
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(records[0]["status"], "other_source_contact")

    def test_paired_distance_bins_do_not_mix_unmatched_distances(self) -> None:
        entry_distance = np.array([0.2, 0.8, 1.2])
        exit_distance = np.array([0.1, 0.7, 2.2])
        entry = np.array([[1.0, 10.0], [3.0, 14.0], [50.0, 60.0]])
        exit_values = np.array([[5.0, 18.0], [7.0, 22.0], [100.0, 120.0]])
        differences = paired_distance_bin_differences(
            entry_distance, exit_distance, entry, exit_values, bin_width_cm=1.0
        )
        self.assertEqual(set(differences), {0})
        np.testing.assert_allclose(differences[0], [4.0, 8.0])


class TrainingAndVideoTests(unittest.TestCase):
    def test_h264_writer_uses_h264_and_preserves_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "writer.mp4"
            writer = H264VideoWriter(path, 7.5, (64, 48))
            for index in range(6):
                writer.write(np.full((48, 64, 3), index * 30, dtype=np.uint8))
            writer.release()
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,nb_frames,width,height",
                    "-of",
                    "default=noprint_wrappers=1",
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            self.assertIn("codec_name=h264", probe)
            self.assertIn("width=64", probe)
            self.assertIn("height=48", probe)
            self.assertIn("nb_frames=6", probe)
            self.assertEqual(list(Path(temporary).glob(".*.h264-writing.mp4")), [])

    def test_atomic_checkpoint_metadata_rewrite_and_failure_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            payload = {
                "format_version": 1,
                "config": {"name": "before"},
                "signature": {"shape": [1, 2]},
                "model": {"weight": torch.arange(4)},
                "optimizer": {"state": {0: {"step": torch.tensor(3)}}},
                "epoch": 7,
            }
            torch.save(payload, path)
            rewrite_checkpoint_metadata_atomic(
                path,
                {
                    "config": {"name": "after"},
                    "signature": {"shape": [1, 2], "anchor": 23},
                },
            )
            migrated = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(migrated["config"], {"name": "after"})
            self.assertEqual(migrated["signature"]["anchor"], 23)
            torch.testing.assert_close(migrated["model"]["weight"], payload["model"]["weight"])
            torch.testing.assert_close(
                migrated["optimizer"]["state"][0]["step"],
                payload["optimizer"]["state"][0]["step"],
            )
            before_failure = path.read_bytes()
            with mock.patch("analysis.spatial_sequence.training.torch.save", side_effect=OSError("disk")):
                with self.assertRaisesRegex(OSError, "disk"):
                    rewrite_checkpoint_metadata_atomic(
                        path,
                        {"config": {"name": "never-written"}},
                    )
            self.assertEqual(path.read_bytes(), before_failure)
            self.assertFalse(path.with_name(path.name + ".migrating").exists())

    def test_rng_resume_collision_manifest_and_video_panel(self) -> None:
        generator = torch.Generator().manual_seed(9)
        random.seed(9)
        np.random.seed(9)
        torch.manual_seed(9)
        state = capture_rng_state(generator)
        expected = (
            random.random(),
            np.random.rand(),
            torch.rand(1),
            torch.rand(1, generator=generator),
        )
        restore_rng_state(state, generator)
        actual = (
            random.random(),
            np.random.rand(),
            torch.rand(1),
            torch.rand(1, generator=generator),
        )
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        torch.testing.assert_close(expected[2], actual[2])
        torch.testing.assert_close(expected[3], actual[3])

        self.assertEqual(MANIFEST_PATH.name, "run_manifest.csv")
        self.assertEqual(len(validate_run_manifest()), 180)
        self.assertEqual(len(validate_run_manifest(SEQUENCE_LENGTH_MANIFEST_PATH)), 40)
        self.assertEqual(len(validate_run_manifest(LOSO_MANIFEST_PATH)), 50)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "occupied").write_text("x")
            config = RunConfig("transformer", "causal", "height", output_directory=str(path))
            with self.assertRaises(FileExistsError):
                ensure_output_directory(config, None)

        joint_probabilities = np.full((13, 13), 1 / (13 * 13))
        probability_maximum = position_probability_maximum(joint_probabilities[None])
        panel = render_position_panel(
            joint_probabilities,
            np.array([np.nan, np.nan]),
            False,
            probability_maximum,
            size=640,
        )
        self.assertEqual(panel.shape, (640, 640, 3))
        self.assertGreater(int(panel.std()), 0)
        self.assertEqual(nearest_video_frame_index(5, 10, 101), 50)

    def test_tiny_synthetic_end_to_end_best_resume_and_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "experiment"
            root.mkdir()
            manifest_rows = []
            infos = []
            for session_id, layout in zip(SESSION_IDS, LAYOUTS):
                trial_id = f"trial-{session_id}"
                info = _session_info(root, session_id, trial_id, layout)
                frame = _full_frame()
                if session_id == SESSION_IDS[0]:
                    frame.loc[250, "snout_position_valid"] = False
                    frame.loc[250, ["snout_desk_x_cm", "snout_desk_y_cm", "snout_desk_z_cm"]] = np.nan
                    frame.loc[250, ["snout_cam_x_cm", "snout_cam_y_cm", "snout_cam_z_cm"]] = np.nan
                frame.to_csv(info.csv_path, index=False)
                _write_test_video(info.video_path)
                manifest_rows.append(
                    {
                        "session": session_id,
                        "trial_id": trial_id,
                        "layout": layout,
                        "host_duration_s": 50,
                        "readings": len(frame),
                        "sync_match_percent": 100,
                        "sync_p95_ms": 1,
                        "analysis_status": "usable",
                    }
                )
                infos.append(info)
            pd.DataFrame(manifest_rows).to_csv(root / "trial_manifest.csv", index=False)
            for info in infos:
                polygons = {}
                if "mint" in info.source_names:
                    polygons["mint"] = np.array(
                        [[-10, -2], [-2, -2], [-2, 2], [-10, 2]]
                    )
                if "lavender" in info.source_names:
                    polygons["lavender"] = np.array(
                        [[2, -2], [10, -2], [10, 2], [2, 2]]
                    )
                write_annotation(
                    info,
                    0,
                    499,
                    canonical_paper_corners(),
                    polygons,
                    diagnostics={"qc_raw_row_at_save": 250},
                )
            first = infos[0]
            output = Path(temporary) / "run-output"
            config = RunConfig(
                architecture="temporal-cnn",
                temporal_mode="causal",
                target="position",
                session=SESSION_IDS[0],
                epochs=2,
                batch_size=128,
                device="cpu",
                output_directory=str(output),
                experiment_root=str(root),
            )
            video_path = output / "video.mp4"
            with mock.patch(
                "analysis.spatial_sequence.training.position_video_path",
                return_value=video_path,
            ):
                result = train_run(config, render_position_video=True)
            self.assertEqual(result, output.resolve())
            for name in (
                "best.pt",
                "final.pt",
                "history.csv",
                "metrics.json",
                "aggregated_predictions.npz",
                "diagnostics.png",
            ):
                self.assertTrue((output / name).is_file(), name)
            checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
            final_checkpoint = torch.load(
                output / "final.pt",
                map_location="cpu",
                weights_only=False,
            )
            history = final_checkpoint["history"]
            selected = None
            for row in history:
                candidate = (
                    row["selection_metric_value"],
                    row["anchor_validation_loss"],
                    row["epoch"],
                )
                if selected is None or selection_is_better(
                    *candidate,
                    selected[0],
                    selected[1],
                    selected[2],
                    row["selection_direction"],
                ):
                    selected = candidate
            assert selected is not None
            self.assertEqual(final_checkpoint["best_epoch"], selected[2])
            self.assertEqual(checkpoint["epoch"], selected[2])
            self.assertAlmostEqual(final_checkpoint["best_loss"], selected[1])
            self.assertIn("optimizer", checkpoint)
            self.assertIn("scheduler", checkpoint)
            self.assertIn("rng_state", checkpoint)
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertTrue(metrics["exploratory_validation_only"])

            with np.load(output / "aggregated_predictions.npz") as aggregate:
                missing = np.flatnonzero(aggregate["raw_row"] == 250)
                self.assertEqual(len(missing), 1)
                self.assertFalse(bool(np.all(aggregate["mask"][missing[0]])))
                self.assertTrue(np.all(np.isnan(aggregate["truth_xy_cm"][missing[0]])))
            self.assertTrue(video_path.is_file())
            rendered = cv2.VideoCapture(str(video_path))
            self.assertEqual(int(rendered.get(cv2.CAP_PROP_FRAME_COUNT)), 57)
            ok, rendered_frame = rendered.read()
            self.assertTrue(ok)
            warning_pixels = (
                (rendered_frame[:, :, 2] > 100)
                & (rendered_frame[:, :, 1] < 90)
                & (rendered_frame[:, :, 0] < 90)
            )
            self.assertGreater(int(np.sum(warning_pixels)), 100)
            rendered.release()
            resumed = train_run(config, resume_checkpoint=output / "final.pt")
            self.assertEqual(resumed, output.resolve())
            incompatible = RunConfig(
                architecture="transformer",
                temporal_mode="causal",
                target="position",
                session=SESSION_IDS[0],
                epochs=2,
                batch_size=128,
                device="cpu",
                output_directory=str(output),
                experiment_root=str(root),
            )
            with self.assertRaisesRegex(ValueError, "incompatible"):
                train_run(incompatible, resume_checkpoint=output / "final.pt")
            with self.assertRaises(FileExistsError):
                train_run(config)

            short_output = Path(temporary) / "short-output"
            short_config = RunConfig(
                architecture="temporal-cnn",
                temporal_mode="bidirectional",
                target="area",
                sequence_length=6,
                epochs=1,
                batch_size=256,
                device="cpu",
                output_directory=str(short_output),
                experiment_root=str(root),
            )
            train_run(short_config)
            short_signature = json.loads(
                (short_output / "data_signature.json").read_text()
            )
            self.assertEqual(short_signature["normalization_method"], "unique-training-physical-rows")
            self.assertEqual(short_signature["fixed_topology"]["sequence_length"], 6)
            self.assertFalse((short_output / "diagnostics.png").exists())
            with np.load(short_output / "aggregated_predictions.npz") as aggregate:
                self.assertEqual(len(aggregate["raw_row"]), 5 * 57)

            loso_output = Path(temporary) / "loso-output"
            held_out = infos[0].run_directory.name
            loso_config = RunConfig(
                architecture="temporal-cnn",
                temporal_mode="bidirectional",
                target="distance",
                evaluation_scheme="leave-one-session-out",
                held_out_session=held_out,
                epochs=1,
                batch_size=256,
                device="cpu",
                output_directory=str(loso_output),
                experiment_root=str(root),
            )
            train_run(loso_config)
            loso_metrics = json.loads((loso_output / "metrics.json").read_text())
            self.assertEqual(loso_metrics["evaluation_scheme"], "leave-one-session-out")
            self.assertEqual(loso_metrics["held_out_session"], held_out)
            self.assertGreater(loso_metrics["per_session"][held_out]["mint"]["count"], 0)
            split = json.loads((loso_output / "split_summary.json").read_text())
            for slug, summary in split["sessions"].items():
                expected_split = "validation" if slug == held_out else "train"
                self.assertEqual(summary["blocks"], [
                    {
                        "active_start_inclusive": 0,
                        "active_stop_exclusive": 500,
                        "fragment": 0,
                        "row_count": 500,
                        "split": expected_split,
                    }
                ])
            self.assertFalse((loso_output / "diagnostics.png").exists())


if __name__ == "__main__":
    unittest.main()
