"""Sensor-level sanity checks for the long random-trajectory pilot.

The functions in this module deliberately operate on physical readings and
paper coordinates.  They do not read model predictions or change the training
protocol.  Notebook 10 is a thin, reproducible presentation layer over these
tested helpers.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as PolygonPatch

from .core import (
    EXPERIMENT_ROOT,
    MIN_STANDARD_DEVIATION,
    PAPER_HALF_CM,
    PAPER_SIZE_CM,
    SENSOR_COLUMNS,
    SOURCE_NAMES,
    SessionData,
    SessionInfo,
    SpatialAnnotation,
    _build_session_data,
    discover_sessions,
    distances_to_polygon,
    load_annotation,
    paper_contains,
    points_in_polygon,
)


CELL_SIZE_CM = 0.5
GRID_SPACING_CM = 0.5
KERNEL_RADIUS_CM = 2.5
KERNEL_SIGMA_CM = 1.25
KERNEL_MIN_NEIGHBORS = 3
KERNEL_MIN_ESS = 2.0
ROBUST_DISPLAY_LIMIT = 3.0
VISIT_GAP_S = 1.5
ENCOUNTER_MERGE_GAP_S = 3.0
ENCOUNTER_ROWS = 10
DISTANCE_BIN_CM = 1.0
SOURCE_COLORS = {"mint": "#22a968", "lavender": "#9b59c6"}
SESSION_COLORS = ("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e")


@dataclass(frozen=True)
class SanitySession:
    data: SessionData
    annotation: SpatialAnnotation
    sample_time_s: np.ndarray

    @property
    def info(self) -> SessionInfo:
        return self.data.info

    @property
    def slug(self) -> str:
        return self.info.run_directory.name


@dataclass(frozen=True)
class CellBalancedData:
    cell_ids: np.ndarray
    cell_xy_cm: np.ndarray
    values: np.ndarray
    counts: np.ndarray
    cell_ix: np.ndarray
    cell_iy: np.ndarray


@dataclass(frozen=True)
class KernelOperator:
    x_axis_cm: np.ndarray
    y_axis_cm: np.ndarray
    normalized_weights: np.ndarray
    neighbor_count: np.ndarray
    effective_sample_size: np.ndarray
    supported: np.ndarray


@dataclass(frozen=True)
class PCAFit:
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = _finite_matrix(values, "PCA transform values")
        if values.shape[1] != self.mean.size:
            raise ValueError("PCA transform feature count does not match the fitted model")
        return ((values - self.mean) / self.scale) @ self.components.T


@dataclass(frozen=True)
class Visit:
    cell_id: int
    indices: np.ndarray


@dataclass(frozen=True)
class SourceEncounter:
    first_inside: int
    final_inside: int


@dataclass(frozen=True)
class EncounterWindow:
    first_inside: int
    final_inside: int
    entry_indices: np.ndarray
    exit_indices: np.ndarray


@dataclass(frozen=True)
class SanityArtifacts:
    output_directory: Path
    figures: tuple[Path, ...]
    tables: dict[str, pd.DataFrame]
    summary: dict[str, Any]


def _finite_matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or not len(array):
        raise ValueError(f"{name} must be a non-empty vector or matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def load_sanity_sessions(
    experiment_root: Path = EXPERIMENT_ROOT,
    verify_hashes: bool = True,
) -> list[SanitySession]:
    """Load the five annotated flag-2 sessions without model normalization."""
    result: list[SanitySession] = []
    for info in discover_sessions(experiment_root):
        annotation = load_annotation(info, verify_hashes=verify_hashes)
        data = _build_session_data(info, annotation)
        timestamp_column = "pcnose_sample_time_estimate_utc"
        if timestamp_column not in data.frame:
            raise ValueError(f"Session {info.session_id} is missing {timestamp_column}")
        timestamps = pd.to_datetime(
            data.frame[timestamp_column], utc=True, errors="raise", format="mixed"
        )
        nanoseconds = timestamps.astype("int64").to_numpy(dtype=np.int64)
        sample_time_s = (nanoseconds - nanoseconds[0]).astype(np.float64) / 1e9
        if len(sample_time_s) > 1 and np.any(np.diff(sample_time_s) <= 0):
            raise ValueError(f"Sample timestamps are not strictly increasing for {info.session_id}")
        result.append(SanitySession(data, annotation, sample_time_s))
    if len(result) != 5:
        raise ValueError(f"Expected five sanity-check sessions, found {len(result)}")
    return result


def spatial_cell_ids(
    points_xy_cm: np.ndarray,
    cell_size_cm: float = CELL_SIZE_CM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return canonical paper cell IDs and x/y indices; outside points are -1."""
    points = np.asarray(points_xy_cm, dtype=np.float64).reshape(-1, 2)
    if cell_size_cm <= 0 or not np.isclose(PAPER_SIZE_CM / cell_size_cm, round(PAPER_SIZE_CM / cell_size_cm)):
        raise ValueError("Cell size must divide the canonical paper width exactly")
    count = int(round(PAPER_SIZE_CM / cell_size_cm))
    inside = np.all(np.isfinite(points), axis=1) & paper_contains(
        np.nan_to_num(points, nan=2 * PAPER_HALF_CM)
    )
    ix = np.full(len(points), -1, dtype=np.int64)
    iy = np.full(len(points), -1, dtype=np.int64)
    scaled = np.floor((points[inside] + PAPER_HALF_CM) / cell_size_cm).astype(np.int64)
    scaled = np.clip(scaled, 0, count - 1)
    ix[inside] = scaled[:, 0]
    iy[inside] = scaled[:, 1]
    cell_ids = np.full(len(points), -1, dtype=np.int64)
    cell_ids[inside] = iy[inside] * count + ix[inside]
    return cell_ids, ix, iy


def balance_spatial_cells(
    points_xy_cm: np.ndarray,
    values: np.ndarray,
    cell_size_cm: float = CELL_SIZE_CM,
) -> CellBalancedData:
    """Collapse all observations in each paper cell to one median observation."""
    points = np.asarray(points_xy_cm, dtype=np.float64).reshape(-1, 2)
    matrix = _finite_matrix(values, "cell values")
    if len(points) != len(matrix):
        raise ValueError("Point and value counts differ")
    ids, ix, iy = spatial_cell_ids(points, cell_size_cm)
    valid = ids >= 0
    if not np.any(valid):
        raise ValueError("No observations fall inside the canonical paper")
    unique = np.unique(ids[valid])
    cell_xy: list[np.ndarray] = []
    cell_values: list[np.ndarray] = []
    counts: list[int] = []
    cell_ix: list[int] = []
    cell_iy: list[int] = []
    for cell_id in unique:
        selection = ids == cell_id
        cell_xy.append(np.median(points[selection], axis=0))
        cell_values.append(np.median(matrix[selection], axis=0))
        counts.append(int(np.sum(selection)))
        first = int(np.flatnonzero(selection)[0])
        cell_ix.append(int(ix[first]))
        cell_iy.append(int(iy[first]))
    return CellBalancedData(
        cell_ids=unique,
        cell_xy_cm=np.asarray(cell_xy, dtype=np.float64),
        values=np.asarray(cell_values, dtype=np.float64),
        counts=np.asarray(counts, dtype=np.int64),
        cell_ix=np.asarray(cell_ix, dtype=np.int64),
        cell_iy=np.asarray(cell_iy, dtype=np.int64),
    )


def scaled_mad(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    matrix = _finite_matrix(values, "robust-scale values")
    median = np.median(matrix, axis=axis)
    scale = 1.4826 * np.median(np.abs(matrix - median), axis=axis)
    if np.any(~np.isfinite(scale)) or np.any(scale < MIN_STANDARD_DEVIATION):
        bad = np.flatnonzero(np.atleast_1d(scale) < MIN_STANDARD_DEVIATION)
        raise ValueError(f"Robust scale is too small for feature indices {bad.tolist()}")
    return np.asarray(median, dtype=np.float64), np.asarray(scale, dtype=np.float64)


def robust_zscore(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = _finite_matrix(values, "robust-z values")
    median, scale = scaled_mad(matrix)
    return (matrix - median) / scale, median, scale


def build_kernel_operator(
    reference_xy_cm: np.ndarray,
    grid_spacing_cm: float = GRID_SPACING_CM,
    radius_cm: float = KERNEL_RADIUS_CM,
    sigma_cm: float = KERNEL_SIGMA_CM,
    min_neighbors: int = KERNEL_MIN_NEIGHBORS,
    min_effective_sample_size: float = KERNEL_MIN_ESS,
) -> KernelOperator:
    points = np.asarray(reference_xy_cm, dtype=np.float64).reshape(-1, 2)
    if not len(points) or not np.all(np.isfinite(points)) or not np.all(paper_contains(points)):
        raise ValueError("Kernel reference points must be finite and on paper")
    if min_neighbors < 1 or radius_cm <= 0 or sigma_cm <= 0 or grid_spacing_cm <= 0:
        raise ValueError("Invalid Gaussian-kernel parameters")
    axis = np.arange(
        -PAPER_HALF_CM,
        PAPER_HALF_CM + grid_spacing_cm * 0.25,
        grid_spacing_cm,
        dtype=np.float64,
    )
    grid_x, grid_y = np.meshgrid(axis, axis)
    queries = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    squared_distance = np.sum((queries[:, None, :] - points[None, :, :]) ** 2, axis=2)
    within = squared_distance <= radius_cm**2 + 1e-12
    weights = np.exp(-0.5 * squared_distance / sigma_cm**2) * within
    weight_sum = weights.sum(axis=1)
    squared_weight_sum = np.square(weights).sum(axis=1)
    neighbor_count = within.sum(axis=1)
    ess = np.divide(
        np.square(weight_sum),
        squared_weight_sum,
        out=np.zeros_like(weight_sum),
        where=squared_weight_sum > 0,
    )
    supported = (neighbor_count >= min_neighbors) & (ess >= min_effective_sample_size)
    normalized = np.zeros_like(weights)
    normalized[supported] = weights[supported] / weight_sum[supported, None]
    shape = grid_x.shape
    return KernelOperator(
        x_axis_cm=axis,
        y_axis_cm=axis,
        normalized_weights=normalized,
        neighbor_count=neighbor_count.reshape(shape),
        effective_sample_size=ess.reshape(shape),
        supported=supported.reshape(shape),
    )


def apply_kernel(operator: KernelOperator, values: np.ndarray) -> np.ndarray:
    matrix = _finite_matrix(values, "kernel values")
    if len(matrix) != operator.normalized_weights.shape[1]:
        raise ValueError("Kernel value count differs from the reference-point count")
    result = operator.normalized_weights @ matrix
    shape = (*operator.supported.shape, matrix.shape[1])
    result = result.reshape(shape)
    result[~operator.supported] = np.nan
    return result


def kernel_support_at(
    query_xy_cm: np.ndarray,
    reference_xy_cm: np.ndarray,
    radius_cm: float = KERNEL_RADIUS_CM,
    sigma_cm: float = KERNEL_SIGMA_CM,
    min_neighbors: int = KERNEL_MIN_NEIGHBORS,
    min_effective_sample_size: float = KERNEL_MIN_ESS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    queries = np.asarray(query_xy_cm, dtype=np.float64).reshape(-1, 2)
    reference = np.asarray(reference_xy_cm, dtype=np.float64).reshape(-1, 2)
    if not len(queries) or not len(reference) or not np.all(np.isfinite(queries)) or not np.all(np.isfinite(reference)):
        raise ValueError("Kernel support queries and references must be non-empty and finite")
    squared_distance = np.sum((queries[:, None, :] - reference[None, :, :]) ** 2, axis=2)
    within = squared_distance <= radius_cm**2 + 1e-12
    weights = np.exp(-0.5 * squared_distance / sigma_cm**2) * within
    neighbor_count = within.sum(axis=1)
    denominator = np.square(weights).sum(axis=1)
    ess = np.divide(
        np.square(weights.sum(axis=1)),
        denominator,
        out=np.zeros(len(queries), dtype=np.float64),
        where=denominator > 0,
    )
    supported = (neighbor_count >= min_neighbors) & (ess >= min_effective_sample_size)
    return supported, neighbor_count, ess


def weighted_quantile(
    values: np.ndarray,
    quantiles: float | Sequence[float],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    requested = np.atleast_1d(np.asarray(quantiles, dtype=np.float64))
    if not len(values) or np.any(~np.isfinite(values)) or np.any((requested < 0) | (requested > 1)):
        raise ValueError("Weighted quantiles require finite values and quantiles in [0, 1]")
    active_weights = np.ones(len(values), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(active_weights) != len(values) or np.any(~np.isfinite(active_weights)) or np.any(active_weights < 0) or active_weights.sum() <= 0:
        raise ValueError("Invalid quantile weights")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = active_weights[order]
    centers = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    centers /= sorted_weights.sum()
    result = np.interp(requested, centers, sorted_values, left=sorted_values[0], right=sorted_values[-1])
    return result[0] if np.ndim(quantiles) == 0 else result


def equal_group_weights(groups: Sequence[str]) -> np.ndarray:
    labels = np.asarray(groups, dtype=object).reshape(-1)
    if not len(labels):
        raise ValueError("At least one group is required")
    unique, counts = np.unique(labels, return_counts=True)
    per_group = {name: 1.0 / (len(unique) * count) for name, count in zip(unique, counts)}
    return np.asarray([per_group[name] for name in labels], dtype=np.float64)


def fit_weighted_pca(
    values: np.ndarray,
    weights: np.ndarray | None = None,
    component_count: int = 3,
) -> PCAFit:
    matrix = _finite_matrix(values, "PCA values")
    if not 1 <= component_count <= matrix.shape[1]:
        raise ValueError("Invalid PCA component count")
    active_weights = np.ones(len(matrix), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(active_weights) != len(matrix) or np.any(~np.isfinite(active_weights)) or np.any(active_weights < 0) or active_weights.sum() <= 0:
        raise ValueError("Invalid PCA sample weights")
    active_weights = active_weights / active_weights.sum()
    mean = np.sum(matrix * active_weights[:, None], axis=0)
    variance = np.sum(np.square(matrix - mean) * active_weights[:, None], axis=0)
    scale = np.sqrt(variance)
    if np.any(scale < MIN_STANDARD_DEVIATION):
        bad = np.flatnonzero(scale < MIN_STANDARD_DEVIATION)
        raise ValueError(f"PCA standard deviation is too small for feature indices {bad.tolist()}")
    standardized = (matrix - mean) / scale
    covariance = (standardized * active_weights[:, None]).T @ standardized
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = eigenvectors[:, order].T[:component_count]
    for component in components:
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0:
            component *= -1
    explained = eigenvalues[:component_count]
    total = float(eigenvalues.sum())
    ratios = explained / total if total > 0 else np.zeros_like(explained)
    return PCAFit(mean, scale, components, explained, ratios)


def fit_equal_session_pca(
    session_values: Sequence[np.ndarray],
    component_count: int = 3,
) -> tuple[PCAFit, list[np.ndarray]]:
    matrices = [_finite_matrix(values, "session PCA values") for values in session_values]
    if not matrices or len({matrix.shape[1] for matrix in matrices}) != 1:
        raise ValueError("PCA sessions must be non-empty with matching feature counts")
    combined = np.concatenate(matrices, axis=0)
    labels = np.concatenate(
        [np.full(len(matrix), index, dtype=np.int64) for index, matrix in enumerate(matrices)]
    )
    weights = equal_group_weights(labels.astype(str))
    fit = fit_weighted_pca(combined, weights, component_count)
    return fit, [fit.transform(matrix) for matrix in matrices]


def symmetric_score_limits(
    session_scores: Sequence[np.ndarray],
    quantile: float = 0.99,
    equal_session: bool = True,
) -> np.ndarray:
    matrices = [_finite_matrix(values, "PCA scores") for values in session_scores]
    combined = np.concatenate(matrices, axis=0)
    if equal_session:
        labels = np.concatenate(
            [np.full(len(matrix), index, dtype=np.int64) for index, matrix in enumerate(matrices)]
        )
        weights = equal_group_weights(labels.astype(str))
    else:
        weights = np.ones(len(combined), dtype=np.float64)
    limits = np.asarray(
        [weighted_quantile(np.abs(combined[:, column]), quantile, weights) for column in range(combined.shape[1])]
    )
    if np.any(limits < MIN_STANDARD_DEVIATION):
        raise ValueError("PCA RGB limit is too small")
    return limits


def scores_to_rgb(scores: np.ndarray, limits: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    limits = np.asarray(limits, dtype=np.float64).reshape(-1)
    if values.shape[-1] != 3 or limits.shape != (3,) or np.any(limits <= 0):
        raise ValueError("RGB mapping requires three scores and three positive limits")
    return np.clip(0.5 + 0.5 * values / limits, 0.0, 1.0)


def segment_cell_visits(
    cell_ids: np.ndarray,
    timestamps_s: np.ndarray,
    valid: np.ndarray | None = None,
    maximum_gap_s: float = VISIT_GAP_S,
) -> list[Visit]:
    ids = np.asarray(cell_ids, dtype=np.int64).reshape(-1)
    timestamps = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    active = ids >= 0 if valid is None else (ids >= 0) & np.asarray(valid, dtype=bool).reshape(-1)
    if len(ids) != len(timestamps) or len(active) != len(ids) or np.any(~np.isfinite(timestamps)):
        raise ValueError("Invalid visit arrays")
    visits: list[Visit] = []
    start: int | None = None
    for index in range(len(ids)):
        continuation = (
            start is not None
            and active[index]
            and active[index - 1]
            and ids[index] == ids[index - 1]
            and timestamps[index] - timestamps[index - 1] <= maximum_gap_s
        )
        if continuation:
            continue
        if start is not None:
            visits.append(Visit(int(ids[start]), np.arange(start, index, dtype=np.int64)))
            start = None
        if active[index]:
            start = index
    if start is not None:
        visits.append(Visit(int(ids[start]), np.arange(start, len(ids), dtype=np.int64)))
    return visits


def normalized_wasserstein_1d(
    first: np.ndarray,
    second: np.ndarray,
    scale: float,
) -> float:
    first = np.sort(np.asarray(first, dtype=np.float64).reshape(-1))
    second = np.sort(np.asarray(second, dtype=np.float64).reshape(-1))
    if not len(first) or not len(second) or np.any(~np.isfinite(first)) or np.any(~np.isfinite(second)):
        raise ValueError("Wasserstein samples must be non-empty and finite")
    if not np.isfinite(scale) or scale < MIN_STANDARD_DEVIATION:
        raise ValueError("Wasserstein normalization scale is too small")
    support = np.sort(np.concatenate([first, second]))
    if len(support) < 2:
        return 0.0
    deltas = np.diff(support)
    first_cdf = np.searchsorted(first, support[:-1], side="right") / len(first)
    second_cdf = np.searchsorted(second, support[:-1], side="right") / len(second)
    return float(np.sum(np.abs(first_cdf - second_cdf) * deltas) / scale)


def group_source_encounters(
    inside: np.ndarray,
    timestamps_s: np.ndarray,
    merge_gap_s: float = ENCOUNTER_MERGE_GAP_S,
) -> list[SourceEncounter]:
    membership = np.asarray(inside, dtype=bool).reshape(-1)
    timestamps = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    if len(membership) != len(timestamps) or np.any(~np.isfinite(timestamps)):
        raise ValueError("Invalid encounter arrays")
    hits = np.flatnonzero(membership)
    if not len(hits):
        return []
    result: list[SourceEncounter] = []
    first = int(hits[0])
    previous = int(hits[0])
    for current_value in hits[1:]:
        current = int(current_value)
        if timestamps[current] - timestamps[previous] > merge_gap_s:
            result.append(SourceEncounter(first, previous))
            first = current
        previous = current
    result.append(SourceEncounter(first, previous))
    return result


def validate_encounter_windows(
    encounters: Sequence[SourceEncounter],
    eligible: np.ndarray,
    same_source_inside: np.ndarray,
    other_source_inside: np.ndarray | None = None,
    row_count: int = ENCOUNTER_ROWS,
) -> tuple[list[EncounterWindow], list[dict[str, Any]]]:
    eligible = np.asarray(eligible, dtype=bool).reshape(-1)
    same = np.asarray(same_source_inside, dtype=bool).reshape(-1)
    other = np.zeros_like(same) if other_source_inside is None else np.asarray(other_source_inside, dtype=bool).reshape(-1)
    if len(eligible) != len(same) or len(other) != len(same) or row_count < 2:
        raise ValueError("Invalid encounter-window arrays")
    accepted: list[EncounterWindow] = []
    records: list[dict[str, Any]] = []
    for encounter_index, encounter in enumerate(encounters):
        entry_start = encounter.first_inside - (row_count - 1)
        exit_stop = encounter.final_inside + row_count
        reason = "accepted"
        if entry_start < 0 or exit_stop > len(same):
            reason = "boundary"
        else:
            entry = np.arange(entry_start, encounter.first_inside + 1, dtype=np.int64)
            exit_indices = np.arange(encounter.final_inside, exit_stop, dtype=np.int64)
            full = np.arange(entry_start, exit_stop, dtype=np.int64)
            if not np.all(eligible[np.unique(np.concatenate([entry, exit_indices]))]):
                reason = "ineligible_row"
            elif np.any(same[entry[:-1]]) or np.any(same[exit_indices[1:]]):
                reason = "unclean_same_source"
            elif np.any(other[full]):
                reason = "other_source_contact"
            else:
                accepted.append(
                    EncounterWindow(
                        encounter.first_inside,
                        encounter.final_inside,
                        entry,
                        exit_indices,
                    )
                )
        records.append(
            {
                "encounter_index": encounter_index,
                "first_inside": encounter.first_inside,
                "final_inside": encounter.final_inside,
                "status": reason,
            }
        )
    return accepted, records


def paired_distance_bin_differences(
    entry_distance_cm: np.ndarray,
    exit_distance_cm: np.ndarray,
    entry_values: np.ndarray,
    exit_values: np.ndarray,
    bin_width_cm: float = DISTANCE_BIN_CM,
) -> dict[int, np.ndarray]:
    entry_distance = np.asarray(entry_distance_cm, dtype=np.float64).reshape(-1)
    exit_distance = np.asarray(exit_distance_cm, dtype=np.float64).reshape(-1)
    entry_matrix = _finite_matrix(entry_values, "entry values")
    exit_matrix = _finite_matrix(exit_values, "exit values")
    if len(entry_distance) != len(entry_matrix) or len(exit_distance) != len(exit_matrix) or entry_matrix.shape[1] != exit_matrix.shape[1]:
        raise ValueError("Distance and value arrays do not align")
    if np.any(entry_distance < 0) or np.any(exit_distance < 0) or bin_width_cm <= 0:
        raise ValueError("Distances and bin width must be non-negative")
    entry_bins = np.floor(entry_distance / bin_width_cm).astype(np.int64)
    exit_bins = np.floor(exit_distance / bin_width_cm).astype(np.int64)
    common = np.intersect1d(np.unique(entry_bins), np.unique(exit_bins))
    return {
        int(bin_index): (
            np.median(exit_matrix[exit_bins == bin_index], axis=0)
            - np.median(entry_matrix[entry_bins == bin_index], axis=0)
        )
        for bin_index in common
    }


def _source_membership(session: SanitySession, source: str) -> np.ndarray:
    polygon = session.annotation.source_polygons_paper_cm.get(source)
    membership = np.zeros(len(session.data.sensors), dtype=bool)
    if polygon is None:
        return membership
    valid = session.data.pose_mask & paper_contains(session.data.paper_xy_cm)
    membership[valid] = points_in_polygon(session.data.paper_xy_cm[valid], polygon)
    return membership


def _cell_balanced_sessions(sessions: Sequence[SanitySession]) -> list[CellBalancedData]:
    result: list[CellBalancedData] = []
    for session in sessions:
        valid = session.data.pose_mask & paper_contains(session.data.paper_xy_cm)
        result.append(balance_spatial_cells(session.data.paper_xy_cm[valid], session.data.sensors[valid]))
    return result


def _save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _figure_path(output_directory: Path, name: str) -> Path:
    return output_directory / "figures" / f"{name}.png"


def _save_figure(figure: matplotlib.figure.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _source_legend(session: SanitySession) -> list[Line2D]:
    return [
        Line2D([0], [0], color=SOURCE_COLORS[name], linestyle="--", linewidth=1.2, label=name)
        for name in session.info.source_names
    ]


def _decorate_spatial_axis(axis: matplotlib.axes.Axes, session: SanitySession) -> None:
    for source, polygon in session.annotation.source_polygons_paper_cm.items():
        axis.add_patch(
            PolygonPatch(
                polygon,
                closed=True,
                fill=False,
                edgecolor=SOURCE_COLORS[source],
                linewidth=1.0,
                linestyle="--",
                zorder=5,
            )
        )
    axis.set_xlim(-PAPER_HALF_CM, PAPER_HALF_CM)
    axis.set_ylim(PAPER_HALF_CM, -PAPER_HALF_CM)
    axis.set_aspect("equal")
    axis.set_xticks([-10, 0, 10])
    axis.set_yticks([-10, 0, 10])


def build_integrity_table(sessions: Sequence[SanitySession]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for session in sessions:
        data = session.data
        delta = np.diff(session.sample_time_s)
        on_paper = data.pose_mask & paper_contains(data.paper_xy_cm)
        records.append(
            {
                "session": session.slug,
                "session_id": session.info.session_id,
                "usable_rows": session.annotation.usable_end_row - session.annotation.usable_start_row + 1,
                "flag2_rows": len(data.sensors),
                "finite_sensor_rows": int(np.sum(np.all(np.isfinite(data.sensors), axis=1))),
                "pose_valid_rows": int(data.pose_mask.sum()),
                "pose_missing_rows": int((~data.pose_mask).sum()),
                "on_paper_rows": int(on_paper.sum()),
                "off_paper_pose_rows": int((data.pose_mask & ~on_paper).sum()),
                "duration_s": float(session.sample_time_s[-1] - session.sample_time_s[0]),
                "sample_interval_p01_s": float(np.quantile(delta, 0.01)),
                "sample_interval_median_s": float(np.median(delta)),
                "sample_interval_p99_s": float(np.quantile(delta, 0.99)),
                "sample_interval_max_s": float(np.max(delta)),
                "gaps_over_1_5_s": int(np.sum(delta > VISIT_GAP_S)),
                "timestamp_strictly_increasing": bool(np.all(delta > 0)),
            }
        )
    return pd.DataFrame.from_records(records)


def build_channel_distribution_table(sessions: Sequence[SanitySession]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    quantiles = (0.01, 0.25, 0.5, 0.75, 0.99)
    for session in sessions:
        summary = np.quantile(session.data.sensors.astype(np.float64), quantiles, axis=0)
        for channel_index, channel in enumerate(SENSOR_COLUMNS):
            records.append(
                {
                    "session": session.slug,
                    "channel": channel.replace("pcnose_", "").replace("_kohm", ""),
                    "q01_kohm": float(summary[0, channel_index]),
                    "q25_kohm": float(summary[1, channel_index]),
                    "median_kohm": float(summary[2, channel_index]),
                    "q75_kohm": float(summary[3, channel_index]),
                    "q99_kohm": float(summary[4, channel_index]),
                }
            )
    return pd.DataFrame.from_records(records)


def render_channel_distribution(
    sessions: Sequence[SanitySession],
    table: pd.DataFrame,
    output_directory: Path,
) -> Path:
    figure, axes = plt.subplots(4, 8, figsize=(22, 13), constrained_layout=False)
    for channel_index, axis in enumerate(axes.flat):
        channel = f"S{channel_index + 1}"
        selected = table.loc[table["channel"] == channel].set_index("session").loc[
            [session.slug for session in sessions]
        ]
        x = np.arange(len(sessions))
        for session_index, (_, row) in enumerate(selected.iterrows()):
            color = SESSION_COLORS[session_index]
            axis.vlines(session_index, row.q01_kohm, row.q99_kohm, color=color, linewidth=0.8, alpha=0.8)
            axis.vlines(session_index, row.q25_kohm, row.q75_kohm, color=color, linewidth=4.0, alpha=0.9)
            axis.scatter(session_index, row.median_kohm, color=color, s=16, zorder=3)
        axis.set_title(channel, fontsize=10)
        axis.set_xticks(x)
        axis.set_xticklabels([])
        axis.grid(axis="y", alpha=0.18, linewidth=0.5)
        if channel_index % 8 == 0:
            axis.set_ylabel("kΩ")
    handles = [
        Line2D([0], [0], color=SESSION_COLORS[index], marker="o", label=session.slug, linewidth=2)
        for index, session in enumerate(sessions)
    ]
    figure.suptitle("Raw Channel Distributions", fontsize=18, y=0.995)
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    figure.subplots_adjust(left=0.055, right=0.995, top=0.95, bottom=0.10, wspace=0.30, hspace=0.35)
    return _save_figure(figure, _figure_path(output_directory, "raw_channel_distributions"))


def render_concatenated_time_traces(
    sessions: Sequence[SanitySession],
    output_directory: Path,
) -> Path:
    figure, axes = plt.subplots(4, 8, figsize=(24, 13), sharex=True, constrained_layout=False)
    boundaries: list[float] = []
    offset = 0.0
    traces: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    for session_index, session in enumerate(sessions):
        elapsed = session.sample_time_s - session.sample_time_s[0]
        x = elapsed + offset
        traces.append((x, session.data.sensors, session.slug, SESSION_COLORS[session_index]))
        offset = float(x[-1])
        boundaries.append(offset)
    for channel_index, axis in enumerate(axes.flat):
        for x, sensors, _, color in traces:
            axis.plot(x, sensors[:, channel_index], color=color, linewidth=0.65, alpha=0.88)
        for boundary in boundaries[:-1]:
            axis.axvline(boundary, color="0.25", linewidth=0.7, alpha=0.7)
        values = np.concatenate([session.data.sensors[:, channel_index] for session in sessions])
        lower, upper = float(np.min(values)), float(np.max(values))
        padding = max((upper - lower) * 0.04, 1e-5)
        axis.set_ylim(lower - padding, upper + padding)
        axis.set_title(f"S{channel_index + 1}", fontsize=10)
        axis.grid(alpha=0.14, linewidth=0.5)
        if channel_index % 8 == 0:
            axis.set_ylabel("kΩ")
        if channel_index >= 24:
            axis.set_xlabel("Concatenated elapsed time (s)")
    handles = [
        Line2D([0], [0], color=SESSION_COLORS[index], label=session.slug, linewidth=2)
        for index, session in enumerate(sessions)
    ]
    figure.suptitle("Raw Sensor Readings Across Concatenated Sessions", fontsize=18, y=0.995)
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    figure.subplots_adjust(left=0.055, right=0.995, top=0.95, bottom=0.11, wspace=0.30, hspace=0.35)
    return _save_figure(figure, _figure_path(output_directory, "raw_readings_concatenated_time"))


def _equal_session_channel_limits(
    cells: Sequence[CellBalancedData],
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([cell.values for cell in cells], axis=0)
    labels = np.concatenate(
        [np.full(len(cell.values), index, dtype=np.int64) for index, cell in enumerate(cells)]
    )
    weights = equal_group_weights(labels.astype(str))
    lower = np.asarray(
        [weighted_quantile(values[:, channel], lower_quantile, weights) for channel in range(values.shape[1])]
    )
    upper = np.asarray(
        [weighted_quantile(values[:, channel], upper_quantile, weights) for channel in range(values.shape[1])]
    )
    too_small = upper - lower < MIN_STANDARD_DEVIATION
    if np.any(too_small):
        upper[too_small] = lower[too_small] + MIN_STANDARD_DEVIATION
    return lower, upper


def render_channel_spatial_maps(
    sessions: Sequence[SanitySession],
    cells: Sequence[CellBalancedData],
    operators: Sequence[KernelOperator],
    output_directory: Path,
) -> tuple[list[Path], pd.DataFrame]:
    raw_lower, raw_upper = _equal_session_channel_limits(cells)
    paths: list[Path] = []
    scale_records: list[dict[str, Any]] = []
    for channel_index in range(32):
        for session in sessions:
            scale_records.append(
                {
                    "session": session.slug,
                    "channel": f"S{channel_index + 1}",
                    "raw_vmin_kohm": float(raw_lower[channel_index]),
                    "raw_vmax_kohm": float(raw_upper[channel_index]),
                    "robust_vmin": -ROBUST_DISPLAY_LIMIT,
                    "robust_vmax": ROBUST_DISPLAY_LIMIT,
                }
            )
    for session, cell, operator in zip(sessions, cells, operators):
        raw_surface = apply_kernel(operator, cell.values)
        robust_values, _, _ = robust_zscore(cell.values)
        robust_surface = apply_kernel(operator, robust_values)

        figure, axes = plt.subplots(4, 8, figsize=(22, 18), constrained_layout=False)
        for channel_index, axis in enumerate(axes.flat):
            axis.pcolormesh(
                operator.x_axis_cm,
                operator.y_axis_cm,
                raw_surface[:, :, channel_index],
                shading="nearest",
                cmap="viridis",
                vmin=raw_lower[channel_index],
                vmax=raw_upper[channel_index],
                rasterized=True,
            )
            _decorate_spatial_axis(axis, session)
            axis.set_title(
                f"S{channel_index + 1}  [{raw_lower[channel_index]:.4g}, {raw_upper[channel_index]:.4g}] kΩ",
                fontsize=8.5,
            )
            if channel_index % 8 != 0:
                axis.set_yticklabels([])
            if channel_index < 24:
                axis.set_xticklabels([])
        figure.suptitle(f"Raw Spatial Response — {session.slug}", fontsize=18, y=0.995)
        legend = _source_legend(session)
        if legend:
            figure.legend(handles=legend, loc="lower center", ncol=len(legend), frameon=False)
        figure.subplots_adjust(left=0.045, right=0.995, top=0.965, bottom=0.055, wspace=0.14, hspace=0.23)
        paths.append(_save_figure(figure, _figure_path(output_directory, f"spatial_raw_{session.slug}")))

        figure, axes = plt.subplots(4, 8, figsize=(22, 18), constrained_layout=False)
        mappable = None
        for channel_index, axis in enumerate(axes.flat):
            mappable = axis.pcolormesh(
                operator.x_axis_cm,
                operator.y_axis_cm,
                robust_surface[:, :, channel_index],
                shading="nearest",
                cmap="coolwarm",
                vmin=-ROBUST_DISPLAY_LIMIT,
                vmax=ROBUST_DISPLAY_LIMIT,
                rasterized=True,
            )
            _decorate_spatial_axis(axis, session)
            axis.set_title(f"S{channel_index + 1}", fontsize=9)
            if channel_index % 8 != 0:
                axis.set_yticklabels([])
            if channel_index < 24:
                axis.set_xticklabels([])
        assert mappable is not None
        colorbar_axis = figure.add_axes([0.945, 0.27, 0.012, 0.46])
        colorbar = figure.colorbar(mappable, cax=colorbar_axis)
        colorbar.set_label("Within-session robust z-score")
        figure.suptitle(f"Robust Spatial Response — {session.slug}", fontsize=18, y=0.995)
        legend = _source_legend(session)
        if legend:
            figure.legend(handles=legend, loc="lower center", ncol=len(legend), frameon=False)
        figure.subplots_adjust(left=0.045, right=0.925, top=0.965, bottom=0.055, wspace=0.14, hspace=0.23)
        paths.append(_save_figure(figure, _figure_path(output_directory, f"spatial_robust_{session.slug}")))
    return paths, pd.DataFrame.from_records(scale_records)


def render_occupancy_and_support(
    sessions: Sequence[SanitySession],
    cells: Sequence[CellBalancedData],
    operators: Sequence[KernelOperator],
    output_directory: Path,
) -> tuple[list[Path], pd.DataFrame]:
    paths: list[Path] = []
    records: list[dict[str, Any]] = []
    for session, cell, operator in zip(sessions, cells, operators):
        supported_ess = operator.effective_sample_size[operator.supported]
        supported_neighbors = operator.neighbor_count[operator.supported]
        if not len(supported_ess):
            raise ValueError(f"No supported kernel grid cells for {session.slug}")
        records.append(
            {
                "session": session.slug,
                "occupied_cells": len(cell.cell_ids),
                "on_paper_readings": int(cell.counts.sum()),
                "median_readings_per_cell": float(np.median(cell.counts)),
                "supported_grid_nodes": int(operator.supported.sum()),
                "grid_nodes": int(operator.supported.size),
                "support_fraction": float(operator.supported.mean()),
                "supported_neighbor_median": float(np.median(supported_neighbors)),
                "supported_ess_median": float(np.median(supported_ess)),
            }
        )
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
        scatter = axes[0].scatter(
            cell.cell_xy_cm[:, 0],
            cell.cell_xy_cm[:, 1],
            c=cell.counts,
            cmap="viridis",
            s=14,
            norm=matplotlib.colors.LogNorm(vmin=1, vmax=max(2, int(cell.counts.max()))),
        )
        axes[0].set_title("Readings per occupied cell")
        figure.colorbar(scatter, ax=axes[0], label="count")
        neighbor_map = operator.neighbor_count.astype(float)
        neighbor_map[~operator.supported] = np.nan
        image = axes[1].pcolormesh(
            operator.x_axis_cm,
            operator.y_axis_cm,
            neighbor_map,
            shading="nearest",
            cmap="viridis",
        )
        axes[1].set_title("Supported neighbor count")
        figure.colorbar(image, ax=axes[1], label="occupied cells")
        ess_map = operator.effective_sample_size.copy()
        ess_map[~operator.supported] = np.nan
        image = axes[2].pcolormesh(
            operator.x_axis_cm,
            operator.y_axis_cm,
            ess_map,
            shading="nearest",
            cmap="magma",
        )
        axes[2].set_title("Supported Gaussian ESS")
        figure.colorbar(image, ax=axes[2], label="ESS")
        for axis in axes:
            _decorate_spatial_axis(axis, session)
            axis.set_xlabel("paper x (cm)")
            axis.set_ylabel("paper y (cm)")
        figure.suptitle(f"Spatial Sampling Support — {session.slug}", fontsize=18)
        paths.append(_save_figure(figure, _figure_path(output_directory, f"kernel_support_{session.slug}")))
    return paths, pd.DataFrame.from_records(records)


def _render_pca_spatial(
    session: SanitySession,
    operator: KernelOperator,
    smoothed_scores: np.ndarray,
    limits: np.ndarray,
    title: str,
    path: Path,
) -> Path:
    rgb = scores_to_rgb(smoothed_scores, limits)
    rgba = np.ones((*operator.supported.shape, 4), dtype=np.float64)
    rgba[:, :, :3] = np.nan_to_num(rgb, nan=1.0)
    rgba[:, :, 3] = operator.supported.astype(np.float64)
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 9.5), constrained_layout=True)
    axes[0, 0].imshow(
        rgba,
        origin="upper",
        extent=[
            operator.x_axis_cm[0],
            operator.x_axis_cm[-1],
            operator.y_axis_cm[-1],
            operator.y_axis_cm[0],
        ],
        interpolation="nearest",
        aspect="equal",
    )
    axes[0, 0].set_title("PC1/PC2/PC3 → R/G/B")
    for component in range(3):
        axis = axes.flat[component + 1]
        image = axis.pcolormesh(
            operator.x_axis_cm,
            operator.y_axis_cm,
            smoothed_scores[:, :, component],
            shading="nearest",
            cmap="coolwarm",
            vmin=-limits[component],
            vmax=limits[component],
            rasterized=True,
        )
        axis.set_title(f"PC{component + 1} score")
        figure.colorbar(image, ax=axis, fraction=0.047, pad=0.03)
    for axis in axes.flat:
        _decorate_spatial_axis(axis, session)
        axis.set_xlabel("paper x (cm)")
        axis.set_ylabel("paper y (cm)")
    figure.suptitle(title, fontsize=18)
    legend = _source_legend(session)
    if legend:
        figure.legend(handles=legend, loc="lower center", ncol=len(legend), frameon=False)
    return _save_figure(figure, path)


def _loading_records(fit_name: str, fit: PCAFit) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for channel_index in range(fit.components.shape[1]):
        record: dict[str, Any] = {"pca_fit": fit_name, "channel": f"S{channel_index + 1}"}
        for component in range(fit.components.shape[0]):
            record[f"PC{component + 1}_loading"] = float(fit.components[component, channel_index])
        records.append(record)
    return records


def _variance_records(fit_name: str, fit: PCAFit) -> list[dict[str, Any]]:
    return [
        {
            "pca_fit": fit_name,
            "component": f"PC{index + 1}",
            "explained_variance": float(fit.explained_variance[index]),
            "explained_variance_ratio": float(fit.explained_variance_ratio[index]),
        }
        for index in range(len(fit.explained_variance_ratio))
    ]


def _render_pca_diagnostics(
    fit_name: str,
    fit: PCAFit,
    output_directory: Path,
    scores: np.ndarray | None = None,
    score_sessions: Sequence[str] | None = None,
) -> Path:
    column_count = 3 if scores is not None else 2
    figure, axes = plt.subplots(1, column_count, figsize=(6.0 * column_count, 5.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    components = np.arange(1, len(fit.explained_variance_ratio) + 1)
    axes[0].bar(components, fit.explained_variance_ratio, color="#4c78a8")
    axes[0].set_xticks(components)
    axes[0].set_xlabel("component")
    axes[0].set_ylabel("explained variance ratio")
    axes[0].set_ylim(0, max(0.05, float(fit.explained_variance_ratio.max()) * 1.12))
    loading_limit = float(np.max(np.abs(fit.components)))
    image = axes[1].imshow(
        fit.components.T,
        aspect="auto",
        cmap="coolwarm",
        vmin=-loading_limit,
        vmax=loading_limit,
        origin="upper",
    )
    axes[1].set_xticks(range(3), ["PC1", "PC2", "PC3"])
    axes[1].set_yticks(range(32), [f"S{index}" for index in range(1, 33)], fontsize=7)
    axes[1].set_title("Channel loadings")
    figure.colorbar(image, ax=axes[1], label="signed loading")
    if scores is not None:
        if score_sessions is None or len(score_sessions) != len(scores):
            raise ValueError("Global PCA score labels do not align")
        labels = np.asarray(score_sessions, dtype=object)
        unique = list(dict.fromkeys(labels.tolist()))
        for session_index, session_name in enumerate(unique):
            selection = labels == session_name
            axes[2].scatter(
                scores[selection, 0],
                scores[selection, 1],
                s=8,
                alpha=0.28,
                color=SESSION_COLORS[session_index],
                label=session_name,
                edgecolors="none",
            )
            centroid = np.mean(scores[selection, :2], axis=0)
            axes[2].scatter(
                centroid[0], centroid[1], marker="X", s=95, color=SESSION_COLORS[session_index], edgecolor="white", linewidth=0.8
            )
        axes[2].axhline(0, color="0.7", linewidth=0.6)
        axes[2].axvline(0, color="0.7", linewidth=0.6)
        axes[2].set_xlabel("PC1 score")
        axes[2].set_ylabel("PC2 score")
        axes[2].set_title("Cell-balanced global scores")
        axes[2].legend(loc="best", fontsize=7, frameon=False)
    figure.suptitle(f"PCA Diagnostics — {fit_name}", fontsize=18)
    safe_name = fit_name.lower().replace(" ", "_").replace("/", "_")
    return _save_figure(figure, _figure_path(output_directory, f"pca_diagnostics_{safe_name}"))


def run_pca_analysis(
    sessions: Sequence[SanitySession],
    cells: Sequence[CellBalancedData],
    operators: Sequence[KernelOperator],
    output_directory: Path,
) -> tuple[list[Path], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths: list[Path] = []
    loading_records: list[dict[str, Any]] = []
    variance_records: list[dict[str, Any]] = []
    separation_records: list[dict[str, Any]] = []

    global_fit, global_scores = fit_equal_session_pca([cell.values for cell in cells], 3)
    global_limits = symmetric_score_limits(global_scores, 0.99, equal_session=True)
    all_global_scores = np.concatenate(global_scores, axis=0)
    all_labels = np.concatenate(
        [np.full(len(scores), session.slug, dtype=object) for session, scores in zip(sessions, global_scores)]
    )
    loading_records.extend(_loading_records("global", global_fit))
    variance_records.extend(_variance_records("global", global_fit))
    centroids: dict[str, np.ndarray] = {}
    radii: dict[str, float] = {}
    for session, scores, operator in zip(sessions, global_scores, operators):
        smoothed = apply_kernel(operator, scores)
        paths.append(
            _render_pca_spatial(
                session,
                operator,
                smoothed,
                global_limits,
                f"Global PCA Spatial Scores — {session.slug}",
                _figure_path(output_directory, f"pca_global_spatial_{session.slug}"),
            )
        )
        centroid = np.mean(scores[:, :3], axis=0)
        centroids[session.slug] = centroid
        radii[session.slug] = float(np.sqrt(np.mean(np.sum((scores[:, :3] - centroid) ** 2, axis=1))))
        separation_records.append(
            {
                "record_type": "within_session_radius",
                "session_a": session.slug,
                "session_b": "",
                "value": radii[session.slug],
            }
        )
    for first_index, first in enumerate(sessions):
        for second in sessions[first_index + 1 :]:
            separation_records.append(
                {
                    "record_type": "centroid_distance_pc1_pc3",
                    "session_a": first.slug,
                    "session_b": second.slug,
                    "value": float(np.linalg.norm(centroids[first.slug] - centroids[second.slug])),
                }
            )
    paths.append(
        _render_pca_diagnostics(
            "Global equal-session",
            global_fit,
            output_directory,
            all_global_scores,
            all_labels,
        )
    )

    for session, cell, operator in zip(sessions, cells, operators):
        fit = fit_weighted_pca(cell.values, component_count=3)
        scores = fit.transform(cell.values)
        limits = symmetric_score_limits([scores], 0.99, equal_session=False)
        smoothed = apply_kernel(operator, scores)
        fit_name = session.slug
        loading_records.extend(_loading_records(fit_name, fit))
        variance_records.extend(_variance_records(fit_name, fit))
        paths.append(
            _render_pca_spatial(
                session,
                operator,
                smoothed,
                limits,
                f"Session-Fitted PCA Spatial Scores — {session.slug}",
                _figure_path(output_directory, f"pca_session_spatial_{session.slug}"),
            )
        )
        paths.append(_render_pca_diagnostics(fit_name, fit, output_directory))
    return (
        paths,
        pd.DataFrame.from_records(loading_records),
        pd.DataFrame.from_records(variance_records),
        pd.DataFrame.from_records(separation_records),
    )


def build_visit_dispersion_table(
    sessions: Sequence[SanitySession],
    cells: Sequence[CellBalancedData],
    minimum_visits: int = 3,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for session, cell in zip(sessions, cells):
        data = session.data
        valid = data.pose_mask & paper_contains(data.paper_xy_cm)
        cell_ids, _, _ = spatial_cell_ids(data.paper_xy_cm)
        visits = segment_cell_visits(cell_ids, session.sample_time_s, valid)
        by_cell: dict[int, list[Visit]] = {}
        for visit in visits:
            by_cell.setdefault(visit.cell_id, []).append(visit)
        _, overall_scale = scaled_mad(cell.values)
        for cell_id, current_visits in by_cell.items():
            if len(current_visits) < minimum_visits:
                continue
            visit_values = np.asarray(
                [np.median(data.sensors[visit.indices], axis=0) for visit in current_visits],
                dtype=np.float64,
            )
            dispersion = 1.4826 * np.median(
                np.abs(visit_values - np.median(visit_values, axis=0)), axis=0
            )
            normalized = dispersion / overall_scale
            member_indices = np.concatenate([visit.indices for visit in current_visits])
            location = np.median(data.paper_xy_cm[member_indices], axis=0)
            temporal_span = float(
                session.sample_time_s[member_indices].max() - session.sample_time_s[member_indices].min()
            )
            for channel_index in range(32):
                records.append(
                    {
                        "session": session.slug,
                        "cell_id": cell_id,
                        "paper_x_cm": float(location[0]),
                        "paper_y_cm": float(location[1]),
                        "channel": f"S{channel_index + 1}",
                        "visit_count": len(current_visits),
                        "temporal_span_s": temporal_span,
                        "visit_scaled_mad_kohm": float(dispersion[channel_index]),
                        "normalized_dispersion": float(normalized[channel_index]),
                    }
                )
    if not records:
        raise ValueError("No paper cells have at least three independent visits")
    return pd.DataFrame.from_records(records)


def build_split_shift_tables(
    sessions: Sequence[SanitySession],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    channel_records: list[dict[str, Any]] = []
    support_records: list[dict[str, Any]] = []
    for session in sessions:
        data = session.data
        train_mask = np.zeros(len(data.sensors), dtype=bool)
        validation_mask = np.zeros(len(data.sensors), dtype=bool)
        for block in data.blocks:
            if block.split == "train":
                train_mask[block.start : block.stop] = True
            elif block.split == "validation":
                validation_mask[block.start : block.stop] = True
        if np.any(train_mask & validation_mask) or not np.any(train_mask) or not np.any(validation_mask):
            raise ValueError(f"Invalid train/validation physical-row masks for {session.slug}")
        train = data.sensors[train_mask].astype(np.float64)
        validation = data.sensors[validation_mask].astype(np.float64)
        train_median, train_scale = scaled_mad(train)
        validation_median = np.median(validation, axis=0)
        for channel_index in range(32):
            channel_records.append(
                {
                    "session": session.slug,
                    "channel": f"S{channel_index + 1}",
                    "train_rows": int(train_mask.sum()),
                    "validation_rows": int(validation_mask.sum()),
                    "robust_standardized_median_shift": float(
                        (validation_median[channel_index] - train_median[channel_index])
                        / train_scale[channel_index]
                    ),
                    "normalized_wasserstein": normalized_wasserstein_1d(
                        train[:, channel_index], validation[:, channel_index], train_scale[channel_index]
                    ),
                }
            )

        train_spatial = train_mask & data.pose_mask & paper_contains(data.paper_xy_cm)
        validation_spatial = validation_mask & data.pose_mask & paper_contains(data.paper_xy_cm)
        train_cells = balance_spatial_cells(data.paper_xy_cm[train_spatial], data.sensors[train_spatial])
        validation_cells = balance_spatial_cells(
            data.paper_xy_cm[validation_spatial], data.sensors[validation_spatial]
        )
        supported, neighbors, ess = kernel_support_at(
            validation_cells.cell_xy_cm, train_cells.cell_xy_cm
        )
        support_records.append(
            {
                "session": session.slug,
                "train_spatial_cells": len(train_cells.cell_ids),
                "validation_spatial_cells": len(validation_cells.cell_ids),
                "validation_cells_supported_by_train": int(supported.sum()),
                "validation_support_fraction": float(supported.mean()),
                "validation_neighbor_median": float(np.median(neighbors)),
                "validation_ess_median": float(np.median(ess)),
            }
        )
    return pd.DataFrame.from_records(channel_records), pd.DataFrame.from_records(support_records)


def render_split_shift(
    sessions: Sequence[SanitySession],
    shift_table: pd.DataFrame,
    support_table: pd.DataFrame,
    output_directory: Path,
) -> list[Path]:
    order = [session.slug for session in sessions]
    columns = [f"S{i}" for i in range(1, 33)]
    median_shift = shift_table.pivot(index="session", columns="channel", values="robust_standardized_median_shift").reindex(index=order, columns=columns)
    wasserstein = shift_table.pivot(index="session", columns="channel", values="normalized_wasserstein").reindex(index=order, columns=columns)
    figure, axes = plt.subplots(2, 1, figsize=(16, 8.5), constrained_layout=True)
    shift_limit = float(np.nanquantile(np.abs(median_shift.to_numpy()), 0.99))
    image = axes[0].imshow(
        median_shift.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-shift_limit, vmax=shift_limit
    )
    axes[0].set_title("Validation − Train Robust Median Shift")
    figure.colorbar(image, ax=axes[0], label="train-MAD units")
    wasserstein_limit = float(np.nanquantile(wasserstein.to_numpy(), 0.99))
    image = axes[1].imshow(
        wasserstein.to_numpy(), aspect="auto", cmap="magma", vmin=0, vmax=max(wasserstein_limit, 1e-8)
    )
    axes[1].set_title("Validation vs Train Normalized Wasserstein Distance")
    figure.colorbar(image, ax=axes[1], label="train-MAD units")
    for axis in axes:
        axis.set_yticks(range(len(order)), order)
        axis.set_xticks(range(32), columns, rotation=90)
    figure.suptitle("Physical-Row Train/Validation Distribution Shift", fontsize=18)
    shift_path = _save_figure(figure, _figure_path(output_directory, "train_validation_channel_shift"))

    selected = support_table.set_index("session").loc[order]
    figure, axis = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    bars = axis.barh(order, selected.validation_support_fraction, color=SESSION_COLORS)
    axis.set_xlim(0, 1)
    axis.set_xlabel("fraction of validation cells supported by train cells")
    axis.set_title("Train Spatial Support for Validation Cells", fontsize=18)
    axis.bar_label(bars, fmt="%.3f", padding=3)
    support_path = _save_figure(figure, _figure_path(output_directory, "train_validation_spatial_support"))
    return [shift_path, support_path]


def build_hysteresis_tables(
    sessions: Sequence[SanitySession],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    encounter_records: list[dict[str, Any]] = []
    trajectory_records: list[dict[str, Any]] = []
    difference_records: list[dict[str, Any]] = []
    for session in sessions:
        data = session.data
        eligible = data.pose_mask & paper_contains(data.paper_xy_cm)
        session_median = np.median(data.sensors.astype(np.float64), axis=0)
        memberships = {source: _source_membership(session, source) for source in SOURCE_NAMES}
        for source in session.info.source_names:
            inside = memberships[source]
            other = np.zeros(len(inside), dtype=bool)
            for other_source in SOURCE_NAMES:
                if other_source != source:
                    other |= memberships[other_source]
            grouped = group_source_encounters(inside, session.sample_time_s)
            accepted, statuses = validate_encounter_windows(grouped, eligible, inside, other)
            accepted_lookup = {
                (window.first_inside, window.final_inside): index
                for index, window in enumerate(accepted)
            }
            for status in statuses:
                key = (status["first_inside"], status["final_inside"])
                encounter_records.append(
                    {
                        "session": session.slug,
                        "source": source,
                        **status,
                        "accepted_index": accepted_lookup.get(key, -1),
                        "first_raw_row": int(data.raw_row_indices[status["first_inside"]]),
                        "final_raw_row": int(data.raw_row_indices[status["final_inside"]]),
                    }
                )
            source_index = SOURCE_NAMES.index(source)
            for encounter_index, window in enumerate(accepted):
                centered = data.sensors.astype(np.float64) - session_median
                for leg, indices, offsets in (
                    ("entry", window.entry_indices, np.arange(-(ENCOUNTER_ROWS - 1), 1)),
                    ("exit", window.exit_indices, np.arange(0, ENCOUNTER_ROWS)),
                ):
                    distances = data.distance_cm[indices, source_index]
                    for local_index, (row_index, offset) in enumerate(zip(indices, offsets)):
                        for channel_index in range(32):
                            trajectory_records.append(
                                {
                                    "session": session.slug,
                                    "source": source,
                                    "encounter": encounter_index,
                                    "leg": leg,
                                    "sample_offset": int(offset),
                                    "distance_cm": float(distances[local_index]),
                                    "channel": f"S{channel_index + 1}",
                                    "centered_kohm": float(centered[row_index, channel_index]),
                                }
                            )
                differences = paired_distance_bin_differences(
                    data.distance_cm[window.entry_indices, source_index],
                    data.distance_cm[window.exit_indices, source_index],
                    centered[window.entry_indices],
                    centered[window.exit_indices],
                )
                for bin_index, values in differences.items():
                    for channel_index, value in enumerate(values):
                        difference_records.append(
                            {
                                "session": session.slug,
                                "source": source,
                                "encounter": encounter_index,
                                "distance_bin": bin_index,
                                "distance_bin_start_cm": bin_index * DISTANCE_BIN_CM,
                                "distance_bin_stop_cm": (bin_index + 1) * DISTANCE_BIN_CM,
                                "channel": f"S{channel_index + 1}",
                                "exit_minus_entry_kohm": float(value),
                            }
                        )
    encounter_table = pd.DataFrame.from_records(encounter_records)
    trajectory_table = pd.DataFrame.from_records(trajectory_records)
    difference_table = pd.DataFrame.from_records(difference_records)
    if trajectory_table.empty or difference_table.empty:
        raise ValueError("No valid hysteresis trajectories or paired distance bins")
    return encounter_table, trajectory_table, difference_table


def _weighted_group_summary(
    frame: pd.DataFrame,
    value_column: str,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for group_values, selected in frame.groupby(list(group_columns), sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        weights = equal_group_weights(selected["session"].astype(str).to_numpy())
        quantiles = weighted_quantile(selected[value_column].to_numpy(), [0.25, 0.5, 0.75], weights)
        record = {name: value for name, value in zip(group_columns, group_values)}
        record.update(
            {
                "q25": float(quantiles[0]),
                "median": float(quantiles[1]),
                "q75": float(quantiles[2]),
                "session_count": int(selected["session"].nunique()),
                "encounter_count": int(selected[["session", "encounter"]].drop_duplicates().shape[0]),
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def render_hysteresis(
    trajectory_table: pd.DataFrame,
    difference_table: pd.DataFrame,
    output_directory: Path,
) -> tuple[list[Path], pd.DataFrame, pd.DataFrame]:
    paths: list[Path] = []
    trajectory_summary = _weighted_group_summary(
        trajectory_table,
        "centered_kohm",
        ("source", "leg", "sample_offset", "channel"),
    )
    difference_summary = _weighted_group_summary(
        difference_table,
        "exit_minus_entry_kohm",
        ("source", "distance_bin", "channel"),
    )
    supported = (difference_summary.session_count >= 2) & (difference_summary.encounter_count >= 3)
    difference_summary["supported"] = supported

    for source in SOURCE_NAMES:
        selected = trajectory_summary.loc[trajectory_summary["source"] == source]
        if selected.empty:
            continue
        figure, axes = plt.subplots(4, 8, figsize=(22, 13.5), constrained_layout=False)
        for channel_index, axis in enumerate(axes.flat):
            channel = f"S{channel_index + 1}"
            for leg, color in (("entry", "#2b6cb0"), ("exit", "#dd6b20")):
                leg_data = selected.loc[(selected.channel == channel) & (selected.leg == leg)].sort_values(
                    "sample_offset"
                )
                x = leg_data.sample_offset.to_numpy(dtype=float)
                axis.plot(x, leg_data["median"], color=color, linewidth=1.4, label=leg)
                axis.fill_between(x, leg_data.q25, leg_data.q75, color=color, alpha=0.18, linewidth=0)
            axis.axvline(0, color="0.25", linestyle="--", linewidth=0.7)
            axis.axhline(0, color="0.75", linewidth=0.6)
            axis.set_title(channel, fontsize=10)
            axis.grid(alpha=0.14, linewidth=0.5)
            if channel_index % 8 == 0:
                axis.set_ylabel("ΔkΩ")
            if channel_index >= 24:
                axis.set_xlabel("relative sample (≈0.557 s/sample)")
        handles = [
            Line2D([0], [0], color="#2b6cb0", linewidth=2, label="entry"),
            Line2D([0], [0], color="#dd6b20", linewidth=2, label="exit"),
        ]
        figure.suptitle(f"{source.title()} Strip Entry and Exit Trajectories", fontsize=18, y=0.995)
        figure.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
        figure.subplots_adjust(left=0.055, right=0.995, top=0.95, bottom=0.085, wspace=0.30, hspace=0.35)
        paths.append(_save_figure(figure, _figure_path(output_directory, f"hysteresis_trajectory_{source}")))

    supported_values = difference_summary.loc[difference_summary.supported, "median"].to_numpy()
    if not len(supported_values):
        raise ValueError("No paired hysteresis distance bins have sufficient support")
    shared_limit = float(np.quantile(np.abs(supported_values), 0.99))
    shared_limit = max(shared_limit, 1e-8)
    for source in SOURCE_NAMES:
        selected = difference_summary.loc[
            (difference_summary.source == source) & difference_summary.supported
        ]
        if selected.empty:
            continue
        maximum_bin = int(selected.distance_bin.max())
        matrix = np.full((32, maximum_bin + 1), np.nan, dtype=np.float64)
        support_matrix = np.zeros_like(matrix)
        for row in selected.itertuples(index=False):
            channel_index = int(str(row.channel)[1:]) - 1
            matrix[channel_index, int(row.distance_bin)] = float(row.median)
            support_matrix[channel_index, int(row.distance_bin)] = float(row.encounter_count)
        figure, axes = plt.subplots(2, 1, figsize=(13, 10), constrained_layout=True, height_ratios=[4, 1.5])
        image = axes[0].imshow(
            matrix,
            aspect="auto",
            origin="upper",
            cmap="coolwarm",
            vmin=-shared_limit,
            vmax=shared_limit,
        )
        axes[0].set_yticks(range(32), [f"S{i}" for i in range(1, 33)], fontsize=8)
        axes[0].set_xticks(range(maximum_bin + 1), [f"{i}–{i + 1}" for i in range(maximum_bin + 1)], rotation=45, ha="right")
        axes[0].set_xlabel("distance to strip bin (cm)")
        axes[0].set_title("Equal-session weighted median exit − entry")
        figure.colorbar(image, ax=axes[0], label="kΩ")
        support_by_bin = np.max(support_matrix, axis=0)
        axes[1].bar(np.arange(maximum_bin + 1), np.nan_to_num(support_by_bin), color="#6b7280")
        axes[1].set_xticks(range(maximum_bin + 1), [f"{i}–{i + 1}" for i in range(maximum_bin + 1)], rotation=45, ha="right")
        axes[1].set_ylabel("paired encounters")
        axes[1].set_xlabel("distance to strip bin (cm)")
        axes[1].set_title("Maximum contributing encounter count across channels")
        figure.suptitle(f"{source.title()} Approach/Departure Hysteresis", fontsize=18)
        paths.append(_save_figure(figure, _figure_path(output_directory, f"hysteresis_distance_{source}")))
    return paths, trajectory_summary, difference_summary


def _summarize_results(
    integrity: pd.DataFrame,
    kernel: pd.DataFrame,
    visits: pd.DataFrame,
    shift: pd.DataFrame,
    split_support: pd.DataFrame,
    encounters: pd.DataFrame,
    pca_variance: pd.DataFrame,
    figure_count: int,
) -> dict[str, Any]:
    accepted = encounters.loc[encounters.status == "accepted"]
    encounter_counts = (
        accepted.groupby(["source", "session"]).size().rename("accepted").reset_index().to_dict("records")
    )
    global_variance = pca_variance.loc[pca_variance.pca_fit == "global"]
    return {
        "session_count": int(len(integrity)),
        "channel_count": 32,
        "figure_count": int(figure_count),
        "all_timestamps_strictly_increasing": bool(integrity.timestamp_strictly_increasing.all()),
        "total_pose_missing_rows": int(integrity.pose_missing_rows.sum()),
        "kernel_support_fraction_min": float(kernel.support_fraction.min()),
        "kernel_support_fraction_max": float(kernel.support_fraction.max()),
        "median_repeat_visit_dispersion": float(visits.normalized_dispersion.median()),
        "maximum_absolute_train_validation_median_shift": float(
            np.max(np.abs(shift.robust_standardized_median_shift))
        ),
        "maximum_normalized_train_validation_wasserstein": float(shift.normalized_wasserstein.max()),
        "minimum_train_support_for_validation": float(split_support.validation_support_fraction.min()),
        "global_pc1_pc3_explained_variance": float(global_variance.explained_variance_ratio.sum()),
        "accepted_encounters": encounter_counts,
        "automatic_data_exclusions": 0,
        "interpretation": "Descriptive sanity checks only; no session, annotation, split, or model was changed.",
    }


def run_sensor_sanity_checks(
    output_directory: Path,
    experiment_root: Path = EXPERIMENT_ROOT,
    verify_hashes: bool = True,
) -> SanityArtifacts:
    """Run all Notebook 10 calculations and write deterministic tables/figures."""
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    sessions = load_sanity_sessions(experiment_root, verify_hashes=verify_hashes)
    cells = _cell_balanced_sessions(sessions)
    operators = [build_kernel_operator(cell.cell_xy_cm) for cell in cells]

    figures: list[Path] = []
    integrity = build_integrity_table(sessions)
    distributions = build_channel_distribution_table(sessions)
    figures.append(render_channel_distribution(sessions, distributions, output_directory))
    figures.append(render_concatenated_time_traces(sessions, output_directory))
    spatial_paths, spatial_scales = render_channel_spatial_maps(
        sessions, cells, operators, output_directory
    )
    figures.extend(spatial_paths)
    occupancy_paths, kernel_coverage = render_occupancy_and_support(
        sessions, cells, operators, output_directory
    )
    figures.extend(occupancy_paths)
    pca_paths, pca_loadings, pca_variance, pca_separation = run_pca_analysis(
        sessions, cells, operators, output_directory
    )
    figures.extend(pca_paths)
    visit_dispersion = build_visit_dispersion_table(sessions, cells)
    split_shift, split_support = build_split_shift_tables(sessions)
    figures.extend(render_split_shift(sessions, split_shift, split_support, output_directory))
    encounter_table, trajectory_table, difference_table = build_hysteresis_tables(sessions)
    hysteresis_paths, trajectory_summary, difference_summary = render_hysteresis(
        trajectory_table, difference_table, output_directory
    )
    figures.extend(hysteresis_paths)

    tables = {
        "data_integrity": integrity,
        "channel_distribution_summary": distributions,
        "spatial_map_scales": spatial_scales,
        "kernel_coverage": kernel_coverage,
        "repeat_visit_dispersion": visit_dispersion,
        "pca_explained_variance": pca_variance,
        "pca_loadings": pca_loadings,
        "pca_session_separation": pca_separation,
        "split_shift": split_shift,
        "split_spatial_support": split_support,
        "hysteresis_encounters": encounter_table,
        "hysteresis_trajectory_summary": trajectory_summary,
        "hysteresis_distance_observations": difference_table,
        "hysteresis_distance_summary": difference_summary,
    }
    for name, frame in tables.items():
        _save_table(frame, output_directory / f"{name}.csv")
    summary = _summarize_results(
        integrity,
        kernel_coverage,
        visit_dispersion,
        split_shift,
        split_support,
        encounter_table,
        pca_variance,
        len(figures),
    )
    _write_json(summary, output_directory / "summary.json")
    manifest = pd.DataFrame(
        {
            "artifact_type": ["figure"] * len(figures) + ["table"] * len(tables),
            "name": [path.stem for path in figures] + list(tables),
            "relative_path": [
                str(path.relative_to(output_directory)) for path in figures
            ]
            + [f"{name}.csv" for name in tables],
        }
    )
    _save_table(manifest, output_directory / "artifact_manifest.csv")
    return SanityArtifacts(
        output_directory=output_directory,
        figures=tuple(figures),
        tables=tables,
        summary=summary,
    )
