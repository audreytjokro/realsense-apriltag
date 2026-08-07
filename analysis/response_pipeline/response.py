"""Odor-response intensity from raw 32-channel Cyranose readings.

The odor "response" used throughout this project's spatial analyses is a
single scalar per reading: how far, on average across every sensor channel,
the current reading has moved from a clean-air reference, combined by
root-mean-square so that channels moving in opposite directions don't cancel.

Two call shapes are needed: notebooks 03/04 work with lists of CSV-row dicts
(compute_baseline/sensor_response), while the classifier-application
notebooks 05/06/08/09 already hold whole sessions as numpy arrays
(compute_baseline_array/normalized_channels). Both pairs are the same
formula: the dict-based functions are convenience wrappers around the array
versions, not a second implementation of the math.
"""

from __future__ import annotations

import math
import statistics
from typing import Sequence

import numpy as np

FLAG_COLUMN = "pcnose_flag"


def compute_baseline_array(
    resistances: np.ndarray,
    flags: np.ndarray,
    baseline_flag: int | float | str,
) -> np.ndarray:
    """Per-channel clean-air reference: median over the back half of the baseline-flag rows.

    `resistances` is one row per reading, one column per sensor channel;
    `flags` is the matching 1-D flag array. The back half (rather than the
    whole phase) is used so the reference reflects the sensor once it has
    settled into that phase, not its transient entry from whatever phase
    came before.
    """
    baseline_rows = resistances[flags == baseline_flag]
    if len(baseline_rows) == 0:
        raise ValueError(f"No rows found with flag == {baseline_flag!r}")
    late_rows = baseline_rows[len(baseline_rows) // 2 :]
    return np.median(late_rows, axis=0)


def normalized_channels(resistances: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Percent-change-from-baseline for every row and channel at once."""
    return 100.0 * (resistances - baseline) / baseline


def compute_baseline(
    raw_rows: Sequence[dict[str, str]],
    sensor_fields: Sequence[str],
    baseline_flag: str,
) -> dict[str, float]:
    """Dict-of-CSV-rows convenience wrapper around compute_baseline_array."""
    resistances = np.array([[float(row[field]) for field in sensor_fields] for row in raw_rows])
    flags = np.array([row.get(FLAG_COLUMN) for row in raw_rows])
    baseline_array = compute_baseline_array(resistances, flags, baseline_flag=baseline_flag)
    return dict(zip(sensor_fields, (float(value) for value in baseline_array)))


def sensor_response(
    raw_row: dict[str, str],
    baseline: dict[str, float],
    sensor_fields: Sequence[str],
) -> tuple[list[float], float]:
    """Single-row convenience wrapper around normalized_channels, plus its RMS."""
    resistances = np.array([[float(raw_row[field]) for field in sensor_fields]])
    baseline_array = np.array([baseline[field] for field in sensor_fields])
    fractional = normalized_channels(resistances, baseline_array)[0]
    response = math.sqrt(float(np.mean(fractional**2)))
    return list(fractional), response


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile: matches the notebooks' original definition exactly."""
    if not values:
        raise ValueError("percentile of an empty sequence is undefined")
    ordered = sorted(values)
    return ordered[round(fraction * (len(ordered) - 1))]


def median_fingerprint(fractional_rows: Sequence[Sequence[float]], sensor_count: int) -> list[float]:
    """Per-channel median signed percent change across a set of readings."""
    return [
        statistics.median(row[index] for row in fractional_rows) for index in range(sensor_count)
    ]


def summarize_responses(responses: Sequence[float]) -> dict[str, float]:
    """The count/median/p90 summary reported alongside every trial's retained readings."""
    if not responses:
        raise ValueError("Cannot summarize zero retained readings")
    return {
        "count": len(responses),
        "response_median": statistics.median(responses),
        "response_p90": percentile(responses, 0.90),
    }
