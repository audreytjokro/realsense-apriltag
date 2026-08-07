"""Lag-shifted pose lookup and the height/desk-bounds quality filters.

The Cyranose does not react instantly: a reading at time T reflects roughly
where the snout was `lag_s` seconds earlier, once the odor had time to reach
the sensor. `qualifying_rows` re-associates each measurement reading with the
snout's interpolated position at that earlier time, then applies the physical
plausibility filters (was the snout actually close enough to the desk, and
inside the scan area) before a reading is trusted for spatial analysis.

The four CSV column names below are fixed by record_cyranose_reading_pose.py's
output format, not by TrialConfig -- any session recorded with that recorder
has this same column shape regardless of odor or desk layout.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Sequence

from .config import TrialConfig
from .response import FLAG_COLUMN, compute_baseline, sensor_response

TIME_COLUMN = "pose_elapsed_s"
X_COLUMN = "snout_desk_x_cm"
Y_COLUMN = "snout_desk_y_cm"
Z_COLUMN = "snout_desk_z_cm"


def _parse_float(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


@dataclass(frozen=True)
class PoseSample:
    time_s: float
    x: float
    y: float
    z: float


def build_pose_series(
    raw_rows: Sequence[dict[str, str]],
) -> tuple[list[PoseSample], list[float]]:
    """Every row with a complete, finite pose (any flag), in recorded order."""
    samples: list[PoseSample] = []
    for row in raw_rows:
        time_s = _parse_float(row.get(TIME_COLUMN))
        x = _parse_float(row.get(X_COLUMN))
        y = _parse_float(row.get(Y_COLUMN))
        z = _parse_float(row.get(Z_COLUMN))
        if None not in (time_s, x, y, z):
            samples.append(PoseSample(time_s, x, y, z))  # type: ignore[arg-type]
    return samples, [sample.time_s for sample in samples]


def interpolate_pose(
    pose_samples: Sequence[PoseSample],
    pose_times: Sequence[float],
    target_time: float,
    max_gap_s: float,
) -> PoseSample | None:
    """Linear interpolation between the two pose samples bracketing target_time.

    Returns None at the series' edges or across a gap wider than max_gap_s,
    rather than extrapolating past the data or bridging a tracking dropout.
    """
    index = bisect.bisect_left(pose_times, target_time)
    if index == 0 or index == len(pose_samples):
        return None
    before, after = pose_samples[index - 1], pose_samples[index]
    gap = after.time_s - before.time_s
    if gap <= 0 or gap > max_gap_s:
        return None
    fraction = (target_time - before.time_s) / gap
    return PoseSample(
        time_s=target_time,
        x=before.x + fraction * (after.x - before.x),
        y=before.y + fraction * (after.y - before.y),
        z=before.z + fraction * (after.z - before.z),
    )


@dataclass(frozen=True)
class QualifyingRow:
    x: float
    y: float
    z: float
    height_cm: float
    height_in_band: bool
    response: float
    fractional: list[float]
    reading_time_s: float
    pose_time_s: float
    scan_elapsed_s: float | None
    scan_fraction: float | None


def qualifying_rows(
    raw_rows: Sequence[dict[str, str]],
    config: TrialConfig,
    *,
    apply_height_filter: bool = True,
) -> list[QualifyingRow]:
    """Lag-shift each measurement reading, then apply the height and desk-bounds QC.

    Setting apply_height_filter=False keeps out-of-band readings instead of
    dropping them (each still carries height_in_band), which is what the
    height-QC diagnostic plots need.
    """
    baseline = compute_baseline(raw_rows, config.sensor_fields, config.baseline_flag)
    pose_samples, pose_times = build_pose_series(raw_rows)
    xmin, xmax, ymin, ymax = config.desk_bounds_cm
    pose_window = config.pose_window_s

    rows: list[QualifyingRow] = []
    for raw_row in raw_rows:
        if raw_row.get(FLAG_COLUMN) != config.measurement_flag:
            continue
        reading_time = _parse_float(raw_row.get(TIME_COLUMN))
        if reading_time is None:
            continue
        pose_time = reading_time - config.lag_s
        if pose_window is not None and not (pose_window[0] <= pose_time <= pose_window[1]):
            continue
        pose = interpolate_pose(pose_samples, pose_times, pose_time, config.max_interpolation_gap_s)
        if pose is None:
            continue
        height = abs(pose.z)
        height_in_band = config.height_band_cm[0] <= height <= config.height_band_cm[1]
        if apply_height_filter and not height_in_band:
            continue
        if not (xmin <= pose.x <= xmax and ymin <= pose.y <= ymax):
            continue
        fractional, response = sensor_response(raw_row, baseline, config.sensor_fields)
        rows.append(
            QualifyingRow(
                x=pose.x,
                y=pose.y,
                z=pose.z,
                height_cm=height,
                height_in_band=height_in_band,
                response=response,
                fractional=fractional,
                reading_time_s=reading_time,
                pose_time_s=pose_time,
                scan_elapsed_s=(pose_time - pose_window[0]) if pose_window else None,
                scan_fraction=(
                    (pose_time - pose_window[0]) / (pose_window[1] - pose_window[0])
                    if pose_window
                    else None
                ),
            )
        )
    return rows
