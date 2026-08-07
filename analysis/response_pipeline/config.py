"""Per-trial configuration for the response pipeline.

`TrialConfig` holds the parameters that differ between physical setups and
between recording sessions. Two fields are never derived automatically and
must be supplied by whoever is running a new trial:

- `lag_s` comes from a dedicated response-lag calibration trial (see
  analysis/notebooks/02_response_lag_validation.ipynb for the method used on
  this rig).
- `desk_bounds_cm` comes from the desk-tag calibration for that physical rig
  (see calibration/).

Everything else has a default matching this repository's existing rig and
recorder (record_cyranose_reading_pose.py).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrialConfig:
    trial_id: str
    lag_s: float
    height_band_cm: tuple[float, float]
    desk_bounds_cm: tuple[float, float, float, float]  # (xmin, xmax, ymin, ymax)
    pose_window_s: tuple[float, float] | None = None
    baseline_flag: str = "1"
    measurement_flag: str = "2"
    max_interpolation_gap_s: float = 1.5
    sensor_prefix: str = "pcnose_S"
    sensor_suffix: str = "_kohm"
    sensor_count: int = 32

    def __post_init__(self) -> None:
        if self.height_band_cm[0] > self.height_band_cm[1]:
            raise ValueError("height_band_cm must be (min_cm, max_cm)")
        xmin, xmax, ymin, ymax = self.desk_bounds_cm
        if xmin > xmax or ymin > ymax:
            raise ValueError("desk_bounds_cm must be (xmin, xmax, ymin, ymax)")
        if self.pose_window_s is not None and self.pose_window_s[0] > self.pose_window_s[1]:
            raise ValueError("pose_window_s must be (start_s, end_s)")
        if self.sensor_count <= 0:
            raise ValueError("sensor_count must be positive")

    @property
    def sensor_fields(self) -> tuple[str, ...]:
        return tuple(
            f"{self.sensor_prefix}{index}{self.sensor_suffix}"
            for index in range(1, self.sensor_count + 1)
        )
