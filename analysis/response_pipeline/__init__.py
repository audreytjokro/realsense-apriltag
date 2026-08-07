"""Shared response-mapping primitives: baseline/RMS, lag-shifted pose QC, and spatial smoothing.

This package replaces the load_trial/interpolate_pose/smooth_grid definitions
that were previously copy-pasted into notebooks 01-09. See
analysis/response_pipeline/README.md for the pipeline stages and the physical
reasoning behind each one.
"""

from __future__ import annotations

from .config import TrialConfig
from .io import SessionData, find_session_by_trial_id, load_session
from .pose import PoseSample, QualifyingRow, build_pose_series, interpolate_pose, qualifying_rows
from .response import (
    compute_baseline,
    compute_baseline_array,
    median_fingerprint,
    normalized_channels,
    percentile,
    sensor_response,
    summarize_responses,
)
from .smoothing import smooth_grid

__all__ = [
    "TrialConfig",
    "SessionData",
    "find_session_by_trial_id",
    "load_session",
    "PoseSample",
    "QualifyingRow",
    "build_pose_series",
    "interpolate_pose",
    "qualifying_rows",
    "compute_baseline",
    "compute_baseline_array",
    "sensor_response",
    "normalized_channels",
    "percentile",
    "median_fingerprint",
    "summarize_responses",
    "smooth_grid",
]
