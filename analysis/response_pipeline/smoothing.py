"""Gaussian-weighted spatial smoothing onto a fixed grid."""

from __future__ import annotations

import math
from typing import Callable, Sequence, TypeVar

Point = TypeVar("Point")


def smooth_grid(
    points: Sequence[Point],
    grid_x: Sequence[float],
    grid_y: Sequence[float],
    *,
    radius_cm: float,
    sigma_cm: float,
    min_count: int = 2,
    x_of: Callable[[Point], float] = lambda point: point.x,  # type: ignore[attr-defined]
    y_of: Callable[[Point], float] = lambda point: point.y,  # type: ignore[attr-defined]
    value_of: Callable[[Point], float] = lambda point: point.response,  # type: ignore[attr-defined]
) -> list[list[float | None]]:
    """Weighted-average `points` onto (grid_y x grid_x), Gaussian-weighted by distance.

    A cell is None (unsupported) unless at least `min_count` points fall
    within `radius_cm`, so a cell is never interpolated from a single nearby
    reading. `x_of`/`y_of`/`value_of` default to attribute access
    (point.x/point.y/point.response, matching pose.QualifyingRow) and can be
    overridden for plain dicts or other point types.
    """
    grid: list[list[float | None]] = []
    for gy in grid_y:
        grid_row: list[float | None] = []
        for gx in grid_x:
            nearby: list[tuple[float, float]] = []
            for point in points:
                distance_sq = (x_of(point) - gx) ** 2 + (y_of(point) - gy) ** 2
                if distance_sq <= radius_cm**2:
                    weight = math.exp(-distance_sq / (2 * sigma_cm**2))
                    nearby.append((weight, value_of(point)))
            if len(nearby) < min_count:
                grid_row.append(None)
            else:
                total_weight = sum(weight for weight, _ in nearby)
                grid_row.append(sum(weight * value for weight, value in nearby) / total_weight)
        grid.append(grid_row)
    return grid
