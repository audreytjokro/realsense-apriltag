from __future__ import annotations

from collections.abc import Iterable, Sequence

import cv2
import numpy as np


TEXT_COLOR = (255, 255, 255)
TEXT_SHADOW = (0, 0, 0)
TAG_COLOR = (0, 220, 255)
QUALIFIED_TAG_COLOR = (0, 255, 0)
SNOUT_COLOR = (255, 0, 255)
PLANE_LINE_COLOR = (0, 255, 255)


def draw_text_lines(
    image: np.ndarray,
    lines: Sequence[str],
    origin: tuple[int, int] = (12, 26),
) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(
            image,
            line,
            (x + 1, y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            TEXT_SHADOW,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        y += 23


def draw_tag_observations(image: np.ndarray, observations: Iterable[object]) -> None:
    for observation in observations:
        corners = np.rint(observation.corners).astype(np.int32)
        color = QUALIFIED_TAG_COLOR if observation.qualified else TAG_COLOR
        cv2.polylines(image, [corners], True, color, 2, cv2.LINE_AA)
        anchor = tuple(corners[0])
        label = f"ID {observation.tag_id}"
        if np.isfinite(observation.reprojection_rms_px):
            label += f"  {observation.reprojection_rms_px:.2f}px"
        cv2.putText(
            image,
            label,
            (int(anchor[0]), int(anchor[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_axes(image: np.ndarray, projected_points: np.ndarray) -> None:
    points = np.rint(projected_points).astype(np.int32)
    origin, x_axis, y_axis, z_axis = [tuple(point) for point in points]
    cv2.line(image, origin, x_axis, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.line(image, origin, y_axis, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(image, origin, z_axis, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, "X", x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(image, "Y", y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(image, "Z", z_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


def draw_snout_geometry(
    image: np.ndarray,
    snout_pixel: np.ndarray,
    plane_pixel: np.ndarray,
) -> None:
    snout = tuple(np.rint(snout_pixel).astype(np.int32))
    plane = tuple(np.rint(plane_pixel).astype(np.int32))
    cv2.line(image, plane, snout, PLANE_LINE_COLOR, 3, cv2.LINE_AA)
    cv2.circle(image, plane, 5, PLANE_LINE_COLOR, -1, cv2.LINE_AA)
    cv2.circle(image, snout, 7, SNOUT_COLOR, -1, cv2.LINE_AA)


def draw_click(image: np.ndarray, pixel: tuple[int, int] | None) -> None:
    if pixel is None:
        return
    cv2.drawMarker(
        image,
        pixel,
        SNOUT_COLOR,
        cv2.MARKER_CROSS,
        28,
        3,
        cv2.LINE_AA,
    )


def window_was_closed(window_name: str) -> bool:
    return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1.0
