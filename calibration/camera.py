from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs


COLOR_WIDTH = 1280
COLOR_HEIGHT = 720
COLOR_FPS = 30


@dataclass(frozen=True)
class RectifiedColorFrame:
    image: np.ndarray
    camera_timestamp_ms: float
    frame_number: int


def _camera_matrix(intrinsics: rs.intrinsics) -> np.ndarray:
    return np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _intrinsics_record(intrinsics: rs.intrinsics) -> dict[str, Any]:
    matrix = _camera_matrix(intrinsics)
    return {
        "image_width": int(intrinsics.width),
        "image_height": int(intrinsics.height),
        "fps": COLOR_FPS,
        "source_K": matrix,
        "source_distortion_model": intrinsics.model.name,
        "source_distortion_coefficients": [float(value) for value in intrinsics.coeffs],
        "rectified_K": matrix.copy(),
    }


def _rs_intrinsics(record: dict[str, Any]) -> rs.intrinsics:
    source_k = np.asarray(record["source_K"], dtype=np.float64)
    intrinsics = rs.intrinsics()
    intrinsics.width = int(record["image_width"])
    intrinsics.height = int(record["image_height"])
    intrinsics.fx = float(source_k[0, 0])
    intrinsics.fy = float(source_k[1, 1])
    intrinsics.ppx = float(source_k[0, 2])
    intrinsics.ppy = float(source_k[1, 2])
    intrinsics.model = getattr(rs.distortion, str(record["source_distortion_model"]))
    intrinsics.coeffs = [
        float(value) for value in record["source_distortion_coefficients"]
    ]
    return intrinsics


def _rectification_maps(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    width = int(record["image_width"])
    height = int(record["image_height"])
    rectified_k = np.asarray(record["rectified_K"], dtype=np.float64)
    source_intrinsics = _rs_intrinsics(record)

    fx = float(rectified_k[0, 0])
    fy = float(rectified_k[1, 1])
    cx = float(rectified_k[0, 2])
    cy = float(rectified_k[1, 2])

    map_x = np.empty((height, width), dtype=np.float32)
    map_y = np.empty((height, width), dtype=np.float32)
    for v in range(height):
        y = (v - cy) / fy
        for u in range(width):
            x = (u - cx) / fx
            source_pixel = rs.rs2_project_point_to_pixel(
                source_intrinsics,
                [x, y, 1.0],
            )
            map_x[v, u] = source_pixel[0]
            map_y[v, u] = source_pixel[1]
    return map_x, map_y


class RectifiedColorCamera:
    def __init__(self, saved_intrinsics: dict[str, Any] | None) -> None:
        self._saved_intrinsics = saved_intrinsics
        self._pipeline: rs.pipeline | None = None
        self._map_x: np.ndarray | None = None
        self._map_y: np.ndarray | None = None
        self.intrinsics_record: dict[str, Any] | None = None

    def start(self) -> None:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(
            rs.stream.color,
            COLOR_WIDTH,
            COLOR_HEIGHT,
            rs.format.bgr8,
            COLOR_FPS,
        )
        profile = pipeline.start(config)
        try:
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            active_intrinsics = color_profile.get_intrinsics()
            record = self._saved_intrinsics or _intrinsics_record(active_intrinsics)
            map_x, map_y = _rectification_maps(record)
        except Exception:
            pipeline.stop()
            raise

        self._pipeline = pipeline
        self.intrinsics_record = record
        self._map_x = map_x
        self._map_y = map_y

    def read_frame(self) -> RectifiedColorFrame:
        if self._pipeline is None or self._map_x is None or self._map_y is None:
            raise RuntimeError("Camera is not started")
        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("RealSense returned no color frame")
        source = np.asanyarray(color_frame.get_data())
        image = cv2.remap(
            source,
            self._map_x,
            self._map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return RectifiedColorFrame(
            image=image,
            camera_timestamp_ms=float(color_frame.get_timestamp()),
            frame_number=int(color_frame.get_frame_number()),
        )

    def read(self) -> np.ndarray:
        return self.read_frame().image

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

    def __enter__(self) -> "RectifiedColorCamera":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.stop()
