from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .core import (
    SessionInfo,
    discover_sessions,
    load_annotation,
    load_camera_matrix,
    project_desk_points,
    reconstruct_camera_to_desk,
    write_annotation,
)
from .visualization import nearest_video_frame_index


@dataclass(frozen=True)
class DeskReferenceView:
    image_rgb: np.ndarray
    raw_image_rgb: np.ndarray
    bounds_cm: tuple[float, float, float, float]
    source_raw_row: int
    video_frame_index: int


def _desk_view_bounds(frame: pd.DataFrame) -> tuple[float, float, float, float]:
    valid = frame["snout_position_valid"].astype(bool)
    xy = frame.loc[valid, ["snout_desk_x_cm", "snout_desk_y_cm"]].to_numpy(dtype=float)
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    if not len(xy):
        raise ValueError("No valid desk-coordinate snout positions are available")
    minimum = xy.min(axis=0) - 8.0
    maximum = xy.max(axis=0) + 8.0
    center = (minimum + maximum) / 2.0
    span = np.maximum(maximum - minimum, 40.0)
    minimum = center - span / 2.0
    maximum = center + span / 2.0
    return float(minimum[0]), float(maximum[0]), float(minimum[1]), float(maximum[1])


def _read_video_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Cannot read video frame {index}")
    return frame


def _qualified_desk_rows(frame: pd.DataFrame) -> np.ndarray:
    required = [
        "tag_6_cam_x_cm",
        "tag_6_cam_y_cm",
        "tag_6_cam_z_cm",
        "tag_6_cam_rvec_x_rad",
        "tag_6_cam_rvec_y_rad",
        "tag_6_cam_rvec_z_rad",
        "pose_elapsed_s",
    ]
    missing = [name for name in ["tag_6_visible", *required] if name not in frame]
    if missing:
        raise ValueError("Missing desk-calibration columns: " + ", ".join(missing))
    qualified = frame["tag_6_visible"].astype(bool).to_numpy()
    qualified &= np.all(np.isfinite(frame[required].to_numpy(dtype=float)), axis=1)
    return frame.index[qualified].to_numpy(dtype=np.int64)


def nearest_qualified_desk_row(frame: pd.DataFrame, requested_raw_row: int) -> int:
    candidates = _qualified_desk_rows(frame)
    if not len(candidates):
        raise ValueError("No frame has a qualified desk-tag pose")
    distances = np.abs(candidates - int(requested_raw_row))
    return int(candidates[np.argmin(distances)])


def build_calibrated_desk_reference(
    info: SessionInfo,
    frame: pd.DataFrame,
    requested_raw_row: int,
    capture: cv2.VideoCapture | None = None,
    camera_matrix: np.ndarray | None = None,
    bounds_cm: tuple[float, float, float, float] | None = None,
    output_width: int = 800,
) -> DeskReferenceView:
    """Warp one raw frame using the desk-tag pose measured at that same CSV row."""
    camera_matrix = load_camera_matrix() if camera_matrix is None else camera_matrix
    bounds = _desk_view_bounds(frame) if bounds_cm is None else bounds_cm
    x_min, x_max, y_min, y_max = bounds
    output_height = int(round(output_width * (y_max - y_min) / (x_max - x_min)))
    output_height = int(np.clip(output_height, 480, 900))
    desk_corners = np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
        dtype=np.float64,
    )
    destination = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )

    source_raw_row = nearest_qualified_desk_row(frame, requested_raw_row)
    row = frame.loc[source_raw_row]
    desk_to_camera = reconstruct_camera_to_desk(row)
    source = project_desk_points(
        desk_corners,
        camera_matrix,
        desk_to_camera,
    ).astype(np.float32)

    owns_capture = capture is None
    active_capture = capture or cv2.VideoCapture(str(info.video_path))
    if not active_capture.isOpened():
        raise RuntimeError(f"Cannot open video: {info.video_path}")
    frame_count = int(active_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = float(frame["pose_elapsed_s"].max())
    try:
        video_index = nearest_video_frame_index(
            float(row["pose_elapsed_s"]),
            duration_s,
            frame_count,
        )
        raw_bgr = _read_video_frame(active_capture, video_index)
    finally:
        if owns_capture:
            active_capture.release()

    homography = cv2.getPerspectiveTransform(source, destination)
    desk_bgr = cv2.warpPerspective(
        raw_bgr,
        homography,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return DeskReferenceView(
        image_rgb=cv2.cvtColor(desk_bgr, cv2.COLOR_BGR2RGB),
        raw_image_rgb=cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB),
        bounds_cm=bounds,
        source_raw_row=source_raw_row,
        video_frame_index=video_index,
    )


def undo_last_click(
    points: dict[str, list[tuple[float, float]]],
    click_history: list[str],
) -> str | None:
    if not click_history:
        return None
    feature = click_history.pop()
    if feature not in points or not points[feature]:
        raise RuntimeError("Click history is inconsistent with annotation points")
    points[feature].pop()
    return feature


def annotation_status_table(
    sessions: list[SessionInfo] | None = None,
) -> pd.DataFrame:
    sessions = sessions or discover_sessions()
    records: list[dict[str, Any]] = []
    for info in sessions:
        status = "missing"
        usable_start: int | None = None
        usable_end: int | None = None
        fit_rms_cm: float | None = None
        if info.annotation_path.is_file():
            try:
                annotation = load_annotation(info)
                status = "valid"
                usable_start = annotation.usable_start_row
                usable_end = annotation.usable_end_row
                with info.annotation_path.open(encoding="utf-8") as handle:
                    import json

                    fit_rms_cm = float(json.load(handle)["paper"]["fit_rms_cm"])
            except (KeyError, TypeError, ValueError):
                status = "invalid"
        records.append(
            {
                "session_id": info.session_id,
                "trial_id": info.trial_id,
                "layout": info.layout,
                "annotation": status,
                "usable_start_row": usable_start,
                "usable_end_row": usable_end,
                "paper_fit_rms_cm": fit_rms_cm,
            }
        )
    return pd.DataFrame.from_records(records)


class SpatialAnnotationTool:
    def __init__(self, sessions: list[SessionInfo] | None = None) -> None:
        import ipywidgets as widgets
        import matplotlib.pyplot as plt

        self.sessions = sessions or discover_sessions()
        self.camera_matrix = load_camera_matrix()
        self.frame: pd.DataFrame | None = None
        self.desk_bounds: tuple[float, float, float, float] | None = None
        self.reference_view: DeskReferenceView | None = None
        self.reference_requested_row: int | None = None
        self.capture: cv2.VideoCapture | None = None
        self.points: dict[str, list[tuple[float, float]]] = {}
        self.click_history: list[str] = []
        self.info = self.sessions[0]
        self._loading = False
        self._widgets = widgets

        self.session_dropdown = widgets.Dropdown(
            options=[
                (f"{item.session_id} | {item.layout}", index)
                for index, item in enumerate(self.sessions)
            ],
            description="Session",
            layout=widgets.Layout(width="800px"),
        )
        self.feature_dropdown = widgets.Dropdown(description="Click target")
        self.range_slider = widgets.IntRangeSlider(
            description="Usable rows",
            continuous_update=False,
            layout=widgets.Layout(width="800px"),
        )
        self.raw_row_slider = widgets.IntSlider(
            description="Reference row",
            continuous_update=False,
            layout=widgets.Layout(width="800px"),
        )
        self.undo_button = widgets.Button(description="Undo last click")
        self.reset_button = widgets.Button(
            description="Reset current",
            button_style="warning",
        )
        self.save_button = widgets.Button(
            description="Save final JSON",
            button_style="success",
        )
        self.status = widgets.HTML()
        self.figure, (self.desk_axis, self.raw_axis) = plt.subplots(
            1,
            2,
            figsize=(13, 6),
            constrained_layout=True,
        )
        self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.session_dropdown.observe(self._on_session_change, names="value")
        self.feature_dropdown.observe(self._on_feature_change, names="value")
        self.raw_row_slider.observe(self._on_raw_row_change, names="value")
        self.undo_button.on_click(self._on_undo)
        self.reset_button.on_click(self._on_reset)
        self.save_button.on_click(self._on_save)
        self.widget = widgets.VBox(
            [
                widgets.HTML(
                    "<b>Choose a sharp, unobstructed reference row, then click four "
                    "cyclic outer corners in the calibrated desk view. For paper use "
                    "top-left, top-right, bottom-right, bottom-left.</b>"
                ),
                self.session_dropdown,
                self.range_slider,
                self.raw_row_slider,
                self.feature_dropdown,
                widgets.HBox(
                    [self.undo_button, self.reset_button, self.save_button]
                ),
                self.status,
                self.figure.canvas,
            ]
        )
        self._load_session(0)

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _feature_names(self) -> list[str]:
        return ["paper", *self.info.source_names]

    def _load_session(self, index: int) -> None:
        self._loading = True
        try:
            self.close()
            self.info = self.sessions[index]
            self.status.value = "<b>Loading calibrated reference frame...</b>"
            self.frame = pd.read_csv(self.info.csv_path)
            active_rows = self.frame.index[
                self.frame["pcnose_flag"].eq(2)
            ].to_numpy()
            if not len(active_rows):
                raise ValueError(
                    f"No flag-2 rows in session {self.info.session_id}"
                )
            default_row = nearest_qualified_desk_row(
                self.frame,
                int(active_rows[len(active_rows) // 2]),
            )
            self.range_slider.min = 0
            self.range_slider.max = len(self.frame) - 1
            self.range_slider.value = (
                int(active_rows[0]),
                int(active_rows[-1]),
            )
            self.raw_row_slider.min = 0
            self.raw_row_slider.max = len(self.frame) - 1
            self.raw_row_slider.value = default_row
            self.feature_dropdown.options = self._feature_names()
            self.feature_dropdown.value = "paper"
            self.points = {name: [] for name in self._feature_names()}
            self.click_history = []
            self.desk_bounds = _desk_view_bounds(self.frame)
            self.reference_view = None
            self.reference_requested_row = None
            self.capture = cv2.VideoCapture(str(self.info.video_path))
            if not self.capture.isOpened():
                raise RuntimeError(f"Cannot open video: {self.info.video_path}")
        finally:
            self._loading = False
        self._redraw()
        self._update_status()

    def _on_session_change(self, change: dict[str, Any]) -> None:
        if change["name"] == "value" and not self._loading:
            self._load_session(int(change["new"]))

    def _on_feature_change(self, change: dict[str, Any]) -> None:
        if change["name"] == "value" and not self._loading:
            self._update_status()

    def _on_raw_row_change(self, change: dict[str, Any]) -> None:
        if change["name"] == "value" and not self._loading:
            self.reference_view = None
            self.reference_requested_row = None
            self._redraw()
            self._update_status()

    def _on_click(self, event: Any) -> None:
        if event.inaxes is not self.desk_axis:
            return
        if event.xdata is None or event.ydata is None:
            return
        feature = str(self.feature_dropdown.value)
        if len(self.points[feature]) >= 4:
            self.status.value = (
                f"<b>{feature} already has four points. Reset it before "
                "clicking again.</b>"
            )
            return
        self.points[feature].append(
            (float(event.xdata), float(event.ydata))
        )
        self.click_history.append(feature)
        self._redraw()
        self._update_status()

    def _on_undo(self, _: Any) -> None:
        feature = undo_last_click(self.points, self.click_history)
        if feature is None:
            self._update_status("Nothing to undo.")
            return
        self.feature_dropdown.value = feature
        self._redraw()
        self._update_status(f"Removed the last {feature} point.")

    def _on_reset(self, _: Any) -> None:
        feature = str(self.feature_dropdown.value)
        self.points[feature] = []
        self.click_history = [
            name for name in self.click_history if name != feature
        ]
        self._redraw()
        self._update_status(f"Reset all {feature} points.")

    def _on_save(self, _: Any) -> None:
        incomplete = [
            name for name, points in self.points.items() if len(points) != 4
        ]
        if incomplete:
            self.status.value = (
                "<b>Cannot save; incomplete: "
                + ", ".join(incomplete)
                + "</b>"
            )
            return
        reference = self._load_reference_view()
        start, end = self.range_slider.value
        diagnostics = {
            "annotation_notebook": (
                "analysis/notebooks/"
                "long_random_sequence_spatial_annotation.ipynb"
            ),
            "desk_reference_view": {
                "bounds_desk_cm": list(reference.bounds_cm),
                "image_shape": list(reference.image_rgb.shape),
                "source_raw_row_at_save": reference.source_raw_row,
                "video_frame_index_at_save": reference.video_frame_index,
            },
            "click_order": (
                "cyclic outer corners; paper starts at top-left"
            ),
            "qc_raw_row_at_save": reference.source_raw_row,
        }
        try:
            write_annotation(
                self.info,
                int(start),
                int(end),
                np.asarray(self.points["paper"], dtype=float),
                {
                    name: np.asarray(self.points[name], dtype=float)
                    for name in self.info.source_names
                },
                diagnostics=diagnostics,
            )
        except Exception as error:
            self.status.value = (
                f"<b>Save failed: {type(error).__name__}: {error}</b>"
            )
            raise
        self.status.value = (
            f"<b>Saved final annotation: {self.info.annotation_path}</b>"
        )

    def _update_status(self, note: str | None = None) -> None:
        feature = str(self.feature_dropdown.value)
        counts = ", ".join(
            f"{name}: {len(points)}/4"
            for name, points in self.points.items()
        )
        saved = (
            "already exists"
            if self.info.annotation_path.is_file()
            else "not yet saved"
        )
        reference_text = ""
        if self.reference_view is not None:
            requested = int(self.raw_row_slider.value)
            actual = self.reference_view.source_raw_row
            reference_text = f" | reference row {actual}"
            if actual != requested:
                reference_text += f" (nearest qualified to {requested})"
        note_text = "" if note is None else f" <b>{note}</b>"
        self.status.value = (
            f"<b>Current: {feature}</b> | {counts} | JSON {saved}"
            f"{reference_text}. Saving overwrites this session annotation."
            f"{note_text}"
        )

    def _load_reference_view(self) -> DeskReferenceView:
        if self.frame is None or self.capture is None:
            raise RuntimeError("No annotation session is loaded")
        requested = int(self.raw_row_slider.value)
        if (
            self.reference_view is None
            or self.reference_requested_row != requested
        ):
            self.reference_view = build_calibrated_desk_reference(
                self.info,
                self.frame,
                requested,
                capture=self.capture,
                camera_matrix=self.camera_matrix,
                bounds_cm=self.desk_bounds,
            )
            self.reference_requested_row = requested
        return self.reference_view

    def _redraw(self) -> None:
        reference = self._load_reference_view()
        self.desk_axis.clear()
        x_min, x_max, y_min, y_max = reference.bounds_cm
        self.desk_axis.imshow(
            reference.image_rgb,
            extent=(x_min, x_max, y_max, y_min),
        )
        colors = {
            "paper": "white",
            "mint": "lime",
            "lavender": "violet",
        }
        for name, points in self.points.items():
            if not points:
                continue
            array = np.asarray(points)
            self.desk_axis.plot(
                array[:, 0],
                array[:, 1],
                "o-",
                color=colors[name],
                label=name,
            )
            for index, point in enumerate(array):
                self.desk_axis.text(
                    point[0],
                    point[1],
                    str(index + 1),
                    color=colors[name],
                )
            if len(points) == 4:
                closed = np.vstack([array, array[0]])
                self.desk_axis.plot(
                    closed[:, 0],
                    closed[:, 1],
                    color=colors[name],
                    linewidth=2,
                )
        self.desk_axis.set_title(
            "Calibrated single-frame desk view "
            f"(CSV row {reference.source_raw_row}; click here)"
        )
        self.desk_axis.set_xlabel("Desk x (cm; display right)")
        self.desk_axis.set_ylabel("Desk y (cm; display down)")
        if any(self.points.values()):
            self.desk_axis.legend(loc="upper right")
        self._redraw_raw(reference)
        self.figure.canvas.draw_idle()

    def _redraw_raw(self, reference: DeskReferenceView) -> None:
        if self.frame is None:
            return
        row = self.frame.loc[reference.source_raw_row]
        self.raw_axis.clear()
        self.raw_axis.imshow(reference.raw_image_rgb)
        colors = {
            "paper": "white",
            "mint": "lime",
            "lavender": "violet",
        }
        desk_to_camera = reconstruct_camera_to_desk(row)
        for name, points in self.points.items():
            if len(points) < 2:
                continue
            pixels = project_desk_points(
                np.asarray(points),
                self.camera_matrix,
                desk_to_camera,
            )
            closed = (
                np.vstack([pixels, pixels[0]])
                if len(points) == 4
                else pixels
            )
            self.raw_axis.plot(
                closed[:, 0],
                closed[:, 1],
                "o-",
                color=colors[name],
                linewidth=2,
            )
        self.raw_axis.set_title(
            "Matching raw frame and reprojection QC "
            f"| CSV row {reference.source_raw_row} "
            f"| video frame {reference.video_frame_index}"
        )
        self.raw_axis.set_axis_off()
        self.figure.canvas.draw_idle()
