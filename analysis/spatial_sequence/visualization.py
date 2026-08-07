from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from h264_video import H264VideoWriter
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Rectangle

from .core import (
    PAPER_HALF_CM,
    PreparedData,
    apply_transform_2d,
    load_annotation,
    load_camera_matrix,
    project_desk_points,
    reconstruct_camera_to_desk,
)


VIDEO_FPS = 5.0
PROBABILITY_MINIMUM = 1e-4
PROBABILITY_PERCENTILE = 99.0
FRAGMENT_COLORS = ((35, 155, 255), (80, 210, 120), (220, 120, 70))
ALIGNMENT_WARNING = "APPROXIMATE RAW-FRAME ALIGNMENT"
AREA_CLASS_COLORS = {
    "none": "#6b7280",
    "mint": "#20a65a",
    "lavender": "#9c5bc3",
}
SOURCE_BOUNDARY_COLORS = {"mint": "#20a65a", "lavender": "#9c5bc3"}
PARITY_POINT_COLOR = "#2869a6"
PARITY_POINT_SIZE = 9.0
PARITY_POINT_ALPHA = 0.11


def nearest_video_frame_index(
    pose_elapsed_s: float,
    session_duration_s: float,
    frame_count: int,
) -> int:
    if frame_count <= 0 or session_duration_s <= 0 or not np.isfinite(pose_elapsed_s):
        raise ValueError("Valid elapsed time, duration, and frame count are required")
    fraction = np.clip(pose_elapsed_s / session_duration_s, 0.0, 1.0)
    return int(np.clip(np.rint(fraction * (frame_count - 1)), 0, frame_count - 1))


def _probability_normalized(probability: np.ndarray | float, maximum: float) -> np.ndarray:
    clipped = np.clip(probability, PROBABILITY_MINIMUM, maximum)
    normalized = (
        np.log10(clipped) - np.log10(PROBABILITY_MINIMUM)
    ) / (
        np.log10(maximum) - np.log10(PROBABILITY_MINIMUM)
    )
    return np.clip(normalized, 0.0, 1.0)


def position_probability_maximum(joint_probabilities: np.ndarray) -> float:
    probabilities = np.asarray(joint_probabilities, dtype=np.float64)
    if probabilities.ndim != 3 or probabilities.shape[1:] != (13, 13):
        raise ValueError("Position joint probabilities must have shape (N, 13, 13)")
    finite = probabilities[np.isfinite(probabilities)]
    if finite.size == 0:
        raise ValueError("Position joint probabilities contain no finite interior values")
    percentile = float(np.percentile(finite, PROBABILITY_PERCENTILE))
    return max(percentile, PROBABILITY_MINIMUM * (1.0 + 1e-6))


def _probability_color(probability: float, maximum: float) -> tuple[int, int, int]:
    normalized = float(_probability_normalized(probability, maximum))
    rgb = np.asarray(colormaps["magma"](normalized)[:3]) * 255
    return tuple(int(value) for value in rgb[::-1])


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_raw_geometry(
    image: np.ndarray,
    row: object,
    annotation: object,
    camera_matrix: np.ndarray,
    gt_available: bool,
) -> None:
    try:
        camera_to_desk = reconstruct_camera_to_desk(row)
        paper_desk = apply_transform_2d(
            annotation.paper_to_desk,
            np.array(
                [
                    [-PAPER_HALF_CM, -PAPER_HALF_CM],
                    [PAPER_HALF_CM, -PAPER_HALF_CM],
                    [PAPER_HALF_CM, PAPER_HALF_CM],
                    [-PAPER_HALF_CM, PAPER_HALF_CM],
                ]
            ),
        )
        paper_pixels = project_desk_points(paper_desk, camera_matrix, camera_to_desk)
        cv2.polylines(
            image,
            [np.rint(paper_pixels).astype(np.int32)],
            True,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        source_colors = {"mint": (70, 220, 80), "lavender": (230, 120, 220)}
        for name, polygon_paper in annotation.source_polygons_paper_cm.items():
            polygon_desk = apply_transform_2d(annotation.paper_to_desk, polygon_paper)
            pixels = project_desk_points(polygon_desk, camera_matrix, camera_to_desk)
            cv2.polylines(
                image,
                [np.rint(pixels).astype(np.int32)],
                True,
                source_colors[name],
                3,
                cv2.LINE_AA,
            )
    except (KeyError, ValueError):
        _draw_text(image, "Desk geometry unavailable", (18, 58), color=(80, 180, 255))

    if gt_available:
        camera_xyz = np.asarray(
            [
                row["snout_cam_x_cm"],
                row["snout_cam_y_cm"],
                row["snout_cam_z_cm"],
            ],
            dtype=np.float64,
        )
        if np.all(np.isfinite(camera_xyz)) and camera_xyz[2] > 0:
            u = camera_matrix[0, 0] * camera_xyz[0] / camera_xyz[2] + camera_matrix[0, 2]
            v = camera_matrix[1, 1] * camera_xyz[1] / camera_xyz[2] + camera_matrix[1, 2]
            cv2.drawMarker(
                image,
                (int(round(u)), int(round(v))),
                (0, 255, 255),
                cv2.MARKER_CROSS,
                22,
                3,
                cv2.LINE_AA,
            )
    else:
        _draw_text(image, "GT unavailable", (18, 86), color=(0, 220, 255), thickness=2)


def render_position_panel(
    joint_probabilities: np.ndarray,
    truth_xy_cm: np.ndarray,
    gt_available: bool,
    probability_maximum: float,
    size: int = 640,
) -> np.ndarray:
    joint_probabilities = np.asarray(joint_probabilities, dtype=np.float64)
    if joint_probabilities.shape != (13, 13):
        raise ValueError("Position panel requires a 13x13 joint distribution")
    normalized = _probability_normalized(joint_probabilities, probability_maximum)
    rgb = (colormaps["magma"](normalized)[..., :3] * 255).astype(np.uint8)
    heatmap = cv2.resize(rgb[..., ::-1], (520, 520), interpolation=cv2.INTER_NEAREST)
    panel = np.full((size, size, 3), 245, dtype=np.uint8)
    offset = 60
    panel[offset : offset + 520, offset : offset + 520] = heatmap
    cv2.rectangle(panel, (offset, offset), (offset + 520, offset + 520), (35, 35, 35), 2)

    cv2.putText(
        panel,
        "Joint paper probability (13 x 13)",
        (145, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    if gt_available:
        truth = np.asarray(truth_xy_cm, dtype=np.float64).reshape(2)
        if np.all(np.abs(truth) <= PAPER_HALF_CM):
            u = offset + int(round((truth[0] + PAPER_HALF_CM) / (2 * PAPER_HALF_CM) * 520))
            v = offset + int(round((truth[1] + PAPER_HALF_CM) / (2 * PAPER_HALF_CM) * 520))
            cv2.drawMarker(
                panel,
                (u, v),
                (0, 255, 255),
                cv2.MARKER_CROSS,
                22,
                3,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                panel,
                f"GT outside paper: ({truth[0]:.1f}, {truth[1]:.1f}) cm",
                (150, 602),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
    else:
        cv2.putText(
            panel,
            "GT unavailable",
            (235, 330),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 170, 220),
            2,
            cv2.LINE_AA,
        )
    return panel


def _spatial_axis(
    figure: Figure,
    rows: int,
    columns: int,
    index: int,
    title: str,
    *,
    show_x_label: bool = True,
    show_y_label: bool = True,
):
    axis = figure.add_subplot(rows, columns, index)
    axis.add_patch(
        Rectangle(
            (-PAPER_HALF_CM, -PAPER_HALF_CM),
            2 * PAPER_HALF_CM,
            2 * PAPER_HALF_CM,
            fill=False,
            edgecolor="black",
            linewidth=1.2,
        )
    )
    axis.set_xlim(-PAPER_HALF_CM - 1.0, PAPER_HALF_CM + 1.0)
    axis.set_ylim(PAPER_HALF_CM + 1.0, -PAPER_HALF_CM - 1.0)
    axis.set_aspect("equal")
    axis.set_title(title, fontsize=9)
    if show_x_label:
        axis.set_xlabel("paper x (cm)", fontsize=7)
    if show_y_label:
        axis.set_ylabel("paper y (cm)", fontsize=7)
    axis.tick_params(labelsize=7)
    return axis


def _model_display_name(architecture: str) -> str:
    if architecture == "temporal-cnn":
        return "TCNN"
    if architecture == "transformer":
        return "Transformer"
    if architecture == "bigru":
        return "BiGRU"
    if architecture == "bigru-v2":
        return "BiGRU v2"
    raise ValueError(f"Unknown architecture: {architecture}")


def _configuration_title(
    target: str,
    architecture: str,
    temporal_mode: str,
    *,
    session: str | None = None,
    seed: int | None = None,
    suffix: str | None = None,
) -> str:
    task = target.replace("_", " ").title()
    if suffix:
        task = f"{task} {suffix}"
    parts = [task, f"{temporal_mode.title()} {_model_display_name(architecture)}"]
    if seed is not None:
        parts.append(f"Seed {seed}")
    if session:
        parts.append(session)
    return " — ".join(parts)


def _draw_source_layout(
    axis: object,
    source_polygons_paper_cm: dict[str, np.ndarray] | None,
) -> None:
    """Overlay the annotated source-strip layout for one session."""
    if not source_polygons_paper_cm:
        return
    for name, vertices in source_polygons_paper_cm.items():
        polygon = np.asarray(vertices, dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            continue
        color = SOURCE_BOUNDARY_COLORS.get(name, "black")
        axis.add_patch(
            Polygon(
                polygon,
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=1.0,
                linestyle=(0, (4, 3)),
                zorder=8,
            )
        )


def _categorical_points(
    axis: object,
    xy_cm: np.ndarray,
    classes: np.ndarray,
) -> None:
    """Render unordered area classes with a fixed discrete palette."""
    xy_cm = np.asarray(xy_cm, dtype=np.float64)
    class_values = np.asarray(classes, dtype=np.float64).reshape(-1)
    finite_classes = np.isfinite(class_values)
    discrete_classes = np.full(len(class_values), -1, dtype=np.int64)
    discrete_classes[finite_classes] = class_values[finite_classes].astype(np.int64)
    valid = (
        np.all(np.isfinite(xy_cm), axis=1)
        & finite_classes
        & np.isin(discrete_classes, (0, 1, 2))
    )
    palette = np.asarray(
        [AREA_CLASS_COLORS[name] for name in ("none", "mint", "lavender")]
    )
    axis.scatter(
        xy_cm[valid, 0],
        xy_cm[valid, 1],
        c=palette[discrete_classes[valid]],
        s=19.0,
        marker="o",
        linewidths=0.0,
        alpha=0.9,
        zorder=3,
    )


def _add_layout_legend(
    figure: Figure,
    source_polygons_paper_cm: dict[str, np.ndarray] | None,
    *,
    include_area_classes: bool,
) -> None:
    handles: list[Line2D] = []
    if include_area_classes:
        handles.extend(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=6,
                markerfacecolor=AREA_CLASS_COLORS[name],
                markeredgecolor="none",
                label=f"class: {name}",
            )
            for name in ("none", "mint", "lavender")
        )
    if source_polygons_paper_cm:
        handles.extend(
            Line2D(
                [],
                [],
                color=SOURCE_BOUNDARY_COLORS.get(name, "black"),
                linewidth=1.0,
                linestyle=(0, (4, 3)),
                label=f"{name} source boundary",
            )
            for name in source_polygons_paper_cm
        )
    if handles:
        figure.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.005),
            ncol=len(handles),
            frameon=False,
            fontsize=8,
        )


def _subset_anchor_aggregate(
    aggregate: dict[str, np.ndarray],
    selection: np.ndarray,
) -> dict[str, np.ndarray]:
    """Subset anchor-level arrays while preserving session lookup arrays."""
    selection = np.asarray(selection, dtype=bool)
    anchor_count = len(selection)
    result: dict[str, np.ndarray] = {}
    for key, value in aggregate.items():
        array = np.asarray(value)
        if (
            key not in {"session_ids", "session_slugs"}
            and array.ndim >= 1
            and array.shape[0] == anchor_count
        ):
            result[key] = array[selection]
        else:
            result[key] = array
    return result


def seed_probability_disagreement(probabilities: np.ndarray) -> np.ndarray:
    """Mean pairwise total-variation distance across seed probability vectors."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2:
        raise ValueError("Seed probabilities must have shape (S, N, C) with S >= 2")
    pairwise = [
        0.5 * np.sum(np.abs(values[left] - values[right]), axis=1)
        for left in range(values.shape[0])
        for right in range(left + 1, values.shape[0])
    ]
    return np.mean(np.stack(pairwise, axis=0), axis=0)


def seed_majority_classes(probabilities: np.ndarray) -> np.ndarray:
    """Per-anchor majority vote with mean probability as a deterministic tie-break."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[2] < 2:
        raise ValueError("Seed probabilities must have shape (S, N, C)")
    if not np.all(np.isfinite(values)):
        raise ValueError("Seed probabilities must be finite")
    votes = values.argmax(axis=2)
    counts = np.stack(
        [np.sum(votes == class_index, axis=0) for class_index in range(values.shape[2])],
        axis=1,
    )
    tied = counts == counts.max(axis=1, keepdims=True)
    tie_break_scores = np.where(tied, values.mean(axis=0), -np.inf)
    return tie_break_scores.argmax(axis=1)


def _finite_limits(*values: np.ndarray, nonnegative: bool = False) -> tuple[float, float]:
    finite_parts = [np.asarray(value)[np.isfinite(value)] for value in values]
    finite_parts = [part for part in finite_parts if part.size]
    if not finite_parts:
        return (0.0, 1.0)
    combined = np.concatenate(finite_parts)
    low = 0.0 if nonnegative else float(np.min(combined))
    high = float(np.max(combined))
    if high <= low:
        high = low + 1.0
    return low, high


def _gaussian_glyphs(
    axis: object,
    xy_cm: np.ndarray,
    values: np.ndarray,
    limits: tuple[float, float],
    *,
    cmap_name: str = "viridis",
    sigma_cm: float = 1.0,
    normalization: Normalize | None = None,
    colorbar_extend: str = "neither",
) -> None:
    """Draw independent Gaussian-like glyphs; this never creates an interpolated field."""
    xy_cm = np.asarray(xy_cm, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.all(np.isfinite(xy_cm), axis=1) & np.isfinite(values)
    if normalization is None:
        normalization = Normalize(vmin=limits[0], vmax=limits[1], clip=True)
    cmap = colormaps[cmap_name]
    points = xy_cm[valid]
    colors = cmap(normalization(values[valid]))
    for radius, alpha in ((2.0, 0.035), (1.5, 0.06), (1.0, 0.11), (0.5, 0.20)):
        collection = PatchCollection(
            [Circle(point, radius * sigma_cm) for point in points],
            facecolors=colors,
            edgecolors="none",
            alpha=alpha,
            match_original=False,
        )
        axis.add_collection(collection)
    axis.scatter(
        points[:, 0],
        points[:, 1],
        c=colors,
        s=7.0,
        marker="o",
        linewidths=0.0,
        zorder=3,
    )
    scalar = colormaps[cmap_name]
    from matplotlib.cm import ScalarMappable

    colorbar = axis.figure.colorbar(
        ScalarMappable(norm=normalization, cmap=scalar),
        ax=axis,
        fraction=0.045,
        pad=0.025,
        extend=colorbar_extend,
    )
    if isinstance(normalization, LogNorm):
        ticks = np.geomspace(float(normalization.vmin), float(normalization.vmax), 5)
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels([f"{tick:.1f}" for tick in ticks])
        colorbar.minorticks_off()


def _vector_panel(
    figure: Figure,
    rows: int,
    columns: int,
    index: int,
    title: str,
    xy_cm: np.ndarray,
    vectors: np.ndarray,
    limits: tuple[float, float],
    *,
    show_x_label: bool = True,
    show_y_label: bool = True,
) -> None:
    axis = _spatial_axis(
        figure,
        rows,
        columns,
        index,
        title,
        show_x_label=show_x_label,
        show_y_label=show_y_label,
    )
    speed = np.linalg.norm(vectors, axis=1)
    _gaussian_glyphs(axis, xy_cm, speed, limits, cmap_name="plasma")
    valid = np.all(np.isfinite(xy_cm), axis=1) & np.all(np.isfinite(vectors), axis=1)
    valid_indices = np.flatnonzero(valid & (speed > 1e-12))
    if len(valid_indices):
        stride = max(1, int(np.ceil(len(valid_indices) / 120)))
        arrow_indices = valid_indices[::stride]
        direction = vectors[arrow_indices] / speed[arrow_indices, None]
        display_vectors = 1.4 * direction
    else:
        arrow_indices = np.empty(0, dtype=np.int64)
        display_vectors = np.empty((0, 2), dtype=np.float64)
    axis.quiver(
        xy_cm[arrow_indices, 0],
        xy_cm[arrow_indices, 1],
        display_vectors[:, 0],
        display_vectors[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0020,
        color="black",
        alpha=0.55,
        zorder=5,
    )


def _height_normalizations(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> tuple[LogNorm, SymLogNorm]:
    truth_values = np.asarray(truth, dtype=np.float64).reshape(-1)
    positive_truth = truth_values[np.isfinite(truth_values) & (truth_values > 0)]
    if not positive_truth.size:
        raise ValueError("Height log normalization requires positive finite values")
    # A very small number of pose failures can produce heights an order of
    # magnitude above the physical working range.  Base the shared GT/prediction
    # scale on robust GT limits so those points remain visible as saturated
    # outliers instead of compressing almost every valid height into one color.
    height_low, height_high = (
        float(value) for value in np.quantile(positive_truth, (0.01, 0.99))
    )
    if height_high <= height_low:
        height_high = height_low * 1.01
    error = np.abs(np.asarray(prediction) - np.asarray(truth))
    finite_error = error[np.isfinite(error)]
    error_high = max(float(np.max(finite_error)), 0.05)
    return (
        LogNorm(vmin=height_low, vmax=height_high, clip=True),
        SymLogNorm(
            linthresh=0.05,
            linscale=1.0,
            vmin=0.0,
            vmax=error_high,
            base=10,
            clip=True,
        ),
    )


def render_run_diagnostics(
    prepared: PreparedData,
    aggregate: dict[str, np.ndarray],
    output_path: Path,
    *,
    architecture: str,
    temporal_mode: str,
    seed: int,
    _session_slug: str | None = None,
    _source_polygons_paper_cm: dict[str, np.ndarray] | None = None,
) -> list[Path]:
    """Render per-anchor diagnostics at measured GT positions without interpolation."""
    if prepared.target in {"area", "distance"} and _session_slug is None:
        session_index = np.asarray(aggregate["session_index"], dtype=np.int64)
        unique_sessions = np.unique(session_index)
        if len(unique_sessions) > 1:
            outputs: list[Path] = []
            session_slugs = np.asarray(aggregate["session_slugs"])
            for index in unique_sessions:
                numeric_index = int(index)
                slug = str(session_slugs[numeric_index])
                polygons = load_annotation(
                    prepared.sessions[numeric_index].info,
                    verify_hashes=False,
                ).source_polygons_paper_cm
                session_output = output_path.with_name(
                    f"{output_path.stem}_{prepared.target}_{slug}{output_path.suffix}"
                )
                outputs.extend(
                    render_run_diagnostics(
                        prepared,
                        _subset_anchor_aggregate(
                            aggregate, session_index == numeric_index
                        ),
                        session_output,
                        architecture=architecture,
                        temporal_mode=temporal_mode,
                        seed=seed,
                        _session_slug=slug,
                        _source_polygons_paper_cm=polygons,
                    )
                )
            return outputs
        if len(unique_sessions) == 1 and len(prepared.sessions) > 1:
            numeric_index = int(unique_sessions[0])
            slug = str(np.asarray(aggregate["session_slugs"])[numeric_index])
            polygons = load_annotation(
                prepared.sessions[numeric_index].info,
                verify_hashes=False,
            ).source_polygons_paper_cm
            session_output = output_path.with_name(
                f"{output_path.stem}_{prepared.target}_{slug}{output_path.suffix}"
            )
            return render_run_diagnostics(
                prepared,
                aggregate,
                session_output,
                architecture=architecture,
                temporal_mode=temporal_mode,
                seed=seed,
                _session_slug=slug,
                _source_polygons_paper_cm=polygons,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    xy_cm = np.asarray(aggregate["paper_xy_cm"], dtype=np.float64)
    target = prepared.target
    if target == "distance":
        distance_mask = np.asarray(aggregate["mask"], dtype=bool)
        source_count = sum(
            bool(np.any(distance_mask[:, source_index])) for source_index in range(2)
        )
        figure_size = (12.8, 4.4 * max(source_count, 1))
    elif target in {"height", "velocity"}:
        figure_size = (13.5, 4.6)
    else:
        figure_size = (12, 8)
    figure = Figure(figsize=figure_size, constrained_layout=True)
    FigureCanvasAgg(figure)

    if target == "position":
        joint = np.asarray(aggregate["joint_probabilities"], dtype=np.float64)
        centers = np.linspace(
            -PAPER_HALF_CM + PAPER_HALF_CM / 13.0,
            PAPER_HALF_CM - PAPER_HALF_CM / 13.0,
            13,
        )
        map_flat = joint.reshape(len(joint), -1).argmax(axis=1)
        map_xy = np.column_stack((centers[map_flat % 13], centers[map_flat // 13]))
        expected_xy = np.column_stack(
            (
                aggregate["x_probabilities"] @ centers,
                aggregate["y_probabilities"] @ centers,
            )
        )
        map_error = np.linalg.norm(map_xy - xy_cm, axis=1)
        expected_error = np.linalg.norm(expected_xy - xy_cm, axis=1)
        coordinate_limits = (-PAPER_HALF_CM, PAPER_HALF_CM)
        specifications = (
            (xy_cm[:, 0], coordinate_limits, "GT x (cm)", "coolwarm"),
            (expected_xy[:, 0], coordinate_limits, "Expected x (cm)", "coolwarm"),
            (xy_cm[:, 1], coordinate_limits, "GT y (cm)", "coolwarm"),
            (expected_xy[:, 1], coordinate_limits, "Expected y (cm)", "coolwarm"),
            (map_error, _finite_limits(map_error, nonnegative=True), "MAP error (cm)", "magma"),
            (
                expected_error,
                _finite_limits(expected_error, nonnegative=True),
                "Expected-coordinate error (cm)",
                "magma",
            ),
        )
        for index, (values, limits, title, cmap_name) in enumerate(specifications, 1):
            axis = _spatial_axis(figure, 2, 3, index, title)
            _gaussian_glyphs(axis, xy_cm, values, limits, cmap_name=cmap_name)
    elif target == "distance":
        truth = np.asarray(aggregate["truth"], dtype=np.float64)
        prediction = np.asarray(aggregate["prediction"], dtype=np.float64)
        mask = np.asarray(aggregate["mask"], dtype=bool)
        source_names = tuple(
            (source_index, source_name)
            for source_index, source_name in enumerate(("mint", "lavender"))
            if np.any(mask[:, source_index])
        )
        for row_index, (source_index, source_name) in enumerate(source_names):
            valid_truth = np.where(mask[:, source_index], truth[:, source_index], np.nan)
            valid_prediction = np.where(mask[:, source_index], prediction[:, source_index], np.nan)
            limits = _finite_limits(valid_truth, valid_prediction, nonnegative=True)
            error = np.abs(valid_prediction - valid_truth)
            for column, (values, value_limits, title, cmap_name) in enumerate(
                (
                    (valid_truth, limits, f"{source_name} GT distance (cm)", "viridis"),
                    (valid_prediction, limits, f"{source_name} predicted distance (cm)", "viridis"),
                    (error, _finite_limits(error, nonnegative=True), f"{source_name} absolute error (cm)", "magma"),
                ),
                1,
            ):
                axis = _spatial_axis(
                    figure,
                    len(source_names),
                    3,
                    row_index * 3 + column,
                    title,
                    show_x_label=row_index == len(source_names) - 1,
                    show_y_label=column == 1,
                )
                _draw_source_layout(axis, _source_polygons_paper_cm)
                _gaussian_glyphs(axis, xy_cm, values, value_limits, cmap_name=cmap_name)
    elif target == "area":
        truth = np.asarray(aggregate["truth"], dtype=np.float64)
        probabilities = np.asarray(aggregate["probabilities"], dtype=np.float64)
        prediction = probabilities.argmax(axis=1)
        for index, (values, title) in enumerate(
            ((truth, "Ground Truth"), (prediction, "Prediction")), 1
        ):
            axis = _spatial_axis(figure, 2, 3, index, title)
            _draw_source_layout(axis, _source_polygons_paper_cm)
            _categorical_points(axis, xy_cm, values)
        for class_index, name in enumerate(("none", "mint", "lavender")):
            axis = _spatial_axis(figure, 2, 3, 4 + class_index, f"P({name})")
            _draw_source_layout(axis, _source_polygons_paper_cm)
            _gaussian_glyphs(
                axis,
                xy_cm,
                probabilities[:, class_index],
                (0.0, 1.0),
                cmap_name="magma",
            )
    elif target == "height":
        truth = np.asarray(aggregate["truth"], dtype=np.float64).reshape(-1)
        prediction = np.asarray(aggregate["prediction"], dtype=np.float64).reshape(-1)
        height_norm, error_norm = _height_normalizations(truth, prediction)
        error = np.abs(prediction - truth)
        for index, (values, normalization, title, cmap_name) in enumerate(
            (
                (
                    truth,
                    height_norm,
                    "GT height (cm; robust log color)",
                    "viridis",
                ),
                (
                    prediction,
                    height_norm,
                    "Predicted height (cm; robust log color)",
                    "viridis",
                ),
                (error, error_norm, "Absolute error (cm; symlog color)", "magma"),
            ),
            1,
        ):
            axis = _spatial_axis(
                figure, 1, 3, index, title, show_y_label=index == 1
            )
            _gaussian_glyphs(
                axis,
                xy_cm,
                values,
                (float(normalization.vmin), float(normalization.vmax)),
                cmap_name=cmap_name,
                normalization=normalization,
                colorbar_extend="both" if index < 3 else "neither",
            )
    elif target == "velocity":
        truth = np.asarray(aggregate["truth"], dtype=np.float64)
        prediction = np.asarray(aggregate["prediction"], dtype=np.float64)
        common_limits = _finite_limits(
            np.linalg.norm(truth, axis=1),
            np.linalg.norm(prediction, axis=1),
            nonnegative=True,
        )
        _vector_panel(
            figure,
            1,
            3,
            1,
            "GT speed and direction",
            xy_cm,
            truth,
            common_limits,
            show_y_label=True,
        )
        _vector_panel(
            figure,
            1,
            3,
            2,
            "Predicted speed and direction",
            xy_cm,
            prediction,
            common_limits,
            show_y_label=False,
        )
        error = np.linalg.norm(prediction - truth, axis=1)
        axis = _spatial_axis(
            figure, 1, 3, 3, "Vector error (cm/s)", show_y_label=False
        )
        _gaussian_glyphs(
            axis,
            xy_cm,
            error,
            _finite_limits(error, nonnegative=True),
            cmap_name="magma",
        )
    else:
        raise ValueError(f"Unsupported diagnostic target: {target}")

    if target in {"area", "distance"}:
        _add_layout_legend(
            figure,
            _source_polygons_paper_cm,
            include_area_classes=target == "area",
        )

    session_slug = _session_slug
    if session_slug is None and target == "position" and len(prepared.sessions) == 1:
        session_slug = prepared.sessions[0].info.slug
    figure.suptitle(
        _configuration_title(
            target,
            architecture,
            temporal_mode,
            session=session_slug,
            seed=seed,
        ),
        fontsize=18,
        fontweight="semibold",
    )
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    figure.clear()
    return [output_path]


def render_cross_seed_diagnostics(
    target: str,
    aggregates: Sequence[dict[str, np.ndarray]],
    output_path: Path,
    *,
    architecture: str,
    temporal_mode: str,
    session_layouts: dict[str, dict[str, np.ndarray]] | None = None,
    constrained_layout: bool = True,
    _session_slug: str | None = None,
) -> list[Path]:
    """Render median prediction and seed disagreement on identical anchor rows."""
    if len(aggregates) != 5:
        raise ValueError("Cross-seed diagnostics require exactly five seed aggregates")
    reference = aggregates[0]
    for aggregate in aggregates[1:]:
        for key in ("session_index", "raw_row"):
            if not np.array_equal(reference[key], aggregate[key]):
                raise ValueError("Cross-seed diagnostics require identical physical targets")
    if target in {"area", "distance"} and _session_slug is None:
        session_index = np.asarray(reference["session_index"], dtype=np.int64)
        unique_sessions = np.unique(session_index)
        if len(unique_sessions) > 1:
            outputs: list[Path] = []
            session_slugs = np.asarray(reference["session_slugs"])
            for index in unique_sessions:
                numeric_index = int(index)
                slug = str(session_slugs[numeric_index])
                selection = session_index == numeric_index
                session_output = output_path.with_name(
                    f"{output_path.stem}_{slug}{output_path.suffix}"
                )
                outputs.extend(
                    render_cross_seed_diagnostics(
                        target,
                        [
                            _subset_anchor_aggregate(aggregate, selection)
                            for aggregate in aggregates
                        ],
                        session_output,
                        architecture=architecture,
                        temporal_mode=temporal_mode,
                        session_layouts=session_layouts,
                        constrained_layout=constrained_layout,
                        _session_slug=slug,
                    )
                )
            return outputs
    xy_cm = np.asarray(reference["paper_xy_cm"], dtype=np.float64)
    if target == "distance":
        distance_mask = np.asarray(reference["mask"], dtype=bool)
        source_count = sum(
            bool(np.any(distance_mask[:, source_index])) for source_index in range(2)
        )
        figure_size = (15.8, 4.4 * max(source_count, 1))
    elif target in {"height", "velocity"}:
        figure_size = (15.8, 4.8)
    else:
        figure_size = (12, 8)
    figure = Figure(figsize=figure_size, constrained_layout=constrained_layout)
    FigureCanvasAgg(figure)

    if target == "distance":
        truth = np.asarray(reference["truth"], dtype=np.float64)
        predictions = np.stack(
            [np.asarray(aggregate["prediction"], dtype=np.float64) for aggregate in aggregates]
        )
        median = np.median(predictions, axis=0)
        disagreement = np.std(predictions, axis=0, ddof=0)
        mask = np.asarray(reference["mask"], dtype=bool)
        source_names = tuple(
            (source_index, source_name)
            for source_index, source_name in enumerate(("mint", "lavender"))
            if np.any(mask[:, source_index])
        )
        for row_index, (source_index, source_name) in enumerate(source_names):
            gt = np.where(mask[:, source_index], truth[:, source_index], np.nan)
            prediction = np.where(mask[:, source_index], median[:, source_index], np.nan)
            seed_sd = np.where(mask[:, source_index], disagreement[:, source_index], np.nan)
            common = _finite_limits(gt, prediction, nonnegative=True)
            error = np.abs(prediction - gt)
            for column, (values, limits, title, cmap_name) in enumerate(
                (
                    (gt, common, f"{source_name} GT distance", "viridis"),
                    (prediction, common, f"{source_name} median prediction", "viridis"),
                    (error, _finite_limits(error, nonnegative=True), f"{source_name} median absolute error", "magma"),
                    (seed_sd, _finite_limits(seed_sd, nonnegative=True), f"{source_name} seed SD", "magma"),
                ),
                1,
            ):
                axis = _spatial_axis(
                    figure,
                    len(source_names),
                    4,
                    row_index * 4 + column,
                    title,
                    show_x_label=row_index == len(source_names) - 1,
                    show_y_label=column == 1,
                )
                _draw_source_layout(
                    axis,
                    None
                    if session_layouts is None
                    else session_layouts.get(_session_slug or ""),
                )
                _gaussian_glyphs(axis, xy_cm, values, limits, cmap_name=cmap_name)
    elif target == "area":
        truth = np.asarray(reference["truth"], dtype=np.float64)
        probabilities = np.stack(
            [np.asarray(aggregate["probabilities"], dtype=np.float64) for aggregate in aggregates]
        )
        mean_probability = probabilities.mean(axis=0)
        majority_prediction = seed_majority_classes(probabilities)
        disagreement = seed_probability_disagreement(probabilities)
        source_polygons = (
            None
            if session_layouts is None
            else session_layouts.get(_session_slug or "")
        )
        for index, (values, title) in enumerate(
            (
                (truth, "Ground Truth"),
                (
                    majority_prediction,
                    "Prediction",
                ),
            ),
            1,
        ):
            axis = _spatial_axis(figure, 2, 3, index, title)
            _draw_source_layout(axis, source_polygons)
            _categorical_points(axis, xy_cm, values)

        axis = _spatial_axis(
            figure,
            2,
            3,
            3,
            "Seed Disagreement",
        )
        _draw_source_layout(axis, source_polygons)
        _gaussian_glyphs(
            axis, xy_cm, disagreement, (0.0, 1.0), cmap_name="viridis"
        )

        for class_index, name in enumerate(("none", "mint", "lavender")):
            axis = _spatial_axis(
                figure,
                2,
                3,
                4 + class_index,
                f"Mean P({name})",
            )
            _draw_source_layout(axis, source_polygons)
            _gaussian_glyphs(
                axis,
                xy_cm,
                mean_probability[:, class_index],
                (0.0, 1.0),
                cmap_name="magma",
            )
        _add_layout_legend(
            figure,
            source_polygons,
            include_area_classes=True,
        )
    elif target == "height":
        truth = np.asarray(reference["truth"], dtype=np.float64).reshape(-1)
        predictions = np.stack(
            [np.asarray(aggregate["prediction"], dtype=np.float64).reshape(-1) for aggregate in aggregates]
        )
        median = np.median(predictions, axis=0)
        disagreement = np.std(predictions, axis=0, ddof=0)
        height_norm, error_norm = _height_normalizations(truth, median)
        error = np.abs(median - truth)
        specifications = (
            (
                truth,
                height_norm,
                "GT height (cm; robust log color)",
                "viridis",
            ),
            (
                median,
                height_norm,
                "Median height prediction (cm; robust log color)",
                "viridis",
            ),
            (
                error,
                error_norm,
                "Median absolute error (cm; symlog color)",
                "magma",
            ),
            (
                disagreement,
                Normalize(
                    vmin=0.0,
                    vmax=_finite_limits(disagreement, nonnegative=True)[1],
                    clip=True,
                ),
                "Seed SD (cm)",
                "magma",
            ),
        )
        for index, (values, normalization, title, cmap_name) in enumerate(
            specifications, 1
        ):
            axis = _spatial_axis(
                figure, 1, 4, index, title, show_y_label=index == 1
            )
            _gaussian_glyphs(
                axis,
                xy_cm,
                values,
                (float(normalization.vmin), float(normalization.vmax)),
                cmap_name=cmap_name,
                normalization=normalization,
                colorbar_extend="both" if index < 3 else "neither",
            )
    elif target == "velocity":
        truth = np.asarray(reference["truth"], dtype=np.float64)
        predictions = np.stack(
            [np.asarray(aggregate["prediction"], dtype=np.float64) for aggregate in aggregates]
        )
        median = np.median(predictions, axis=0)
        common = _finite_limits(
            np.linalg.norm(truth, axis=1),
            np.linalg.norm(median, axis=1),
            nonnegative=True,
        )
        _vector_panel(
            figure,
            1,
            4,
            1,
            "GT speed and direction",
            xy_cm,
            truth,
            common,
            show_y_label=True,
        )
        _vector_panel(
            figure,
            1,
            4,
            2,
            "Median speed and direction",
            xy_cm,
            median,
            common,
            show_y_label=False,
        )
        error = np.linalg.norm(median - truth, axis=1)
        axis = _spatial_axis(
            figure, 1, 4, 3, "Median vector error (cm/s)", show_y_label=False
        )
        _gaussian_glyphs(
            axis, xy_cm, error, _finite_limits(error, nonnegative=True), cmap_name="magma"
        )
        vector_sd = np.sqrt(np.sum(np.var(predictions, axis=0, ddof=0), axis=1))
        axis = _spatial_axis(
            figure, 1, 4, 4, "Seed vector SD (cm/s)", show_y_label=False
        )
        _gaussian_glyphs(
            axis,
            xy_cm,
            vector_sd,
            _finite_limits(vector_sd, nonnegative=True),
            cmap_name="magma",
        )
    else:
        raise ValueError("Cross-seed diagnostics support area, distance, height, and velocity")

    if target == "distance":
        _add_layout_legend(
            figure,
            None
            if session_layouts is None
            else session_layouts.get(_session_slug or ""),
            include_area_classes=False,
        )

    figure.suptitle(
        _configuration_title(
            target,
            architecture,
            temporal_mode,
            session=_session_slug,
        ),
        fontsize=18,
        fontweight="semibold",
    )
    if not constrained_layout and target == "area":
        panel_positions = {
            "Ground Truth": (0.055, 0.555, 0.25, 0.325),
            "Prediction": (0.365, 0.555, 0.25, 0.325),
            "Seed Disagreement": (0.675, 0.555, 0.25, 0.325),
            "Mean P(none)": (0.055, 0.130, 0.25, 0.325),
            "Mean P(mint)": (0.365, 0.130, 0.25, 0.325),
            "Mean P(lavender)": (0.675, 0.130, 0.25, 0.325),
        }
        data_axes = {axis.get_title(): axis for axis in figure.axes if axis.get_title()}
        if set(data_axes) != set(panel_positions):
            raise AssertionError("Dense area diagnostic axes are incomplete")
        for title, position in panel_positions.items():
            data_axes[title].set_position(position)
        colorbar_axes = [axis for axis in figure.axes if not axis.get_title()]
        colorbar_positions = (
            (0.930, 0.555, 0.014, 0.325),
            (0.310, 0.130, 0.014, 0.325),
            (0.620, 0.130, 0.014, 0.325),
            (0.930, 0.130, 0.014, 0.325),
        )
        if len(colorbar_axes) != len(colorbar_positions):
            raise AssertionError("Dense area diagnostic colorbars are incomplete")
        for axis, position in zip(colorbar_axes, colorbar_positions):
            axis.set_position(position)
        for legend in figure.legends:
            legend.set_bbox_to_anchor((0.5, 0.015))
    elif not constrained_layout:
        figure.subplots_adjust(
            left=0.055,
            right=0.985,
            bottom=0.105,
            top=0.90,
            wspace=0.30,
            hspace=0.34,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight" if constrained_layout else None,
    )
    figure.clear()
    return [output_path]


def _matching_anchor_array(left: np.ndarray, right: np.ndarray) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return False
    if left_array.dtype.kind in "fc" or right_array.dtype.kind in "fc":
        return bool(np.array_equal(left_array, right_array, equal_nan=True))
    return bool(np.array_equal(left_array, right_array))


def _validate_cross_seed_parity_aggregates(
    aggregates: Sequence[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if len(aggregates) != 5:
        raise ValueError("Cross-seed parity scatter requires exactly five aggregates")
    reference = aggregates[0]
    required = ("session_index", "raw_row", "truth", "mask", "prediction")
    for key in required:
        if key not in reference:
            raise ValueError(f"Cross-seed parity aggregate is missing {key!r}")
    for aggregate in aggregates[1:]:
        for key in required[:-1]:
            if key not in aggregate or not _matching_anchor_array(
                reference[key], aggregate[key]
            ):
                raise ValueError(
                    "Cross-seed parity scatter requires identical physical targets, "
                    f"truth, and masks; mismatch in {key!r}"
                )
        if "prediction" not in aggregate:
            raise ValueError("Cross-seed parity aggregate is missing 'prediction'")
    return reference


def _cross_seed_parity_series(
    target: str,
    aggregates: Sequence[dict[str, np.ndarray]],
) -> tuple[tuple[str, str, np.ndarray, np.ndarray], ...]:
    """Return title, unit, repeated GT, and five-seed predictions per panel."""
    if target not in {"distance", "height", "velocity"}:
        raise ValueError(
            "Cross-seed parity scatter supports distance, height, and velocity"
        )
    reference = _validate_cross_seed_parity_aggregates(aggregates)
    truth = np.asarray(reference["truth"], dtype=np.float64)
    mask = np.asarray(reference["mask"], dtype=bool)
    predictions = np.stack(
        [np.asarray(aggregate["prediction"], dtype=np.float64) for aggregate in aggregates]
    )

    if target == "distance":
        if truth.ndim != 2 or truth.shape[1] != 2 or mask.shape != truth.shape:
            raise ValueError("Distance parity data must have shape (N, 2)")
        if predictions.shape != (5,) + truth.shape:
            raise ValueError("Distance predictions must have shape (5, N, 2)")
        panels = []
        for source_index, source_name in enumerate(("Mint distance", "Lavender distance")):
            valid_anchor = mask[:, source_index]
            if not np.any(valid_anchor):
                continue
            gt = truth[valid_anchor, source_index]
            predicted = predictions[:, valid_anchor, source_index]
            if not np.all(np.isfinite(gt)) or not np.all(np.isfinite(predicted)):
                raise ValueError("Distance parity data contains non-finite valid values")
            panels.append(
                (
                    source_name,
                    "cm",
                    np.broadcast_to(gt, predicted.shape).reshape(-1),
                    predicted.reshape(-1),
                )
            )
        if not panels:
            raise ValueError("Distance parity data has no valid source anchors")
        return tuple(panels)

    if target == "height":
        gt = truth.reshape(-1)
        valid_anchor = mask.reshape(-1)
        if predictions.shape[0] != 5 or predictions.reshape(5, -1).shape[1] != len(gt):
            raise ValueError("Height predictions must have shape (5, N) or (5, N, 1)")
        predicted = predictions.reshape(5, -1)[:, valid_anchor]
        gt = gt[valid_anchor]
        if (
            not np.all(np.isfinite(gt))
            or not np.all(np.isfinite(predicted))
            or np.any(gt <= 0.0)
            or np.any(predicted <= 0.0)
        ):
            raise ValueError("Log-height parity data must be finite and strictly positive")
        return (
            (
                "Snout height",
                "cm",
                np.broadcast_to(gt, predicted.shape).reshape(-1),
                predicted.reshape(-1),
            ),
        )

    if truth.ndim != 2 or truth.shape[1] != 2 or mask.shape != truth.shape:
        raise ValueError("Velocity parity data must have shape (N, 2)")
    if predictions.shape != (5,) + truth.shape:
        raise ValueError("Velocity predictions must have shape (5, N, 2)")
    valid_anchor = np.all(mask, axis=1)
    gt_vectors = truth[valid_anchor]
    predicted_vectors = predictions[:, valid_anchor]
    if not np.all(np.isfinite(gt_vectors)) or not np.all(np.isfinite(predicted_vectors)):
        raise ValueError("Velocity parity data contains non-finite valid vectors")
    gt_speed = np.linalg.norm(gt_vectors, axis=1)
    predicted_speed = np.linalg.norm(predicted_vectors, axis=2)
    return (
        (
            "Snout speed",
            "cm/s",
            np.broadcast_to(gt_speed, predicted_speed.shape).reshape(-1),
            predicted_speed.reshape(-1),
        ),
    )


def _parity_limits(
    panels: Sequence[tuple[str, str, np.ndarray, np.ndarray]],
    *,
    logarithmic: bool,
) -> tuple[float, float]:
    combined = np.concatenate(
        [values for _, _, truth, prediction in panels for values in (truth, prediction)]
    )
    finite = combined[np.isfinite(combined)]
    if not finite.size:
        raise ValueError("Parity scatter has no finite values")
    if logarithmic:
        positive = finite[finite > 0.0]
        if len(positive) != len(finite):
            raise ValueError("Log parity scatter requires strictly positive values")
        low = float(np.min(positive))
        high = float(np.max(positive))
        if high <= low:
            high = low * 1.01
        log_low, log_high = np.log(low), np.log(high)
        padding = max(0.04 * (log_high - log_low), 1e-3)
        return float(np.exp(log_low - padding)), float(np.exp(log_high + padding))
    high = max(float(np.max(finite)), 0.0)
    return 0.0, high * 1.04 if high > 0.0 else 1.0


def _cross_seed_parity_figure(
    target: str,
    aggregates: Sequence[dict[str, np.ndarray]],
    architecture: str,
    temporal_mode: str,
) -> Figure:
    panels = _cross_seed_parity_series(target, aggregates)
    logarithmic = target == "height"
    limits = _parity_limits(panels, logarithmic=logarithmic)
    figure = Figure(
        figsize=(10.4, 4.9) if target == "distance" else (5.6, 5.2),
        constrained_layout=True,
    )
    FigureCanvasAgg(figure)
    for index, (title, unit, truth, prediction) in enumerate(panels, 1):
        axis = figure.add_subplot(1, len(panels), index)
        axis.scatter(
            truth,
            prediction,
            s=PARITY_POINT_SIZE,
            color=PARITY_POINT_COLOR,
            alpha=PARITY_POINT_ALPHA,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
        axis.plot(
            limits,
            limits,
            linestyle="--",
            linewidth=1.25,
            color="#202020",
            zorder=3,
        )
        if logarithmic:
            axis.set_xscale("log")
            axis.set_yscale("log")
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.65, zorder=0)
        if len(panels) > 1:
            axis.set_title(title)
        axis.set_xlabel(f"Ground truth ({unit})")
        axis.set_ylabel(f"Prediction ({unit})")
    figure.suptitle(
        _configuration_title(
            target,
            architecture,
            temporal_mode,
            suffix="Parity",
        ),
        fontsize=18,
        fontweight="semibold",
    )
    return figure


def render_cross_seed_parity_scatter(
    target: str,
    aggregates: Sequence[dict[str, np.ndarray]],
    output_path: Path,
    *,
    architecture: str,
    temporal_mode: str,
) -> Path:
    """Render pooled five-seed parity scatter without changing predictions."""
    figure = _cross_seed_parity_figure(
        target, aggregates, architecture, temporal_mode
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    figure.clear()
    return output_path


def render_position_video(
    prepared: PreparedData,
    aggregate: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    if prepared.target != "position" or len(prepared.sessions) != 1:
        raise ValueError("Position video rendering requires one position session")
    session = prepared.sessions[0]
    annotation = load_annotation(session.info)
    camera_matrix = load_camera_matrix()
    capture = cv2.VideoCapture(str(session.info.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {session.info.video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = float(session.frame["pose_elapsed_s"].max())
    input_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    left_width = int(round(640 * input_width / input_height))
    output_width = left_width + 640
    if output_width % 2:
        output_width += 1
    writer = H264VideoWriter(
        output_path,
        VIDEO_FPS,
        (output_width, 640),
    )

    previous_fragment: int | None = None
    probability_maximum = position_probability_maximum(aggregate["joint_probabilities"])
    try:
        for index in range(len(aggregate["raw_row"])):
            raw_row = int(aggregate["raw_row"][index])
            fragment = int(aggregate["fragment"][index])
            row = session.frame.loc[raw_row]
            video_index = nearest_video_frame_index(
                float(row["pose_elapsed_s"]),
                duration_s,
                frame_count,
            )
            capture.set(cv2.CAP_PROP_POS_FRAMES, video_index)
            ok, raw_frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Cannot read video frame {video_index} for source row {raw_row}"
                )
            gt_available = bool(np.asarray(aggregate["mask"][index]).all())
            _draw_raw_geometry(raw_frame, row, annotation, camera_matrix, gt_available)
            raw_frame = cv2.resize(raw_frame, (left_width, 640), interpolation=cv2.INTER_AREA)
            _draw_text(
                raw_frame,
                f"Validation fragment {chr(ord('A') + fragment)} | CSV row {raw_row}",
                (18, 30),
                color=FRAGMENT_COLORS[fragment],
                thickness=2,
            )
            warning = ALIGNMENT_WARNING
            text_size = cv2.getTextSize(
                warning, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2
            )[0]
            warning_x = max(8, (left_width - text_size[0]) // 2)
            cv2.rectangle(
                raw_frame,
                (warning_x - 8, 40),
                (warning_x + text_size[0] + 8, 72),
                (0, 0, 160),
                -1,
            )
            _draw_text(
                raw_frame,
                warning,
                (warning_x, 64),
                scale=0.72,
                color=(255, 255, 255),
                thickness=2,
            )
            if previous_fragment != fragment:
                cv2.rectangle(
                    raw_frame,
                    (2, 2),
                    (left_width - 3, 637),
                    FRAGMENT_COLORS[fragment],
                    8,
                )
                _draw_text(
                    raw_frame,
                    f"FRAGMENT {chr(ord('A') + fragment)} START",
                    (left_width // 2 - 150, 620),
                    scale=0.8,
                    color=FRAGMENT_COLORS[fragment],
                    thickness=2,
                )
            panel = render_position_panel(
                aggregate["joint_probabilities"][index],
                aggregate["truth_xy_cm"][index],
                gt_available,
                probability_maximum,
            )
            canvas = np.zeros((640, output_width, 3), dtype=np.uint8)
            canvas[:, :left_width] = raw_frame
            canvas[:, left_width : left_width + 640] = panel
            writer.write(canvas)
            previous_fragment = fragment
    finally:
        writer.release()
        capture.release()
