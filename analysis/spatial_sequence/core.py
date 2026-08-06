from __future__ import annotations

import csv
import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = (
    REPOSITORY_ROOT
    / "experiments"
    / "long-sequence"
    / "2026-07-30_batch-01"
)
AUGUST_EXPERIMENT_ROOT = (
    REPOSITORY_ROOT
    / "experiments"
    / "long-sequence"
    / "2026-08-03_batch-01"
)
AUGUST_SESSION_IDS = (
    "20260803_144217",
    "20260803_150952",
    "20260803_154420",
    "20260803_160842",
    "20260803_164023",
    "20260803_170357",
)
CAMERA_INTRINSICS_PATH = REPOSITORY_ROOT / "calibration" / "data" / "camera_intrinsics.json"
ANNOTATION_FILENAME = "spatial_annotation.json"
REFERENCE_SEQUENCE_LENGTH = 24
# Backward-compatible name for the canonical protocol. Individual runs carry
# their actual length in PreparedData/RunConfig.
SEQUENCE_LENGTH = REFERENCE_SEQUENCE_LENGTH
GUARD_LENGTH = REFERENCE_SEQUENCE_LENGTH - 1
PAPER_SIZE_CM = 26.5
PAPER_HALF_CM = PAPER_SIZE_CM / 2.0
POSITION_INTERIOR_BINS = 13
POSITION_CLASS_COUNT = POSITION_INTERIOR_BINS * POSITION_INTERIOR_BINS
CAUSAL_ANCHOR_INDEX = REFERENCE_SEQUENCE_LENGTH - 1
BIDIRECTIONAL_ANCHOR_INDEX = REFERENCE_SEQUENCE_LENGTH // 2
EVALUATION_SCHEMES = ("within-session", "leave-one-session-out")
AREA_CLASS_NAMES = ("none", "mint", "lavender")
SOURCE_NAMES = ("mint", "lavender")
SENSOR_COLUMNS = tuple(f"pcnose_S{index}_kohm" for index in range(1, 33))
MIN_STANDARD_DEVIATION = 1e-8
VELOCITY_RADIUS_S = 1.5
VELOCITY_MIN_POINTS = 4
VELOCITY_MIN_SPAN_S = 1.5


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    trial_id: str
    slug: str
    layout: str
    session_directory: Path
    csv_path: Path
    video_path: Path
    annotation_path: Path
    source_names: tuple[str, ...]


@dataclass(frozen=True)
class SpatialAnnotation:
    schema_version: int
    session_id: str
    trial_id: str
    usable_start_row: int
    usable_end_row: int
    desk_to_paper: np.ndarray
    paper_to_desk: np.ndarray
    paper_corners_desk_cm: np.ndarray
    source_polygons_paper_cm: dict[str, np.ndarray]
    source_hashes: dict[str, str]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class SplitBlock:
    split: str
    fragment: int
    start: int
    stop: int


@dataclass(frozen=True)
class WindowRecord:
    session_index: int
    start: int
    split: str
    fragment: int


@dataclass(frozen=True)
class AnchorRecord:
    session_index: int
    target_index: int
    window_start: int
    target_offset: int
    fragment: int


@dataclass
class SessionData:
    info: SessionInfo
    frame: pd.DataFrame
    raw_row_indices: np.ndarray
    timestamps_s: np.ndarray
    sensors: np.ndarray
    pose_mask: np.ndarray
    paper_xy_cm: np.ndarray
    position_classes: np.ndarray
    position_mask: np.ndarray
    distance_cm: np.ndarray
    distance_mask: np.ndarray
    area_classes: np.ndarray
    height_cm: np.ndarray
    velocity_cm_s: np.ndarray
    velocity_mask: np.ndarray
    blocks: list[SplitBlock]


@dataclass(frozen=True)
class NormalizationStats:
    input_mean: np.ndarray
    input_std: np.ndarray
    target_mean: np.ndarray | None
    target_std: np.ndarray | None
    target_transform: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_mean": self.input_mean.tolist(),
            "input_std": self.input_std.tolist(),
            "target_mean": None if self.target_mean is None else self.target_mean.tolist(),
            "target_std": None if self.target_std is None else self.target_std.tolist(),
            "target_transform": self.target_transform,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NormalizationStats":
        return cls(
            input_mean=np.asarray(value["input_mean"], dtype=np.float32),
            input_std=np.asarray(value["input_std"], dtype=np.float32),
            target_mean=(
                None
                if value.get("target_mean") is None
                else np.asarray(value["target_mean"], dtype=np.float32)
            ),
            target_std=(
                None
                if value.get("target_std") is None
                else np.asarray(value["target_std"], dtype=np.float32)
            ),
            target_transform=value.get("target_transform"),
        )


@dataclass
class PreparedData:
    sessions: list[SessionData]
    train_windows: list[WindowRecord]
    validation_windows: list[WindowRecord]
    stats: NormalizationStats
    target: str
    source_hashes: dict[str, str]
    sequence_length: int = REFERENCE_SEQUENCE_LENGTH
    evaluation_scheme: str = "within-session"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _source_names(layout: str) -> tuple[str, ...]:
    normalized = layout.lower()
    names = tuple(name for name in SOURCE_NAMES if name in normalized)
    if not names:
        raise ValueError(f"Cannot infer odor sources from layout: {layout!r}")
    return names


def _session_slug(trial_id: str) -> str:
    prefix, separator, remainder = trial_id.partition("_")
    if not separator or not prefix.startswith("2026-"):
        raise ValueError(f"Cannot derive session slug from trial_id: {trial_id!r}")
    normalized = remainder.replace("_random_10min_", "_")
    return normalized.replace("_", "-")


def discover_sessions(experiment_root: Path = EXPERIMENT_ROOT) -> list[SessionInfo]:
    manifest_path = experiment_root / "manifest.csv"
    sessions: list[SessionInfo] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["analysis_status"] != "usable":
                continue
            session_id = row["session"]
            session_directory = (
                experiment_root
                / "runs"
                / f"cyranose_reading_pose_session_{session_id}"
            )
            if not session_directory.is_dir():
                raise FileNotFoundError(
                    f"Expected session directory for {session_id}: {session_directory}"
                )
            video_path = session_directory / "rectified_rgb.mp4"
            csv_path = session_directory / "cyranose_reading_pose.csv"
            if not csv_path.is_file() or not video_path.is_file():
                raise FileNotFoundError(f"Incomplete session directory: {session_directory}")
            sessions.append(
                SessionInfo(
                    session_id=session_id,
                    trial_id=row["trial_id"],
                    slug=(row.get("slug") or "").strip()
                    or _session_slug(row["trial_id"]),
                    layout=row["layout"],
                    session_directory=session_directory,
                    csv_path=csv_path,
                    video_path=video_path,
                    annotation_path=session_directory / ANNOTATION_FILENAME,
                    source_names=_source_names(row["layout"]),
                )
            )
    if len(sessions) != 5:
        raise ValueError(f"Expected five usable sessions, found {len(sessions)}")
    return sessions


def discover_august_sessions(
    experiment_root: Path = AUGUST_EXPERIMENT_ROOT,
) -> list[SessionInfo]:
    """Discover the six 2026-08-03 sessions from their per-run metadata.

    Unlike the July batch, this batch has no manifest.  Requiring the known set
    of session IDs prevents a partially copied batch from looking complete in
    the annotation notebook.
    """
    run_root = experiment_root / "runs"
    sessions: list[SessionInfo] = []
    prefix = "cyranose_reading_pose_session_"
    for session_id in AUGUST_SESSION_IDS:
        session_directory = run_root / f"{prefix}{session_id}"
        metadata_path = session_directory / "session_metadata.json"
        csv_path = session_directory / "cyranose_reading_pose.csv"
        video_path = session_directory / "rectified_rgb.mp4"
        missing = [
            path.name
            for path in (metadata_path, csv_path, video_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Incomplete August session {session_id}: missing {', '.join(missing)}"
            )
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        recorded_id = str(metadata.get("session_id", ""))
        if recorded_id not in {session_id, f"{prefix}{session_id}"}:
            raise ValueError(
                f"Session identity mismatch in {metadata_path}: {recorded_id!r}"
            )
        trial_id = str(metadata["trial_id"])
        layout = str(metadata["trial_label"]).replace("_", " ")
        sessions.append(
            SessionInfo(
                session_id=session_id,
                trial_id=trial_id,
                slug=_session_slug(trial_id),
                layout=layout,
                session_directory=session_directory,
                csv_path=csv_path,
                video_path=video_path,
                annotation_path=session_directory / ANNOTATION_FILENAME,
                source_names=_source_names(layout),
            )
        )
    return sessions


def resolve_session(value: str, sessions: Sequence[SessionInfo]) -> SessionInfo:
    candidates = [
        session
        for session in sessions
        if value
        in {
            session.session_id,
            session.trial_id,
            session.slug,
            session.session_directory.name,
        }
    ]
    if len(candidates) != 1:
        choices = ", ".join(session.session_id for session in sessions)
        raise ValueError(f"Unknown or ambiguous session {value!r}; choose one of: {choices}")
    return candidates[0]


def load_camera_matrix(path: Path = CAMERA_INTRINSICS_PATH) -> np.ndarray:
    with path.open(encoding="utf-8") as handle:
        record = json.load(handle)
    matrix = np.asarray(record["rectified_K"], dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"Invalid rectified_K in {path}")
    return matrix


def transform_from_rvec_tvec(rvec: Sequence[float], tvec: Sequence[float]) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
    return inverse


def apply_transform_2d(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    mapped = homogeneous @ np.asarray(transform, dtype=np.float64).T
    return mapped[:, :2] / mapped[:, 2:3]


def fit_similarity_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target, dtype=np.float64).reshape(-1, 2)
    if source.shape != target.shape or len(source) < 2:
        raise ValueError("Matching 2D point sets are required")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (target_centered.T @ source_centered) / len(source)
    left, singular, right_t = np.linalg.svd(covariance)
    correction = np.eye(2)
    if np.linalg.det(left @ right_t) < 0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_t
    variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if variance <= 0.0:
        raise ValueError("Source points have zero variance")
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    transform = np.eye(3, dtype=np.float64)
    transform[:2, :2] = scale * rotation
    transform[:2, 2] = translation
    inverse = np.linalg.inv(transform)
    return transform, inverse


def canonical_paper_corners() -> np.ndarray:
    return np.array(
        [
            [-PAPER_HALF_CM, -PAPER_HALF_CM],
            [PAPER_HALF_CM, -PAPER_HALF_CM],
            [PAPER_HALF_CM, PAPER_HALF_CM],
            [-PAPER_HALF_CM, PAPER_HALF_CM],
        ],
        dtype=np.float64,
    )


def polygon_signed_area(polygon: np.ndarray) -> float:
    polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    return 0.5 * float(
        np.sum(
            polygon[:, 0] * np.roll(polygon[:, 1], -1)
            - np.roll(polygon[:, 0], -1) * polygon[:, 1]
        )
    )


def validate_quadrilateral(polygon: np.ndarray, name: str = "polygon") -> np.ndarray:
    polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if polygon.shape != (4, 2) or not np.all(np.isfinite(polygon)):
        raise ValueError(f"{name} must contain four finite 2D points")
    cross_products = []
    for index in range(4):
        first = polygon[(index + 1) % 4] - polygon[index]
        second = polygon[(index + 2) % 4] - polygon[(index + 1) % 4]
        cross_products.append(float(np.cross(first, second)))
    cross_products_array = np.asarray(cross_products)
    if np.any(np.abs(cross_products_array) < 1e-9) or not (
        np.all(cross_products_array > 0) or np.all(cross_products_array < 0)
    ):
        raise ValueError(f"{name} must be a non-self-intersecting convex quadrilateral")
    if abs(polygon_signed_area(polygon)) < 1e-6:
        raise ValueError(f"{name} has negligible area")
    return polygon


def points_in_polygon(points: np.ndarray, polygon: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    polygon = validate_quadrilateral(polygon)
    inside = np.ones(len(points), dtype=bool)
    area_sign = 1.0 if polygon_signed_area(polygon) > 0 else -1.0
    for start, stop in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = stop - start
        relative = points - start
        cross = edge[0] * relative[:, 1] - edge[1] * relative[:, 0]
        inside &= area_sign * cross >= -tolerance
    return inside


def distances_to_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    polygon = validate_quadrilateral(polygon)
    distances = np.full(len(points), np.inf, dtype=np.float64)
    for start, stop in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = stop - start
        denominator = float(np.dot(edge, edge))
        fraction = np.clip(((points - start) @ edge) / denominator, 0.0, 1.0)
        closest = start + fraction[:, None] * edge
        distances = np.minimum(distances, np.linalg.norm(points - closest, axis=1))
    distances[points_in_polygon(points, polygon)] = 0.0
    return distances


def paper_contains(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return np.all(np.abs(points) <= PAPER_HALF_CM + 1e-9, axis=1)


def position_bin_indices(values: np.ndarray) -> np.ndarray:
    """Return paper-only ``(x, y)`` bins, with ``-1`` outside the paper."""
    values = np.asarray(values, dtype=np.float64).reshape(-1, 2)
    result = np.full((len(values), 2), -1, dtype=np.int64)
    inside = paper_contains(values)
    scaled = (values[inside] + PAPER_HALF_CM) / PAPER_SIZE_CM
    bins = np.floor(scaled * POSITION_INTERIOR_BINS).astype(np.int64)
    result[inside] = np.clip(bins, 0, POSITION_INTERIOR_BINS - 1)
    return result


def position_classes(values: np.ndarray) -> np.ndarray:
    """Return the y-major 13x13 joint class, or ``-1`` outside paper."""
    bins = position_bin_indices(values)
    result = np.full(len(bins), -1, dtype=np.int64)
    valid = np.all(bins >= 0, axis=1)
    result[valid] = bins[valid, 1] * POSITION_INTERIOR_BINS + bins[valid, 0]
    return result


def position_bin_centers() -> np.ndarray:
    width = PAPER_SIZE_CM / POSITION_INTERIOR_BINS
    return -PAPER_HALF_CM + (np.arange(POSITION_INTERIOR_BINS) + 0.5) * width


def reconstruct_camera_to_desk(row: pd.Series) -> np.ndarray:
    required = [
        "tag_6_cam_rvec_x_rad",
        "tag_6_cam_rvec_y_rad",
        "tag_6_cam_rvec_z_rad",
        "tag_6_cam_x_cm",
        "tag_6_cam_y_cm",
        "tag_6_cam_z_cm",
    ]
    values = np.asarray([row[column] for column in required], dtype=np.float64)
    if not bool(row["tag_6_visible"]) or not np.all(np.isfinite(values)):
        raise ValueError("Desk tag 6 is unavailable for this row")
    return transform_from_rvec_tvec(values[:3], values[3:])


def pixel_to_desk(pixel: Sequence[float], camera_matrix: np.ndarray, camera_to_desk: np.ndarray) -> np.ndarray:
    pixel = np.asarray(pixel, dtype=np.float64).reshape(2)
    direction_camera = np.array(
        [
            (pixel[0] - camera_matrix[0, 2]) / camera_matrix[0, 0],
            (pixel[1] - camera_matrix[1, 2]) / camera_matrix[1, 1],
            1.0,
        ],
        dtype=np.float64,
    )
    desk_to_camera = invert_transform(camera_to_desk)
    origin = desk_to_camera[:3, 3]
    direction = desk_to_camera[:3, :3] @ direction_camera
    if abs(direction[2]) < 1e-12:
        raise ValueError("Pixel ray is parallel to the desk plane")
    distance = -origin[2] / direction[2]
    if distance <= 0:
        raise ValueError("Pixel ray intersects the desk plane behind the camera")
    point = origin + distance * direction
    return point[:2]


def project_desk_points(points_xy: np.ndarray, camera_matrix: np.ndarray, camera_to_desk: np.ndarray) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    points = np.column_stack([points_xy, np.zeros(len(points_xy), dtype=np.float64)])
    camera = points @ camera_to_desk[:3, :3].T + camera_to_desk[:3, 3]
    if np.any(camera[:, 2] <= 0):
        raise ValueError("Desk points project behind the camera")
    return np.column_stack(
        [
            camera_matrix[0, 0] * camera[:, 0] / camera[:, 2] + camera_matrix[0, 2],
            camera_matrix[1, 1] * camera[:, 1] / camera[:, 2] + camera_matrix[1, 2],
        ]
    )


def write_annotation(
    info: SessionInfo,
    usable_start_row: int,
    usable_end_row: int,
    paper_corners_desk_cm: np.ndarray,
    source_polygons_desk_cm: dict[str, np.ndarray],
    diagnostics: dict[str, Any] | None = None,
) -> SpatialAnnotation:
    paper_corners = validate_quadrilateral(paper_corners_desk_cm, "paper corners")
    desk_to_paper, paper_to_desk = fit_similarity_transform(
        paper_corners,
        canonical_paper_corners(),
    )
    sources_paper: dict[str, np.ndarray] = {}
    sources_desk: dict[str, list[list[float]]] = {}
    for name in info.source_names:
        if name not in source_polygons_desk_cm:
            raise ValueError(f"Missing {name} source polygon")
        polygon_desk = validate_quadrilateral(source_polygons_desk_cm[name], f"{name} source")
        polygon_paper = validate_quadrilateral(
            apply_transform_2d(desk_to_paper, polygon_desk),
            f"{name} source in paper coordinates",
        )
        sources_paper[name] = polygon_paper
        sources_desk[name] = polygon_desk.tolist()
    frame = pd.read_csv(info.csv_path, usecols=["pcnose_host_time_utc"])
    if not (0 <= usable_start_row <= usable_end_row < len(frame)):
        raise ValueError("Usable row range falls outside the source CSV")
    fitted = apply_transform_2d(desk_to_paper, paper_corners)
    residuals = np.linalg.norm(fitted - canonical_paper_corners(), axis=1)
    record = {
        "schema_version": 1,
        "session_id": info.session_id,
        "trial_id": info.trial_id,
        "coordinate_convention": {
            "units": "cm",
            "origin": "paper_center",
            "x_positive": "right_in_calibrated_desk_view",
            "y_positive": "down_in_calibrated_desk_view",
            "paper_size_cm": PAPER_SIZE_CM,
        },
        "usable_range": {
            "start_raw_row_inclusive": int(usable_start_row),
            "end_raw_row_inclusive": int(usable_end_row),
            "start_time_utc": str(frame.iloc[usable_start_row, 0]),
            "end_time_utc": str(frame.iloc[usable_end_row, 0]),
        },
        "paper": {
            "corners_desk_cm": paper_corners.tolist(),
            "canonical_corners_paper_cm": canonical_paper_corners().tolist(),
            "desk_to_paper": desk_to_paper.tolist(),
            "paper_to_desk": paper_to_desk.tolist(),
            "fit_residuals_cm": residuals.tolist(),
            "fit_rms_cm": float(np.sqrt(np.mean(residuals**2))),
        },
        "sources": {
            name: {
                "corners_desk_cm": sources_desk[name],
                "polygon_paper_cm": sources_paper[name].tolist(),
            }
            for name in info.source_names
        },
        "source_hashes": {
            "cyranose_reading_pose.csv": sha256_file(info.csv_path),
            "rectified_rgb.mp4": sha256_file(info.video_path),
        },
        "diagnostics": diagnostics or {},
    }
    info.annotation_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return load_annotation(info)


def load_annotation(info: SessionInfo, verify_hashes: bool = True) -> SpatialAnnotation:
    if not info.annotation_path.is_file():
        raise FileNotFoundError(
            f"Missing annotation for session {info.session_id}: {info.annotation_path}"
        )
    with info.annotation_path.open(encoding="utf-8") as handle:
        record = json.load(handle)
    if record.get("schema_version") != 1:
        raise ValueError(f"Unsupported annotation schema in {info.annotation_path}")
    if record.get("session_id") != info.session_id or record.get("trial_id") != info.trial_id:
        raise ValueError(f"Annotation identity mismatch in {info.annotation_path}")
    usable = record["usable_range"]
    paper = record["paper"]
    desk_to_paper = np.asarray(paper["desk_to_paper"], dtype=np.float64)
    paper_to_desk = np.asarray(paper["paper_to_desk"], dtype=np.float64)
    if desk_to_paper.shape != (3, 3) or paper_to_desk.shape != (3, 3):
        raise ValueError("Annotation transforms must be 3x3")
    sources: dict[str, np.ndarray] = {}
    for name in info.source_names:
        if name not in record["sources"]:
            raise ValueError(f"Annotation is missing the {name} source")
        sources[name] = validate_quadrilateral(
            np.asarray(record["sources"][name]["polygon_paper_cm"], dtype=np.float64),
            f"{name} source",
        )
    hashes = {str(key): str(value) for key, value in record.get("source_hashes", {}).items()}
    if verify_hashes:
        for filename, path in (
            ("cyranose_reading_pose.csv", info.csv_path),
            ("rectified_rgb.mp4", info.video_path),
        ):
            expected = hashes.get(filename)
            if expected and sha256_file(path) != expected:
                warnings.warn(
                    f"{filename} hash changed after annotation for session {info.session_id}",
                    RuntimeWarning,
                    stacklevel=2,
                )
    return SpatialAnnotation(
        schema_version=1,
        session_id=info.session_id,
        trial_id=info.trial_id,
        usable_start_row=int(usable["start_raw_row_inclusive"]),
        usable_end_row=int(usable["end_raw_row_inclusive"]),
        desk_to_paper=desk_to_paper,
        paper_to_desk=paper_to_desk,
        paper_corners_desk_cm=validate_quadrilateral(
            np.asarray(paper["corners_desk_cm"], dtype=np.float64),
            "paper corners",
        ),
        source_polygons_paper_cm=sources,
        source_hashes=hashes,
        diagnostics=dict(record.get("diagnostics", {})),
    )


def balanced_lengths(total: int, block_count: int) -> list[int]:
    if total < 0 or block_count <= 0:
        raise ValueError("Invalid balanced partition")
    base, remainder = divmod(total, block_count)
    return [base + (1 if index < remainder else 0) for index in range(block_count)]


def make_split_blocks(row_count: int) -> list[SplitBlock]:
    """Build Train-Guard-Validation-Guard-Train blocks with a 4:1 ratio."""
    retained = row_count - 2 * GUARD_LENGTH
    if retained <= 0:
        raise ValueError("Not enough active rows after guard removal")
    validation_total = int(round(retained / 5.0))
    train_total = retained - validation_total
    train_lengths = balanced_lengths(train_total, 2)
    if min(validation_total, *train_lengths) < SEQUENCE_LENGTH:
        raise ValueError("At least one split block is shorter than the sequence length")
    first_train_stop = train_lengths[0]
    validation_start = first_train_stop + GUARD_LENGTH
    validation_stop = validation_start + validation_total
    second_train_start = validation_stop + GUARD_LENGTH
    blocks = [
        SplitBlock("train", 0, 0, first_train_stop),
        SplitBlock("validation", 0, validation_start, validation_stop),
        SplitBlock("train", 1, second_train_start, row_count),
    ]
    if blocks[-1].stop - blocks[-1].start != train_lengths[1]:
        raise AssertionError("Split did not consume the expected second train block")
    return blocks


def assign_evaluation_blocks(
    sessions: Sequence[SessionData],
    evaluation_scheme: str,
    held_out_session: str | None = None,
) -> None:
    """Assign train/validation rows for the requested evaluation scheme."""
    if evaluation_scheme not in EVALUATION_SCHEMES:
        raise ValueError(f"Unknown evaluation scheme: {evaluation_scheme}")
    if evaluation_scheme == "within-session":
        if held_out_session is not None:
            raise ValueError("Within-session evaluation does not accept a held-out session")
        return
    if held_out_session is None:
        raise ValueError("Leave-one-session-out evaluation requires a held-out session")
    held_out_info = resolve_session(held_out_session, [session.info for session in sessions])
    for session in sessions:
        split = "validation" if session.info.session_id == held_out_info.session_id else "train"
        session.blocks = [SplitBlock(split, 0, 0, len(session.sensors))]


def anchor_records_for_sessions(
    sessions: Sequence[SessionData],
    temporal_mode: str,
    sequence_length: int = REFERENCE_SEQUENCE_LENGTH,
    *,
    full_validation_block: bool = False,
) -> list[AnchorRecord]:
    if temporal_mode not in {"causal", "bidirectional"}:
        raise ValueError(f"Unknown temporal mode: {temporal_mode}")
    if not 2 <= sequence_length <= REFERENCE_SEQUENCE_LENGTH:
        raise ValueError(
            f"Sequence length must be between 2 and {REFERENCE_SEQUENCE_LENGTH}"
        )
    target_offset = sequence_length - 1 if temporal_mode == "causal" else sequence_length // 2
    records: list[AnchorRecord] = []
    for session_index, session in enumerate(sessions):
        validation_blocks = [block for block in session.blocks if block.split == "validation"]
        if full_validation_block and not validation_blocks:
            continue
        if len(validation_blocks) != 1:
            raise ValueError("Anchor validation requires exactly one validation block")
        block = validation_blocks[0]
        if full_validation_block:
            left_margin = target_offset
            right_margin = sequence_length - target_offset - 1
        else:
            # Sequence-length ablations retain the exact canonical physical
            # anchors, so the length-24 baseline is directly reusable.
            left_margin = CAUSAL_ANCHOR_INDEX
            right_margin = REFERENCE_SEQUENCE_LENGTH - BIDIRECTIONAL_ANCHOR_INDEX - 1
        for target_index in range(block.start + left_margin, block.stop - right_margin):
            window_start = target_index - target_offset
            if window_start < block.start or window_start + sequence_length > block.stop:
                raise AssertionError("Anchor window crosses the validation boundary")
            records.append(
                AnchorRecord(
                    session_index=session_index,
                    target_index=target_index,
                    window_start=window_start,
                    target_offset=target_offset,
                    fragment=block.fragment,
                )
            )
    return records


def windows_for_sessions(
    sessions: Sequence[SessionData],
    split: str,
    sequence_length: int = REFERENCE_SEQUENCE_LENGTH,
) -> list[WindowRecord]:
    if not 2 <= sequence_length <= REFERENCE_SEQUENCE_LENGTH:
        raise ValueError(
            f"Sequence length must be between 2 and {REFERENCE_SEQUENCE_LENGTH}"
        )
    windows: list[WindowRecord] = []
    for session_index, session in enumerate(sessions):
        for block in session.blocks:
            if block.split != split:
                continue
            for start in range(block.start, block.stop - sequence_length + 1):
                windows.append(WindowRecord(session_index, start, split, block.fragment))
    return windows


def _time_seconds(values: pd.Series) -> np.ndarray:
    timestamps = pd.to_datetime(values, utc=True, errors="raise", format="mixed")
    nanoseconds = timestamps.astype("int64").to_numpy(dtype=np.int64)
    return (nanoseconds - nanoseconds[0]).astype(np.float64) / 1e9


def estimate_velocity(
    timestamps_s: np.ndarray,
    positions_cm: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
    positions_cm = np.asarray(positions_cm, dtype=np.float64).reshape(-1, 2)
    valid = np.asarray(valid, dtype=bool)
    velocity = np.full_like(positions_cm, np.nan)
    mask = np.zeros_like(positions_cm, dtype=bool)
    for index, target_time in enumerate(timestamps_s):
        neighborhood = (
            valid
            & (timestamps_s >= target_time - VELOCITY_RADIUS_S)
            & (timestamps_s <= target_time + VELOCITY_RADIUS_S)
        )
        selected = np.flatnonzero(neighborhood)
        if len(selected) < VELOCITY_MIN_POINTS:
            continue
        selected_times = timestamps_s[selected]
        if not np.any(selected_times < target_time) or not np.any(selected_times > target_time):
            continue
        if selected_times.max() - selected_times.min() < VELOCITY_MIN_SPAN_S:
            continue
        centered = selected_times - target_time
        design = np.column_stack([np.ones(len(selected)), centered])
        coefficients, _, _, _ = np.linalg.lstsq(
            design,
            positions_cm[selected],
            rcond=None,
        )
        velocity[index] = coefficients[1]
        mask[index] = True
    return velocity, mask


def _build_session_data(
    info: SessionInfo,
    annotation: SpatialAnnotation,
) -> SessionData:
    frame = pd.read_csv(info.csv_path)
    if tuple(column for column in SENSOR_COLUMNS if column in frame.columns) != SENSOR_COLUMNS:
        raise ValueError(f"Session {info.session_id} does not contain S1 through S32 in order")
    start, stop = annotation.usable_start_row, annotation.usable_end_row
    if not (0 <= start <= stop < len(frame)):
        raise ValueError(f"Invalid usable range for session {info.session_id}")
    raw_indices = np.arange(start, stop + 1, dtype=np.int64)
    active_mask = frame.loc[start:stop, "pcnose_flag"].to_numpy() == 2
    raw_indices = raw_indices[active_mask]
    active = frame.iloc[raw_indices].copy()
    sensors = active.loc[:, SENSOR_COLUMNS].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(sensors)):
        bad = np.argwhere(~np.isfinite(sensors))[0]
        raise ValueError(
            f"Non-finite sensor value in session {info.session_id}, active row {bad[0]}, "
            f"channel {SENSOR_COLUMNS[int(bad[1])]}"
        )
    desk_xyz = active.loc[
        :, ["snout_desk_x_cm", "snout_desk_y_cm", "snout_desk_z_cm"]
    ].to_numpy(dtype=np.float64)
    pose_mask = active["snout_position_valid"].astype(bool).to_numpy() & np.all(
        np.isfinite(desk_xyz), axis=1
    )
    paper_xy = np.full((len(active), 2), np.nan, dtype=np.float64)
    paper_xy[pose_mask] = apply_transform_2d(
        annotation.desk_to_paper,
        desk_xyz[pose_mask, :2],
    )
    classes = np.full(len(active), -1, dtype=np.int64)
    classes[pose_mask] = position_classes(paper_xy[pose_mask])
    position_mask = pose_mask & (classes >= 0)

    distance = np.full((len(active), 2), np.nan, dtype=np.float64)
    distance_mask = np.zeros((len(active), 2), dtype=bool)
    for source_index, source_name in enumerate(SOURCE_NAMES):
        polygon = annotation.source_polygons_paper_cm.get(source_name)
        if polygon is None:
            continue
        distance[pose_mask, source_index] = distances_to_polygon(
            paper_xy[pose_mask], polygon
        )
        distance_mask[pose_mask, source_index] = True

    area = np.full(len(active), -1, dtype=np.int64)
    area[pose_mask] = 0
    on_paper = np.zeros(len(active), dtype=bool)
    on_paper[pose_mask] = paper_contains(paper_xy[pose_mask])
    membership_count = np.zeros(len(active), dtype=np.int64)
    for source_name, class_id in (("mint", 1), ("lavender", 2)):
        polygon = annotation.source_polygons_paper_cm.get(source_name)
        if polygon is None:
            continue
        inside = np.zeros(len(active), dtype=bool)
        inside[pose_mask] = points_in_polygon(paper_xy[pose_mask], polygon)
        inside &= on_paper
        membership_count += inside.astype(np.int64)
        area[inside] = class_id
    if np.any(membership_count > 1):
        raise ValueError(f"Source polygons overlap for session {info.session_id}")

    height = np.full(len(active), np.nan, dtype=np.float64)
    height[pose_mask] = -desk_xyz[pose_mask, 2]
    if np.any(height[pose_mask] < 0):
        raise ValueError(f"Negative physical height in session {info.session_id}")
    timestamps_s = _time_seconds(active["pcnose_host_time_utc"])
    velocity, velocity_mask = estimate_velocity(timestamps_s, paper_xy, pose_mask)
    blocks = make_split_blocks(len(active))
    return SessionData(
        info=info,
        frame=active,
        raw_row_indices=raw_indices,
        timestamps_s=timestamps_s,
        sensors=sensors.astype(np.float32),
        pose_mask=pose_mask,
        paper_xy_cm=paper_xy.astype(np.float32),
        position_classes=classes,
        position_mask=position_mask,
        distance_cm=distance.astype(np.float32),
        distance_mask=distance_mask,
        area_classes=area,
        height_cm=height.astype(np.float32),
        velocity_cm_s=velocity.astype(np.float32),
        velocity_mask=velocity_mask,
        blocks=blocks,
    )


def training_row_mask(session: SessionData) -> np.ndarray:
    """Return the unique physical rows assigned to training in one session."""
    mask = np.zeros(len(session.sensors), dtype=bool)
    for block in session.blocks:
        if block.split != "train":
            continue
        if np.any(mask[block.start : block.stop]):
            raise ValueError("Training blocks overlap")
        mask[block.start : block.stop] = True
    return mask


def _checked_std(values: np.ndarray, names: Sequence[str]) -> np.ndarray:
    standard_deviation = np.std(values, axis=0, ddof=0)
    standard_deviation = np.atleast_1d(standard_deviation).astype(np.float64)
    for name, value in zip(names, standard_deviation):
        if not np.isfinite(value) or value < MIN_STANDARD_DEVIATION:
            raise ValueError(f"Training standard deviation is too small for {name}: {value}")
    return standard_deviation


def compute_normalization_stats(
    sessions: Sequence[SessionData],
    train_windows: Sequence[WindowRecord] | None,
    target: str,
) -> NormalizationStats:
    # ``train_windows`` remains in the public signature for call-site
    # compatibility, but normalization deliberately ignores overlapping
    # occurrences. Every physical training row contributes exactly once.
    del train_windows
    row_masks = [training_row_mask(session) for session in sessions]
    if not any(np.any(mask) for mask in row_masks):
        raise ValueError("Normalization contains no training rows")
    input_values = np.concatenate(
        [session.sensors[mask] for session, mask in zip(sessions, row_masks) if np.any(mask)],
        axis=0,
    ).astype(np.float64)
    input_mean = input_values.mean(axis=0)
    input_std = _checked_std(input_values, SENSOR_COLUMNS)
    target_mean: np.ndarray | None = None
    target_std: np.ndarray | None = None
    target_transform: str | None = None
    if target == "distance":
        columns: list[np.ndarray] = []
        for source_index, source_name in enumerate(SOURCE_NAMES):
            values = []
            for session, training_mask in zip(sessions, row_masks):
                mask = training_mask & session.distance_mask[:, source_index]
                if np.any(mask):
                    values.append(session.distance_cm[mask, source_index])
            if not values:
                raise ValueError(f"Training contains no labels for {source_name} distance")
            combined = np.concatenate(values).astype(np.float64)
            columns.append(combined)
        target_mean = np.zeros(2, dtype=np.float64)
        target_std = np.array(
            [
                _checked_std(values[:, None], [name])[0]
                for values, name in zip(columns, SOURCE_NAMES)
            ]
        )
        target_transform = "std_only"
    elif target == "height":
        values = []
        for session, training_mask in zip(sessions, row_masks):
            mask = training_mask & session.pose_mask
            if np.any(mask):
                values.append(np.log1p(session.height_cm[mask]))
        if not values:
            raise ValueError("Training contains no height labels")
        combined = np.concatenate(values).astype(np.float64)
        target_mean = np.zeros(1, dtype=np.float64)
        target_std = _checked_std(combined[:, None], ["log1p_height"])
        target_transform = "log1p_std_only"
    elif target == "velocity":
        columns: list[list[np.ndarray]] = [[], []]
        for session, training_mask in zip(sessions, row_masks):
            current = session.velocity_cm_s
            mask = session.velocity_mask & training_mask[:, None]
            for component in range(2):
                if np.any(mask[:, component]):
                    columns[component].append(
                        current[:, component][mask[:, component]]
                    )
        combined = [
            np.concatenate(values).astype(np.float64)
            for values in columns
        ]
        for component in range(2):
            if not np.all(np.isfinite(combined[component])):
                raise ValueError("Velocity labels contain non-finite values")
        target_mean = np.array([values.mean() for values in combined])
        target_std = np.array(
            [
                _checked_std(values[:, None], [name])[0]
                for values, name in zip(combined, ("vx", "vy"))
            ]
        )
        target_transform = "zscore"
    return NormalizationStats(
        input_mean=input_mean.astype(np.float32),
        input_std=input_std.astype(np.float32),
        target_mean=None if target_mean is None else target_mean.astype(np.float32),
        target_std=None if target_std is None else target_std.astype(np.float32),
        target_transform=target_transform,
    )


def build_prepared_data(
    target: str,
    session: str | None = None,
    experiment_root: Path = EXPERIMENT_ROOT,
    sequence_length: int = REFERENCE_SEQUENCE_LENGTH,
    evaluation_scheme: str = "within-session",
    held_out_session: str | None = None,
) -> PreparedData:
    if target not in {"position", "distance", "area", "height", "velocity"}:
        raise ValueError(f"Unknown target: {target}")
    if not 2 <= sequence_length <= REFERENCE_SEQUENCE_LENGTH:
        raise ValueError(
            f"Sequence length must be between 2 and {REFERENCE_SEQUENCE_LENGTH}"
        )
    if evaluation_scheme not in EVALUATION_SCHEMES:
        raise ValueError(f"Unknown evaluation scheme: {evaluation_scheme}")
    infos = discover_sessions(experiment_root)
    if target == "position":
        if evaluation_scheme != "within-session":
            raise ValueError("Position supports only within-session evaluation")
        if not session:
            raise ValueError("Position training requires --session")
        infos = [resolve_session(session, infos)]
    elif session is not None:
        raise ValueError("--session is valid only for the position target")
    annotations = [load_annotation(info) for info in infos]
    sessions = [
        _build_session_data(info, annotation)
        for info, annotation in zip(infos, annotations)
    ]
    assign_evaluation_blocks(sessions, evaluation_scheme, held_out_session)
    train_windows = windows_for_sessions(sessions, "train", sequence_length)
    validation_windows = windows_for_sessions(sessions, "validation", sequence_length)
    if not train_windows or not validation_windows:
        raise ValueError("No legal train or validation windows")
    stats = compute_normalization_stats(sessions, train_windows, target)
    resolved_experiment_root = experiment_root.resolve()
    hashes = {
        str(info.csv_path.resolve().relative_to(resolved_experiment_root)): sha256_file(info.csv_path)
        for info in infos
    }
    hashes.update(
        {
            str(info.annotation_path.resolve().relative_to(resolved_experiment_root)): (
                sha256_file(info.annotation_path)
            )
            for info in infos
        }
    )
    hashes.update(
        {
            str(info.video_path.resolve().relative_to(resolved_experiment_root)): (
                sha256_file(info.video_path)
            )
            for info in infos
        }
    )
    return PreparedData(
        sessions=sessions,
        train_windows=train_windows,
        validation_windows=validation_windows,
        stats=stats,
        target=target,
        source_hashes=hashes,
        sequence_length=sequence_length,
        evaluation_scheme=evaluation_scheme,
    )


class SpatialWindowDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        prepared: PreparedData,
        split: str,
        stats: NormalizationStats | None = None,
    ) -> None:
        self.prepared = prepared
        self.windows = (
            prepared.train_windows if split == "train" else prepared.validation_windows
        )
        self.stats = stats or prepared.stats
        self.target = prepared.target

    def __len__(self) -> int:
        return len(self.windows)

    def _scaled_target(
        self, session: SessionData, selection: slice | int
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.target == "position":
            target = session.position_classes[selection]
            mask = session.position_mask[selection]
        elif self.target == "area":
            target = session.area_classes[selection]
            mask = session.pose_mask[selection]
        elif self.target == "distance":
            assert self.stats.target_std is not None
            target = session.distance_cm[selection] / self.stats.target_std
            mask = session.distance_mask[selection]
        elif self.target == "height":
            assert self.stats.target_std is not None
            selected_height = np.asarray(session.height_cm[selection])
            selected_pose = np.asarray(session.pose_mask[selection])
            target = np.log1p(selected_height)[..., None] / self.stats.target_std
            mask = selected_pose[..., None]
        elif self.target == "velocity":
            assert self.stats.target_mean is not None and self.stats.target_std is not None
            target = (
                session.velocity_cm_s[selection] - self.stats.target_mean
            ) / self.stats.target_std
            mask = session.velocity_mask[selection]
        else:
            raise AssertionError(self.target)
        return target, mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        session = self.prepared.sessions[window.session_index]
        selection = slice(window.start, window.start + self.prepared.sequence_length)
        inputs = (
            session.sensors[selection] - self.stats.input_mean
        ) / self.stats.input_std
        target, mask = self._scaled_target(session, selection)
        return {
            "inputs": torch.as_tensor(inputs, dtype=torch.float32),
            "target": torch.as_tensor(target),
            "mask": torch.as_tensor(mask, dtype=torch.bool),
            "session_index": torch.tensor(window.session_index, dtype=torch.int64),
            "raw_row_indices": torch.as_tensor(
                session.raw_row_indices[selection], dtype=torch.int64
            ),
            "paper_xy_cm": torch.as_tensor(
                session.paper_xy_cm[selection], dtype=torch.float32
            ),
            "fragment": torch.tensor(window.fragment, dtype=torch.int64),
        }


class SpatialAnchorDataset(Dataset[dict[str, Any]]):
    """One full-context validation prediction for each shared physical target row."""

    def __init__(
        self,
        prepared: PreparedData,
        temporal_mode: str,
        stats: NormalizationStats | None = None,
    ) -> None:
        self.prepared = prepared
        self.records = anchor_records_for_sessions(
            prepared.sessions,
            temporal_mode,
            prepared.sequence_length,
            full_validation_block=prepared.evaluation_scheme == "leave-one-session-out",
        )
        self.stats = stats or prepared.stats
        self.target = prepared.target
        self._target_helper = SpatialWindowDataset(prepared, "validation", self.stats)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        session = self.prepared.sessions[record.session_index]
        selection = slice(
            record.window_start,
            record.window_start + self.prepared.sequence_length,
        )
        inputs = (
            session.sensors[selection] - self.stats.input_mean
        ) / self.stats.input_std
        target, mask = self._target_helper._scaled_target(
            session,
            record.target_index,
        )
        return {
            "inputs": torch.as_tensor(inputs, dtype=torch.float32),
            "target": torch.as_tensor(target),
            "mask": torch.as_tensor(mask, dtype=torch.bool),
            "target_offset": torch.tensor(record.target_offset, dtype=torch.int64),
            "session_index": torch.tensor(record.session_index, dtype=torch.int64),
            "raw_row_index": torch.tensor(
                session.raw_row_indices[record.target_index], dtype=torch.int64
            ),
            "paper_xy_cm": torch.as_tensor(
                session.paper_xy_cm[record.target_index], dtype=torch.float32
            ),
            "fragment": torch.tensor(record.fragment, dtype=torch.int64),
        }


def inverse_regression_target(
    target: str,
    values: np.ndarray,
    stats: NormalizationStats,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if target == "distance":
        assert stats.target_std is not None
        return values * stats.target_std
    if target == "height":
        assert stats.target_std is not None
        return np.expm1(values * stats.target_std)
    if target == "velocity":
        assert stats.target_mean is not None and stats.target_std is not None
        return values * stats.target_std + stats.target_mean
    raise ValueError(f"Target {target} is not a regression target")
