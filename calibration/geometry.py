from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


TAG_SIZE_MM = 77.0
CUBE_SIDE_MM = 101.0
REQUIRED_FACE_IDS = (0, 1, 2, 3, 4)
OPTIONAL_FACE_ID = 5
DESK_TAG_ID = 6
REFERENCE_TAG_ID = 2

MIN_TAG_SIDE_PX = 50.0
MAX_TAG_RMS_PX = 2.0
NORMAL_TAG_COUNT = 60
NORMAL_EDGE_COUNT = 20
RELAXED_TAG_COUNT = 30
RELAXED_EDGE_COUNT = 10
MAX_BA_FRAMES = 500
MAX_SNOUT_CANDIDATES = 100
TARGET_SNOUT_CLICKS = 30


def _tag_object_points() -> np.ndarray:
    half = TAG_SIZE_MM / 2.0
    return np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        dtype=np.float64,
    )


TAG_OBJECT_POINTS = _tag_object_points()


@dataclass
class TagObservation:
    tag_id: int
    corners: np.ndarray
    average_side_px: float
    T_camera_tag: np.ndarray | None
    reprojection_rms_px: float
    qualified: bool


@dataclass
class FrameObservation:
    frame_index: int
    observations: list[TagObservation]
    source_path: str = ""
    source_frame_index: int | None = None


@dataclass
class SnoutCandidate:
    frame_index: int
    T_camera_cube: np.ndarray
    used_tag_ids: tuple[int, ...]
    reprojection_rms_px: float
    view_direction_cube: np.ndarray
    source_path: str = ""
    source_frame_index: int | None = None


_DETECTOR_PARAMETERS = cv2.aruco.DetectorParameters()
# OpenCV's AprilTag-specific refinement is materially more stable than the
# generic subpixel mode when two cube faces are viewed obliquely. The generic
# mode was observed to alternate between IDs on consecutive frames even while
# both full tags remained visible; this mode recovers simultaneous, qualified
# observations without loosening the size or reprojection-quality thresholds.
_DETECTOR_PARAMETERS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
_TAG_DETECTOR = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
    _DETECTOR_PARAMETERS,
)


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length == 0.0:
        raise ValueError("Cannot normalize a zero vector")
    return vector / length


def transform_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def transform_to_vector(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    return np.concatenate([rvec.reshape(3), transform[:3, 3]])


def vector_to_transform(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(6)
    return transform_from_rvec_tvec(vector[:3], vector[3:])


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    inverse = np.eye(4, dtype=np.float64)
    rotation = transform[:3, :3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ transform[:3, 3]
    return inverse


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return points @ transform[:3, :3].T + transform[:3, 3]


def project_points(
    points_object: np.ndarray,
    T_camera_object: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    points_camera = transform_points(T_camera_object, points_object)
    z = points_camera[:, 2]
    if np.any(z == 0.0):
        raise ValueError("A projected point has zero camera depth")
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    u = camera_matrix[0, 0] * points_camera[:, 0] / z + camera_matrix[0, 2]
    v = camera_matrix[1, 1] * points_camera[:, 1] / z + camera_matrix[1, 2]
    return np.column_stack([u, v])


def reprojection_rms(projected: np.ndarray, observed: np.ndarray) -> float:
    errors = np.asarray(projected, dtype=np.float64) - np.asarray(
        observed,
        dtype=np.float64,
    )
    return float(np.sqrt(np.mean(np.sum(errors * errors, axis=1))))


def average_tag_side(corners: np.ndarray) -> float:
    corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    sides = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
    return float(np.mean(sides))


def solve_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    success, rvec, tvec = cv2.solvePnP(
        np.asarray(object_points, dtype=np.float64),
        np.asarray(image_points, dtype=np.float64),
        np.asarray(camera_matrix, dtype=np.float64),
        np.zeros(5, dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None, float("inf")
    transform = transform_from_rvec_tvec(rvec, tvec)
    projected = project_points(object_points, transform, camera_matrix)
    return transform, reprojection_rms(projected, image_points)


def detect_tags(image: np.ndarray, camera_matrix: np.ndarray) -> list[TagObservation]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_corners, detected_ids, _ = _TAG_DETECTOR.detectMarkers(gray)
    if detected_ids is None:
        return []

    observations: list[TagObservation] = []
    for tag_id_value, corner_array in zip(detected_ids.reshape(-1), detected_corners):
        corners = np.asarray(corner_array, dtype=np.float64).reshape(4, 2)
        transform, rms = solve_pnp(TAG_OBJECT_POINTS, corners, camera_matrix)
        side_px = average_tag_side(corners)
        observations.append(
            TagObservation(
                tag_id=int(tag_id_value),
                corners=corners,
                average_side_px=side_px,
                T_camera_tag=transform,
                reprojection_rms_px=rms,
                qualified=(
                    transform is not None
                    and side_px >= MIN_TAG_SIDE_PX
                    and rms <= MAX_TAG_RMS_PX
                ),
            )
        )
    return observations


def transform_map_from_json(data: dict[str, Any]) -> dict[int, np.ndarray]:
    return {
        int(tag_id): np.asarray(transform, dtype=np.float64)
        for tag_id, transform in data["T_cube_tag"].items()
    }


def estimate_joint_cube_pose(
    observations: list[TagObservation],
    T_cube_tag: dict[int, np.ndarray],
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray | None, float, tuple[int, ...]]:
    object_blocks: list[np.ndarray] = []
    image_blocks: list[np.ndarray] = []
    used_ids: list[int] = []
    for observation in observations:
        if not observation.qualified or observation.tag_id not in T_cube_tag:
            continue
        object_blocks.append(
            transform_points(T_cube_tag[observation.tag_id], TAG_OBJECT_POINTS)
        )
        image_blocks.append(observation.corners)
        used_ids.append(observation.tag_id)

    if not object_blocks:
        return None, float("inf"), ()

    object_points = np.concatenate(object_blocks, axis=0)
    image_points = np.concatenate(image_blocks, axis=0)
    transform, rms = solve_pnp(object_points, image_points, camera_matrix)
    return transform, rms, tuple(sorted(used_ids))


def _pair_key(first: int, second: int) -> str:
    low, high = sorted((int(first), int(second)))
    return f"{low}-{high}"


def _pair_from_key(key: str) -> tuple[int, int]:
    first, second = key.split("-", maxsplit=1)
    return int(first), int(second)


def _graph_is_connected(
    nodes: set[int],
    edges: set[tuple[int, int]],
) -> bool:
    if not nodes:
        return False
    adjacency = {node: set() for node in nodes}
    for first, second in edges:
        if first in nodes and second in nodes:
            adjacency[first].add(second)
            adjacency[second].add(first)
    visited: set[int] = set()
    pending = [next(iter(nodes))]
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        pending.extend(adjacency[current] - visited)
    return visited == nodes


def evaluate_coverage(
    tag_counts: dict[int, int],
    pair_counts: dict[tuple[int, int], int],
    per_tag_threshold: int,
    per_edge_threshold: int,
) -> dict[str, Any]:
    required = set(REQUIRED_FACE_IDS)
    active_edges = {
        pair
        for pair, count in pair_counts.items()
        if count >= per_edge_threshold and pair[0] in required and pair[1] in required
    }
    required_counts_ok = all(
        tag_counts.get(tag_id, 0) >= per_tag_threshold for tag_id in required
    )
    required_connected = _graph_is_connected(required, active_edges)

    included_optional_ids: list[int] = []
    optional_count_ok = tag_counts.get(OPTIONAL_FACE_ID, 0) >= per_tag_threshold
    optional_edges = {
        pair
        for pair, count in pair_counts.items()
        if count >= per_edge_threshold and OPTIONAL_FACE_ID in pair
    }
    optional_connected = any(
        (pair[0] if pair[1] == OPTIONAL_FACE_ID else pair[1]) in required
        for pair in optional_edges
    )
    if optional_count_ok and optional_connected:
        included_optional_ids.append(OPTIONAL_FACE_ID)
        active_edges |= optional_edges

    return {
        "passes": bool(required_counts_ok and required_connected),
        "graph_connected": bool(required_connected),
        "included_optional_ids": included_optional_ids,
        "used_tag_ids": sorted(required | set(included_optional_ids)),
        "active_edges": sorted(active_edges),
        "per_tag_threshold": int(per_tag_threshold),
        "per_edge_threshold": int(per_edge_threshold),
    }


def make_coverage_report(
    frames: list[FrameObservation],
    total_frames: int,
) -> dict[str, Any]:
    tag_counts: dict[int, int] = {}
    pair_counts: dict[tuple[int, int], int] = {}
    for frame in frames:
        visible = sorted(
            {
                observation.tag_id
                for observation in frame.observations
                if observation.qualified and 0 <= observation.tag_id <= OPTIONAL_FACE_ID
            }
        )
        for tag_id in visible:
            tag_counts[tag_id] = tag_counts.get(tag_id, 0) + 1
        for pair in combinations(visible, 2):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    normal = evaluate_coverage(
        tag_counts,
        pair_counts,
        NORMAL_TAG_COUNT,
        NORMAL_EDGE_COUNT,
    )
    relaxed = evaluate_coverage(
        tag_counts,
        pair_counts,
        RELAXED_TAG_COUNT,
        RELAXED_EDGE_COUNT,
    )
    if normal["passes"]:
        result = "normal"
        included_optional_ids = normal["included_optional_ids"]
        graph_connected = normal["graph_connected"]
    elif relaxed["passes"]:
        result = "relaxed_available"
        included_optional_ids = relaxed["included_optional_ids"]
        graph_connected = relaxed["graph_connected"]
    else:
        result = "insufficient"
        included_optional_ids = []
        graph_connected = relaxed["graph_connected"]

    return {
        "total_frames": int(total_frames),
        "readable_frames": len(frames),
        "qualified_detection_counts": {
            str(tag_id): int(count) for tag_id, count in sorted(tag_counts.items())
        },
        "qualified_co_visible_counts": {
            _pair_key(*pair): int(count) for pair, count in sorted(pair_counts.items())
        },
        "required_ids": list(REQUIRED_FACE_IDS),
        "included_optional_ids": included_optional_ids,
        "graph_connected": graph_connected,
        "normal_thresholds": {
            "per_tag": NORMAL_TAG_COUNT,
            "per_edge": NORMAL_EDGE_COUNT,
        },
        "relaxed_thresholds": {
            "per_tag": RELAXED_TAG_COUNT,
            "per_edge": RELAXED_EDGE_COUNT,
        },
        "result": result,
    }


def coverage_evaluation_from_report(
    report: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    tag_counts = {
        int(tag_id): int(count)
        for tag_id, count in report["qualified_detection_counts"].items()
    }
    pair_counts = {
        _pair_from_key(key): int(count)
        for key, count in report["qualified_co_visible_counts"].items()
    }
    thresholds = report[f"{mode}_thresholds"]
    return evaluate_coverage(
        tag_counts,
        pair_counts,
        int(thresholds["per_tag"]),
        int(thresholds["per_edge"]),
    )


def _qualified_used_observations(
    frame: FrameObservation,
    used_ids: set[int],
) -> list[TagObservation]:
    return [
        observation
        for observation in frame.observations
        if observation.qualified and observation.tag_id in used_ids
    ]


def select_ba_frames(
    frames: list[FrameObservation],
    used_tag_ids: list[int],
    active_edges: list[tuple[int, int]],
    edge_quota: int,
    max_frames: int = MAX_BA_FRAMES,
) -> list[FrameObservation]:
    used_ids = set(used_tag_ids)
    candidates = [
        frame
        for frame in frames
        if len(_qualified_used_observations(frame, used_ids)) >= 2
    ]
    unmet = {tuple(edge): int(edge_quota) for edge in active_edges}
    selected: list[FrameObservation] = []
    selected_indices: set[int] = set()

    while any(value > 0 for value in unmet.values()):
        best_frame: FrameObservation | None = None
        best_score: tuple[int, float, int] | None = None
        best_edges: set[tuple[int, int]] = set()
        for frame in candidates:
            if frame.frame_index in selected_indices:
                continue
            observations = _qualified_used_observations(frame, used_ids)
            visible = sorted({observation.tag_id for observation in observations})
            frame_edges = {tuple(pair) for pair in combinations(visible, 2)}
            covered_edges = {edge for edge in frame_edges if unmet.get(edge, 0) > 0}
            if not covered_edges:
                continue
            mean_rms = float(
                np.mean([observation.reprojection_rms_px for observation in observations])
            )
            score = (len(covered_edges), -mean_rms, -frame.frame_index)
            if best_score is None or score > best_score:
                best_frame = frame
                best_score = score
                best_edges = covered_edges
        if best_frame is None:
            raise RuntimeError("Available frames cannot satisfy every active edge quota")
        selected.append(best_frame)
        selected_indices.add(best_frame.frame_index)
        for edge in best_edges:
            unmet[edge] -= 1

    slots = max_frames - len(selected)
    remaining = [
        frame for frame in candidates if frame.frame_index not in selected_indices
    ]
    remaining.sort(key=lambda frame: frame.frame_index)
    if slots > 0 and remaining:
        count = min(slots, len(remaining))
        positions = np.linspace(0, len(remaining) - 1, count, dtype=int)
        selected.extend(remaining[int(position)] for position in positions)
    selected.sort(key=lambda frame: frame.frame_index)
    return selected


def _relative_pose_initialization(
    frames: list[FrameObservation],
    used_tag_ids: list[int],
) -> dict[int, np.ndarray]:
    used_ids = set(used_tag_ids)
    edge_data: dict[tuple[int, int], dict[str, Any]] = {}
    for frame in frames:
        observations = {
            observation.tag_id: observation
            for observation in _qualified_used_observations(frame, used_ids)
        }
        for first, second in combinations(sorted(observations), 2):
            first_observation = observations[first]
            second_observation = observations[second]
            if (
                first_observation.T_camera_tag is None
                or second_observation.T_camera_tag is None
            ):
                continue
            relative = (
                invert_transform(first_observation.T_camera_tag)
                @ second_observation.T_camera_tag
            )
            quality = (
                first_observation.reprojection_rms_px
                + second_observation.reprojection_rms_px
            )
            record = edge_data.setdefault(
                (first, second),
                {"count": 0, "quality": float("inf"), "transform": None},
            )
            record["count"] += 1
            if quality < record["quality"]:
                record["quality"] = quality
                record["transform"] = relative

    parent = {tag_id: tag_id for tag_id in used_tag_ids}

    def find(tag_id: int) -> int:
        while parent[tag_id] != tag_id:
            parent[tag_id] = parent[parent[tag_id]]
            tag_id = parent[tag_id]
        return tag_id

    tree_edges: list[tuple[int, int, np.ndarray]] = []
    sorted_edges = sorted(
        edge_data.items(),
        key=lambda item: (-item[1]["count"], item[1]["quality"], item[0]),
    )
    for (first, second), record in sorted_edges:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            continue
        parent[second_root] = first_root
        tree_edges.append((first, second, record["transform"]))
        if len(tree_edges) == len(used_tag_ids) - 1:
            break
    if len(tree_edges) != len(used_tag_ids) - 1:
        raise RuntimeError("Cannot initialize a connected tag pose graph")

    adjacency: dict[int, list[tuple[int, np.ndarray]]] = {
        tag_id: [] for tag_id in used_tag_ids
    }
    for first, second, T_first_second in tree_edges:
        adjacency[first].append((second, T_first_second))
        adjacency[second].append((first, invert_transform(T_first_second)))

    transforms = {REFERENCE_TAG_ID: np.eye(4, dtype=np.float64)}
    pending = [REFERENCE_TAG_ID]
    while pending:
        current = pending.pop()
        for neighbor, T_current_neighbor in adjacency[current]:
            if neighbor in transforms:
                continue
            transforms[neighbor] = transforms[current] @ T_current_neighbor
            pending.append(neighbor)
    return transforms


def run_cube_bundle_adjustment(
    frames: list[FrameObservation],
    used_tag_ids: list[int],
    camera_matrix: np.ndarray,
) -> dict[str, Any]:
    if REFERENCE_TAG_ID not in used_tag_ids:
        raise ValueError("Reference tag ID 2 must be part of cube calibration")
    tag_initial = _relative_pose_initialization(frames, used_tag_ids)
    optimized_tag_ids = [
        tag_id for tag_id in sorted(used_tag_ids) if tag_id != REFERENCE_TAG_ID
    ]
    tag_offsets = {tag_id: index * 6 for index, tag_id in enumerate(optimized_tag_ids)}
    frame_base = len(optimized_tag_ids) * 6
    frame_offsets = {
        frame.frame_index: frame_base + index * 6 for index, frame in enumerate(frames)
    }

    initial_parts = [
        transform_to_vector(tag_initial[tag_id]) for tag_id in optimized_tag_ids
    ]
    used_ids = set(used_tag_ids)
    observation_blocks: list[tuple[int, int, np.ndarray]] = []
    for frame in frames:
        observations = _qualified_used_observations(frame, used_ids)
        if len(observations) < 2:
            raise ValueError("Bundle adjustment received a single-tag frame")
        best = min(observations, key=lambda observation: observation.reprojection_rms_px)
        if best.T_camera_tag is None:
            raise RuntimeError("A qualified tag observation has no pose")
        T_camera_rig = best.T_camera_tag @ invert_transform(tag_initial[best.tag_id])
        initial_parts.append(transform_to_vector(T_camera_rig))
        observation_blocks.extend(
            (frame.frame_index, observation.tag_id, observation.corners)
            for observation in observations
        )

    x0 = np.concatenate(initial_parts)
    residual_count = len(observation_blocks) * 8
    parameter_count = len(x0)
    sparsity = lil_matrix((residual_count, parameter_count), dtype=np.int8)
    for block_index, (frame_index, tag_id, _) in enumerate(observation_blocks):
        row = block_index * 8
        frame_offset = frame_offsets[frame_index]
        sparsity[row : row + 8, frame_offset : frame_offset + 6] = 1
        if tag_id != REFERENCE_TAG_ID:
            tag_offset = tag_offsets[tag_id]
            sparsity[row : row + 8, tag_offset : tag_offset + 6] = 1

    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)

    def unpack(vector: np.ndarray) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        tag_transforms = {REFERENCE_TAG_ID: np.eye(4, dtype=np.float64)}
        for tag_id, offset in tag_offsets.items():
            tag_transforms[tag_id] = vector_to_transform(vector[offset : offset + 6])
        frame_transforms = {
            frame_index: vector_to_transform(vector[offset : offset + 6])
            for frame_index, offset in frame_offsets.items()
        }
        return tag_transforms, frame_transforms

    def residuals(vector: np.ndarray) -> np.ndarray:
        tag_transforms, frame_transforms = unpack(vector)
        output = np.empty(residual_count, dtype=np.float64)
        for block_index, (frame_index, tag_id, corners) in enumerate(observation_blocks):
            T_camera_tag = frame_transforms[frame_index] @ tag_transforms[tag_id]
            projected = project_points(TAG_OBJECT_POINTS, T_camera_tag, camera_matrix)
            row = block_index * 8
            output[row : row + 8] = (projected - corners).reshape(8)
        return output

    result = least_squares(
        residuals,
        x0,
        method="trf",
        loss="linear",
        jac_sparsity=sparsity.tocsr(),
        x_scale="jac",
        verbose=2,
    )
    tag_transforms, frame_transforms = unpack(result.x)
    finite = bool(np.all(np.isfinite(result.x)) and np.all(np.isfinite(result.fun)))
    corner_errors = result.fun.reshape(-1, 2)
    corner_norms = np.linalg.norm(corner_errors, axis=1)

    per_tag_values: dict[int, list[float]] = {tag_id: [] for tag_id in used_tag_ids}
    for block_index, (_, tag_id, _) in enumerate(observation_blocks):
        start = block_index * 4
        per_tag_values[tag_id].extend(corner_norms[start : start + 4].tolist())
    per_tag_rms = {
        tag_id: float(np.sqrt(np.mean(np.square(values))))
        for tag_id, values in per_tag_values.items()
    }

    return {
        "optimizer_success": bool(result.success and finite),
        "optimizer_message": str(result.message),
        "T_rig_tag": tag_transforms,
        "T_camera_rig": frame_transforms,
        "frame_count": len(frames),
        "corner_count": len(observation_blocks) * 4,
        "overall_rms_px": float(np.sqrt(np.mean(np.square(corner_norms)))),
        "per_tag_rms_px": per_tag_rms,
    }


def build_cube_center_frame(T_rig_tag: dict[int, np.ndarray]) -> np.ndarray:
    missing = set(REQUIRED_FACE_IDS) - set(T_rig_tag)
    if missing:
        raise ValueError(f"Missing required cube faces: {sorted(missing)}")

    origins = {tag_id: transform[:3, 3] for tag_id, transform in T_rig_tag.items()}
    inward_normals = {
        tag_id: normalize(transform[:3, 2]) for tag_id, transform in T_rig_tag.items()
    }

    x_axis = normalize(inward_normals[1] - inward_normals[3])
    y_raw = inward_normals[0] - inward_normals[4]
    y_axis = normalize(y_raw - x_axis * float(np.dot(x_axis, y_raw)))
    z_axis = normalize(np.cross(x_axis, y_axis))
    if float(np.dot(z_axis, -inward_normals[2])) <= 0.0:
        raise ValueError("Calibrated face normals do not match the cube axis convention")

    center_x = 0.5 * (
        float(np.dot(x_axis, origins[3])) + float(np.dot(x_axis, origins[1]))
    )
    center_y = 0.5 * (
        float(np.dot(y_axis, origins[4])) + float(np.dot(y_axis, origins[0]))
    )
    if OPTIONAL_FACE_ID in T_rig_tag:
        center_z = 0.5 * (
            float(np.dot(z_axis, origins[2]))
            + float(np.dot(z_axis, origins[OPTIONAL_FACE_ID]))
        )
    else:
        center_z = float(np.dot(z_axis, origins[2])) - CUBE_SIDE_MM / 2.0

    T_rig_cube = np.eye(4, dtype=np.float64)
    T_rig_cube[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    T_rig_cube[:3, 3] = (
        center_x * x_axis + center_y * y_axis + center_z * z_axis
    )
    return T_rig_cube


def make_cube_calibration(
    ba_result: dict[str, Any],
    used_tag_ids: list[int],
) -> dict[str, Any]:
    if not ba_result["optimizer_success"]:
        raise RuntimeError(f"Bundle adjustment failed: {ba_result['optimizer_message']}")
    T_rig_cube = build_cube_center_frame(ba_result["T_rig_tag"])
    T_cube_rig = invert_transform(T_rig_cube)
    T_cube_tag = {
        tag_id: T_cube_rig @ ba_result["T_rig_tag"][tag_id]
        for tag_id in used_tag_ids
    }
    return {
        "tag_size_mm": TAG_SIZE_MM,
        "cube_side_mm": CUBE_SIDE_MM,
        "reference_tag_id": REFERENCE_TAG_ID,
        "used_tag_ids": sorted(used_tag_ids),
        "T_cube_tag": {str(tag_id): transform for tag_id, transform in T_cube_tag.items()},
        "ba": {
            "frame_count": int(ba_result["frame_count"]),
            "corner_count": int(ba_result["corner_count"]),
            "optimizer_success": True,
            "overall_rms_px": float(ba_result["overall_rms_px"]),
            "per_tag_rms_px": {
                str(tag_id): float(value)
                for tag_id, value in ba_result["per_tag_rms_px"].items()
            },
        },
    }


def cube_axis_points(axis_length_mm: float = 50.0) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length_mm, 0.0, 0.0],
            [0.0, axis_length_mm, 0.0],
            [0.0, 0.0, axis_length_mm],
        ],
        dtype=np.float64,
    )


def make_snout_candidate(
    frame_index: int,
    T_camera_cube: np.ndarray,
    used_tag_ids: tuple[int, ...],
    reprojection_rms_px: float,
    source_path: str = "",
    source_frame_index: int | None = None,
) -> SnoutCandidate:
    T_cube_camera = invert_transform(T_camera_cube)
    direction = normalize(T_cube_camera[:3, 3])
    return SnoutCandidate(
        frame_index=int(frame_index),
        T_camera_cube=np.asarray(T_camera_cube, dtype=np.float64),
        used_tag_ids=tuple(used_tag_ids),
        reprojection_rms_px=float(reprojection_rms_px),
        view_direction_cube=direction,
        source_path=str(source_path),
        source_frame_index=(
            None if source_frame_index is None else int(source_frame_index)
        ),
    )


def _minimum_view_angle(
    candidate: SnoutCandidate,
    selected: list[SnoutCandidate],
) -> float:
    if not selected:
        return float("inf")
    angles = [
        np.arccos(
            np.clip(
                float(np.dot(candidate.view_direction_cube, item.view_direction_cube)),
                -1.0,
                1.0,
            )
        )
        for item in selected
    ]
    return float(min(angles))


def order_snout_candidates(
    candidates: list[SnoutCandidate],
    max_candidates: int = MAX_SNOUT_CANDIDATES,
) -> list[SnoutCandidate]:
    selected: list[SnoutCandidate] = []
    tiers = [
        [candidate for candidate in candidates if len(candidate.used_tag_ids) >= 2],
        [candidate for candidate in candidates if len(candidate.used_tag_ids) == 1],
    ]
    for tier in tiers:
        if not tier or len(selected) >= max_candidates:
            continue
        remaining = list(tier)
        first = min(
            remaining,
            key=lambda candidate: (candidate.reprojection_rms_px, candidate.frame_index),
        )
        selected.append(first)
        remaining.remove(first)
        while remaining and len(selected) < max_candidates:
            next_candidate = max(
                remaining,
                key=lambda candidate: (
                    _minimum_view_angle(candidate, selected),
                    -candidate.reprojection_rms_px,
                    -candidate.frame_index,
                ),
            )
            selected.append(next_candidate)
            remaining.remove(next_candidate)
    return selected


def pixel_ray_in_cube(
    pixel: tuple[float, float],
    camera_matrix: np.ndarray,
    T_camera_cube: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    direction_camera = normalize(
        np.array(
            [
                (pixel[0] - camera_matrix[0, 2]) / camera_matrix[0, 0],
                (pixel[1] - camera_matrix[1, 2]) / camera_matrix[1, 1],
                1.0,
            ],
            dtype=np.float64,
        )
    )
    T_cube_camera = invert_transform(T_camera_cube)
    origin_cube = T_cube_camera[:3, 3]
    direction_cube = normalize(T_cube_camera[:3, :3] @ direction_camera)
    return origin_cube, direction_cube


def triangulate_rays(
    origins: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    if len(origins) < 2 or len(origins) != len(directions):
        raise ValueError("At least two matching rays are required")
    identity = np.eye(3, dtype=np.float64)
    matrix = np.zeros((3, 3), dtype=np.float64)
    right_hand_side = np.zeros(3, dtype=np.float64)
    for origin, direction in zip(origins, directions):
        direction = normalize(direction)
        projector = identity - np.outer(direction, direction)
        matrix += projector
        right_hand_side += projector @ origin
    point, _, _, _ = np.linalg.lstsq(matrix, right_hand_side, rcond=None)
    return point


def solve_snout_position(
    clicks: list[tuple[SnoutCandidate, tuple[float, float]]],
    camera_matrix: np.ndarray,
) -> dict[str, Any]:
    origins: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    for candidate, pixel in clicks:
        origin, direction = pixel_ray_in_cube(
            pixel,
            camera_matrix,
            candidate.T_camera_cube,
        )
        origins.append(origin)
        directions.append(direction)
    point = triangulate_rays(np.asarray(origins), np.asarray(directions))

    errors: list[float] = []
    for candidate, pixel in clicks:
        projected = project_points(
            point.reshape(1, 3),
            candidate.T_camera_cube,
            camera_matrix,
        )[0]
        errors.append(float(np.linalg.norm(projected - np.asarray(pixel))))

    view_directions = [candidate.view_direction_cube for candidate, _ in clicks]
    view_span = 0.0
    for first, second in combinations(view_directions, 2):
        angle = np.degrees(
            np.arccos(np.clip(float(np.dot(first, second)), -1.0, 1.0))
        )
        view_span = max(view_span, float(angle))

    result = {
        "p_snout_cube_mm": point,
        "click_count": len(clicks),
        "reprojection_rms_px": float(np.sqrt(np.mean(np.square(errors)))),
        "per_click_errors_px": errors,
        "view_span_deg": view_span,
    }
    numeric_values = np.concatenate(
        [
            np.asarray(result["p_snout_cube_mm"], dtype=np.float64),
            np.asarray(result["per_click_errors_px"], dtype=np.float64),
            np.array(
                [result["reprojection_rms_px"], result["view_span_deg"]],
                dtype=np.float64,
            ),
        ]
    )
    if not np.all(np.isfinite(numeric_values)):
        raise RuntimeError("Snout solution contains non-finite values")
    return result


def project_point_to_plane(
    point: np.ndarray,
    plane_origin: np.ndarray,
    plane_normal: np.ndarray,
) -> np.ndarray:
    point = np.asarray(point, dtype=np.float64).reshape(3)
    plane_origin = np.asarray(plane_origin, dtype=np.float64).reshape(3)
    plane_normal = normalize(plane_normal)
    distance = float(np.dot(plane_normal, point - plane_origin))
    return point - distance * plane_normal


def snout_projection_on_id4(
    snout_point_cube: np.ndarray,
    T_cube_tag4: np.ndarray,
) -> np.ndarray:
    return project_point_to_plane(
        snout_point_cube,
        np.asarray(T_cube_tag4, dtype=np.float64)[:3, 3],
        np.asarray(T_cube_tag4, dtype=np.float64)[:3, 2],
    )
