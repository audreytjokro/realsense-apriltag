from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from h264_video import H264VideoWriter

import display
import geometry
import storage
from camera import COLOR_FPS, COLOR_HEIGHT, COLOR_WIDTH, RectifiedColorCamera
from visualize_live import run_live


RECORD_WINDOW = "Cube Calibration Recording"
SNOUT_WINDOW = "Snout Calibration"
PROGRESS_WIDTH = 20
PROGRESS_UPDATE_FRAMES = 10

# Opposite cube faces cannot be photographed together. These are all physically
# possible connections among the five required faces.
PHYSICAL_NEIGHBOR_PAIRS = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
)


def choose_menu(title: str, options: list[tuple[str, str]]) -> str:
    print()
    print(title)
    for index, (_, label) in enumerate(options, start=1):
        print(f"  {index}. {label}")
    while True:
        response = input("Select an option: ").strip()
        if response.isdigit():
            index = int(response) - 1
            if 0 <= index < len(options):
                return options[index][0]
        print("Invalid selection.")


def show_overview() -> None:
    print(
        "\nCube and Snout Calibration\n\n"
        "This workflow contains five stages:\n\n"
        "  1. Record one or more tagged-cube clips from varied viewpoints.\n"
        "  2. Analyze tag visibility and multi-face connectivity.\n"
        "  3. Estimate the fixed 6DoF relationship between the cube faces.\n"
        "  4. Calibrate the snout position from manually marked image points.\n"
        "  5. Inspect the calibrated geometry in a real-time demonstration.\n\n"
        "All images are rectified before detection and pose estimation.\n"
        "Multi-face observations connect the individual tag coordinate systems.\n"
        "Sparse bundle adjustment jointly refines the fixed tag poses and\n"
        "the per-frame cube poses using all selected corner observations.\n\n"
        "The snout is calibrated without depth data. Each manual image point\n"
        "defines a camera ray, and the final position is estimated from the\n"
        "least-squares intersection of rays expressed in cube coordinates.\n\n"
        "If coverage is missing a connection, add a short targeted clip; the\n"
        "existing clips and their counts are preserved and combined.\n\n"
        "Each completed stage is saved automatically.\n"
        "A later run continues from the latest completed stage.\n"
    )
    input("Press Enter to continue.")


def show_recording_guide(output_path: Path, additional: bool) -> None:
    recording_kind = (
        "This is an additional clip. Existing recordings will be kept and all\n"
        "coverage counts will be combined.\n\n"
        if additional
        else "This is the first clip in the calibration set.\n\n"
    )
    print(
        "\nTargeted Cube Recording\n\n"
        f"{recording_kind}"
        "The recording provides the observations used to calibrate the fixed\n"
        "relationship between the tagged faces.\n\n"
        "A pair counts only when both IDs are qualified in the SAME frame.\n"
        "Alternating between two IDs does not connect them. Short simultaneous\n"
        "bursts are fine: pair counts accumulate and need not be continuous.\n\n"
        "Use the live TARGET line to concentrate on a missing graph connection.\n"
        "The NOW line tells you whether a pair is accumulating in this frame.\n"
        "Keep full black tag borders visible and avoid motion blur.\n\n"
        f"Saving this clip to: {output_path}\n\n"
        "Press Space in the camera window to begin recording.\n"
        "Press Space again when the cube has been fully demonstrated.\n"
    )


def show_snout_guide() -> None:
    print(
        "\nSnout Position Calibration\n\n"
        "The following frames were selected to provide diverse camera-to-cube\n"
        "viewing directions.\n\n"
        "In each usable frame, click the visible center of the yellow terminal\n"
        "face. The selected pixel defines a camera ray. After 30 marked views,\n"
        "the rays are transformed into cube coordinates and solved jointly for\n"
        "the snout position.\n\n"
        "Click again to correct the current point before continuing.\n"
        "If the target is occluded or its center is uncertain, leave the image\n"
        "unmarked and continue to the next frame.\n\n"
        "Depth data is not used.\n\n"
        "Closing the window ends this calibration attempt without saving\n"
        "the current clicks.\n"
    )
    input("Press Enter to continue.")


def print_progress(label: str, current: int, total: int) -> None:
    fraction = min(float(current) / float(total), 1.0)
    filled = int(round(PROGRESS_WIDTH * fraction))
    bar = "#" * filled + "-" * (PROGRESS_WIDTH - filled)
    percent = int(round(100.0 * fraction))
    print(
        f"\r{label:<26} [{bar}] {percent:3d}%  {current}/{total} frames",
        end="",
        flush=True,
    )


def print_coverage_guidance(report: dict[str, Any]) -> None:
    result = report["result"]
    if result == "normal":
        print(
            "\nCoverage analysis passed.\n\n"
            "The required face graph is connected, and the recording provides\n"
            "sufficient multi-face observations for cube calibration."
        )
        return
    if result == "relaxed_available":
        print(
            "\nThe normal coverage criterion was not met.\n\n"
            "This recording satisfies the relaxed acceptance criterion and can\n"
            "still be used for calibration.\n\n"
            "Normal criterion:\n"
            "  - 60 qualified observations per required face\n"
            "  - 20 qualified co-visible frames per selected graph edge\n\n"
            "Relaxed criterion:\n"
            "  - 30 qualified observations per required face\n"
            "  - 10 qualified co-visible frames per selected graph edge\n\n"
            "The required face graph must remain connected in both cases.\n"
            "Review the reported face and edge counts before continuing."
        )
        return
    print(
        "\nCoverage analysis did not pass.\n\n"
        "Minimum acceptance requires:\n"
        "  - 30 qualified observations for every required face\n"
        "  - 10 qualified co-visible frames for each selected graph edge\n"
        "  - a connected graph spanning all required faces\n\n"
        "A qualified observation requires:\n"
        "  - mean tag side length of at least 50 pixels\n"
        "  - single-tag corner reprojection RMS no greater than 2 pixels\n\n"
        "Review the reported face and edge counts above. Record another video\n"
        "with more views of missing faces or disconnected face pairs."
    )


def _preview_count_lines(
    tag_counts: dict[int, int],
    pair_counts: dict[tuple[int, int], int],
    visible_ids: list[int],
    recording: bool,
) -> list[str]:
    now_text = ",".join(str(tag_id) for tag_id in visible_ids) or "none"
    if len(visible_ids) >= 2:
        now_state = "PAIR COUNTING" if recording else "PAIR READY; press Space"
    elif len(visible_ids) == 1:
        now_state = "one face only; pair not counting"
    else:
        now_state = "no qualified cube face"

    tag_text = "  ".join(
        f"{tag_id}:{tag_counts.get(tag_id, 0)}/{geometry.NORMAL_TAG_COUNT}"
        for tag_id in geometry.REQUIRED_FACE_IDS
    )
    pair_items = []
    for first, second in PHYSICAL_NEIGHBOR_PAIRS:
        count = pair_counts.get((first, second), 0)
        suffix = " OK" if count >= geometry.NORMAL_EDGE_COUNT else ""
        pair_items.append(
            f"{first}-{second}:{count}/{geometry.NORMAL_EDGE_COUNT}{suffix}"
        )

    components = _coverage_components(pair_counts, geometry.NORMAL_EDGE_COUNT)
    normal = geometry.evaluate_coverage(
        tag_counts,
        pair_counts,
        geometry.NORMAL_TAG_COUNT,
        geometry.NORMAL_EDGE_COUNT,
    )
    relaxed = geometry.evaluate_coverage(
        tag_counts,
        pair_counts,
        geometry.RELAXED_TAG_COUNT,
        geometry.RELAXED_EDGE_COUNT,
    )

    lines = [
        f"NOW qualified: {now_text} | {now_state}",
        f"Face totals: {tag_text}",
        "Edges: " + "  ".join(pair_items[:4]),
        "Edges: " + "  ".join(pair_items[4:]),
    ]
    if normal["passes"]:
        lines.append("GRAPH CONNECTED: normal coverage is ready")
        return lines
    if relaxed["passes"]:
        lines.append("GRAPH CONNECTED: relaxed coverage ready; keep going for normal")
        return lines

    component_text = "  ".join(
        "[" + ",".join(str(tag_id) for tag_id in component) + "]"
        for component in components
    )
    suggestions = _bridge_suggestions(pair_counts, components)
    target_text = " | ".join(
        f"{first}-{second} ({pair_counts.get((first, second), 0)}/{geometry.NORMAL_EDGE_COUNT})"
        for first, second in suggestions[:3]
    )
    lines.append(f"GRAPH NOT CONNECTED: components {component_text}")
    lines.append(f"TARGET simultaneous pair: {target_text or 'build the listed edges'}")
    return lines


def _coverage_components(
    pair_counts: dict[tuple[int, int], int],
    edge_threshold: int,
) -> list[tuple[int, ...]]:
    remaining = set(geometry.REQUIRED_FACE_IDS)
    adjacency = {tag_id: set() for tag_id in remaining}
    for (first, second), count in pair_counts.items():
        if count < edge_threshold or first not in adjacency or second not in adjacency:
            continue
        adjacency[first].add(second)
        adjacency[second].add(first)

    components: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining)
        pending = [start]
        component: set[int] = set()
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(adjacency[current] - component)
        remaining -= component
        components.append(tuple(sorted(component)))
    return components


def _bridge_suggestions(
    pair_counts: dict[tuple[int, int], int],
    components: list[tuple[int, ...]],
) -> list[tuple[int, int]]:
    component_by_tag = {
        tag_id: component_index
        for component_index, component in enumerate(components)
        for tag_id in component
    }
    candidates = [
        pair
        for pair in PHYSICAL_NEIGHBOR_PAIRS
        if component_by_tag[pair[0]] != component_by_tag[pair[1]]
    ]
    candidates.sort(key=lambda pair: (-pair_counts.get(pair, 0), pair))
    return candidates


def _counts_from_report(
    report: dict[str, Any] | None,
) -> tuple[dict[int, int], dict[tuple[int, int], int]]:
    if report is None:
        return {}, {}
    tag_counts = {
        int(tag_id): int(count)
        for tag_id, count in report["qualified_detection_counts"].items()
    }
    pair_counts = {
        tuple(int(value) for value in pair.split("-")): int(count)
        for pair, count in report["qualified_co_visible_counts"].items()
    }
    return tag_counts, pair_counts


def record_video(
    output_path: Path,
    starting_report: dict[str, Any] | None = None,
    additional: bool = False,
) -> None:
    if output_path.exists():
        raise FileExistsError(output_path)
    show_recording_guide(output_path, additional)
    saved_intrinsics = storage.load_camera_intrinsics()
    writer: H264VideoWriter | None = None
    recording = False
    tag_counts, pair_counts = _counts_from_report(starting_report)

    storage.ensure_data_dir()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.namedWindow(RECORD_WINDOW, cv2.WINDOW_NORMAL)
    try:
        with RectifiedColorCamera(saved_intrinsics) as camera:
            if saved_intrinsics is None:
                if camera.intrinsics_record is None:
                    raise RuntimeError("Camera did not expose color intrinsics")
                storage.save_camera_intrinsics(camera.intrinsics_record)
                print(f"Saved {storage.CAMERA_INTRINSICS_PATH}")

            intrinsics = camera.intrinsics_record
            if intrinsics is None:
                raise RuntimeError("Camera intrinsics are unavailable")
            camera_matrix = np.asarray(intrinsics["rectified_K"], dtype=np.float64)

            while True:
                frame = camera.read()
                observations = geometry.detect_tags(frame, camera_matrix)
                visible_ids = sorted(
                    {
                        observation.tag_id
                        for observation in observations
                        if (
                            observation.qualified
                            and 0 <= observation.tag_id <= geometry.OPTIONAL_FACE_ID
                        )
                    }
                )
                if recording:
                    if writer is None:
                        raise RuntimeError("Recording is active without a video writer")
                    writer.write(frame)
                    for tag_id in visible_ids:
                        tag_counts[tag_id] = tag_counts.get(tag_id, 0) + 1
                    for pair in combinations(visible_ids, 2):
                        pair_counts[pair] = pair_counts.get(pair, 0) + 1

                output = frame.copy()
                display.draw_tag_observations(output, observations)
                if recording:
                    lines = [
                        "Rotate through multi-face views",
                        "Space: finish recording | Close window: exit",
                        "Status: RECORDING",
                    ]
                else:
                    lines = [
                        "Space: start recording | Close window: exit",
                        "Status: PREVIEW",
                    ]
                lines.extend(
                    _preview_count_lines(
                        tag_counts,
                        pair_counts,
                        visible_ids,
                        recording,
                    )
                )
                display.draw_text_lines(output, lines)
                cv2.imshow(RECORD_WINDOW, output)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    if not recording:
                        writer = H264VideoWriter(
                            output_path,
                            COLOR_FPS,
                            (COLOR_WIDTH, COLOR_HEIGHT),
                        )
                        recording = True
                        print("Recording started.")
                    else:
                        print("Recording completed.")
                        print(f"Saved clip: {output_path}")
                        return
                if display.window_was_closed(RECORD_WINDOW):
                    raise SystemExit(0)
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def _video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def scan_recordings() -> tuple[list[geometry.FrameObservation], dict[str, Any]]:
    intrinsics = storage.load_camera_intrinsics()
    if intrinsics is None:
        raise FileNotFoundError(storage.CAMERA_INTRINSICS_PATH)
    camera_matrix = np.asarray(intrinsics["rectified_K"], dtype=np.float64)
    recording_paths = storage.recording_paths()
    if not recording_paths:
        raise FileNotFoundError("No cube calibration recordings are available")
    clip_frame_counts = {
        path: _video_frame_count(path) for path in recording_paths
    }
    total_frames = sum(clip_frame_counts.values())
    frames: list[geometry.FrameObservation] = []
    global_frame_index = 0
    print(
        f"\nAnalyzing {len(recording_paths)} rectified recording(s) "
        "and detecting tag corners..."
    )
    recording_records: list[dict[str, Any]] = []
    for clip_index, path in enumerate(recording_paths, start=1):
        print(f"  Clip {clip_index}/{len(recording_paths)}: {path.name}")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        local_frame_index = 0
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                observations = geometry.detect_tags(frame, camera_matrix)
                frames.append(
                    geometry.FrameObservation(
                        frame_index=global_frame_index,
                        observations=observations,
                        source_path=str(path.resolve()),
                        source_frame_index=local_frame_index,
                    )
                )
                local_frame_index += 1
                global_frame_index += 1
                if global_frame_index % PROGRESS_UPDATE_FRAMES == 0:
                    print_progress(
                        "Analyzing recordings",
                        global_frame_index,
                        total_frames,
                    )
        finally:
            capture.release()
        try:
            display_path = str(path.relative_to(storage.DATA_DIR))
        except ValueError:
            display_path = str(path)
        recording_records.append(
            {
                "path": display_path,
                "expected_frames": clip_frame_counts[path],
                "readable_frames": local_frame_index,
            }
        )
    print_progress("Analyzing recordings", global_frame_index, total_frames)
    print()

    print("Evaluating qualified tag observations...")
    print("Building the multi-face co-visibility graph...")
    print("Checking required-face connectivity...")
    report = geometry.make_coverage_report(frames, total_frames)
    report["recordings"] = recording_records
    storage.save_coverage(report)
    print_coverage_report(report)
    print_coverage_guidance(report)
    return frames, report


def print_coverage_report(report: dict[str, Any]) -> None:
    recordings = report.get("recordings", [])
    if recordings:
        print(f"Recordings combined: {len(recordings)}")
        for recording in recordings:
            print(
                f"  {recording['path']}: "
                f"{recording['readable_frames']}/{recording['expected_frames']} frames"
            )
    print(f"Readable frames: {report['readable_frames']}/{report['total_frames']}")
    print("Qualified detections:")
    for tag_id, count in report["qualified_detection_counts"].items():
        print(f"  ID {tag_id}: {count}")
    print("Qualified co-visible pairs:")
    for pair, count in report["qualified_co_visible_counts"].items():
        print(f"  {pair}: {count}")
    print(f"Required graph connected: {report['graph_connected']}")
    print(f"Included optional IDs: {report['included_optional_ids']}")
    print(f"Coverage result: {report['result']}")
    if report["result"] == "insufficient":
        _, pair_counts = _counts_from_report(report)
        components = _coverage_components(pair_counts, geometry.RELAXED_EDGE_COUNT)
        suggestions = _bridge_suggestions(pair_counts, components)
        component_text = " ".join(
            "[" + ",".join(str(tag_id) for tag_id in component) + "]"
            for component in components
        )
        print(f"Components at relaxed edge threshold: {component_text}")
        if suggestions:
            suggestion_text = ", ".join(
                f"{first}-{second} ({pair_counts.get((first, second), 0)}/10)"
                for first, second in suggestions[:3]
            )
            print(f"Target one of these simultaneous bridges: {suggestion_text}")
    print(f"Saved {storage.COVERAGE_PATH}")


def record_additional_clip(report: dict[str, Any]) -> Path:
    output_path = storage.new_additional_recording_path()
    tag_counts, pair_counts = _counts_from_report(report)
    components = _coverage_components(pair_counts, geometry.NORMAL_EDGE_COUNT)
    suggestions = _bridge_suggestions(pair_counts, components)
    deficient_faces = [
        tag_id
        for tag_id in geometry.REQUIRED_FACE_IDS
        if tag_counts.get(tag_id, 0) < geometry.NORMAL_TAG_COUNT
    ]
    print("\nTARGETED ADDITIONAL CLIP")
    print("Existing recordings will remain part of the calibration set.")
    if deficient_faces:
        print(f"Faces still below 60 detections: {deficient_faces}")
    if suggestions:
        text = ", ".join(f"{first}-{second}" for first, second in suggestions[:3])
        print(f"Best graph bridges to target: {text}")
    record_video(output_path, starting_report=report, additional=True)
    return output_path


def calibrate_cube(
    frames: list[geometry.FrameObservation],
    report: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    print("\nCube calibration\n")
    print("[1/4] Evaluating the calibrated face set...")
    intrinsics = storage.load_camera_intrinsics()
    if intrinsics is None:
        raise FileNotFoundError(storage.CAMERA_INTRINSICS_PATH)
    camera_matrix = np.asarray(intrinsics["rectified_K"], dtype=np.float64)
    evaluation = geometry.coverage_evaluation_from_report(report, mode)
    if not evaluation["passes"]:
        raise RuntimeError(f"Coverage does not pass {mode} thresholds")

    print("[2/4] Selecting multi-face frames...")
    selected_frames = geometry.select_ba_frames(
        frames,
        evaluation["used_tag_ids"],
        evaluation["active_edges"],
        evaluation["per_edge_threshold"],
    )
    print(f"      Selected {len(selected_frames)} multi-tag frames.")
    print("[3/4] Initializing and running sparse bundle adjustment...")
    print("      Optimizing fixed tag poses and per-frame cube poses.")
    ba_result = geometry.run_cube_bundle_adjustment(
        selected_frames,
        evaluation["used_tag_ids"],
        camera_matrix,
    )
    print(f"      Optimizer success: {ba_result['optimizer_success']}")
    print(f"      Optimizer message: {ba_result['optimizer_message']}")
    print(f"      Frames: {ba_result['frame_count']}")
    print(f"      Corners: {ba_result['corner_count']}")
    print("[4/4] Constructing the right-handed cube frame...")
    cube_calibration = geometry.make_cube_calibration(
        ba_result,
        evaluation["used_tag_ids"],
    )
    storage.save_cube_calibration(cube_calibration)
    print(f"Saved {storage.CUBE_CALIBRATION_PATH}")
    print(f"BA RMS: {cube_calibration['ba']['overall_rms_px']:.4f}px")
    for tag_id, rms in cube_calibration["ba"]["per_tag_rms_px"].items():
        print(f"  ID {tag_id}: {rms:.4f}px")
    print(
        "\nCube calibration completed.\n\n"
        "The optimized tag geometry has been transformed from the internal\n"
        "ID 2 gauge into the right-handed cube-centered coordinate frame.\n\n"
        "A real-time cube pose demonstration will open next."
    )
    return cube_calibration


def run_pre_cube_workflow() -> dict[str, Any]:
    frames: list[geometry.FrameObservation] | None = None
    report: dict[str, Any] | None = None

    while True:
        if frames is None or report is None:
            if storage.recording_paths():
                recording_count = len(storage.recording_paths())
                action = choose_menu(
                    (
                        f"{recording_count} calibration recording(s) exist and "
                        "cube calibration is missing."
                    ),
                    [
                        ("scan", "Scan all existing recordings"),
                        ("add", "Record an additional targeted clip"),
                        ("exit", "Exit"),
                    ],
                )
                if action == "exit":
                    raise SystemExit(0)
                frames, report = scan_recordings()
                if action == "add":
                    record_additional_clip(report)
                    frames, report = scan_recordings()
            else:
                record_video(storage.RECORDING_PATH)
                frames, report = scan_recordings()

        result = report["result"]
        if result == "normal":
            action = choose_menu(
                "Coverage passes normal thresholds.",
                [
                    ("continue", "Continue to cube calibration"),
                    ("add", "Add another targeted clip"),
                    ("exit", "Exit"),
                ],
            )
            mode = "normal"
        elif result == "relaxed_available":
            action = choose_menu(
                "Coverage passes only relaxed count thresholds.",
                [
                    ("continue", "Accept relaxed thresholds and continue"),
                    ("add", "Add a targeted clip to reach normal coverage"),
                    ("exit", "Exit"),
                ],
            )
            mode = "relaxed"
        else:
            action = choose_menu(
                "Coverage is insufficient.",
                [
                    ("add", "Add a targeted clip (keep all existing clips)"),
                    ("rescan", "Rescan all current clips"),
                    ("exit", "Exit"),
                ],
            )
            mode = "relaxed"

        if action == "exit":
            raise SystemExit(0)
        if action == "rescan":
            frames, report = scan_recordings()
            continue
        if action == "add":
            record_additional_clip(report)
            frames, report = scan_recordings()
            continue
        cube_calibration = calibrate_cube(frames, report, mode)
        run_live(show_snout=False)
        return cube_calibration


def collect_snout_candidates(
    cube_calibration: dict[str, Any],
) -> tuple[list[geometry.SnoutCandidate], np.ndarray]:
    intrinsics = storage.load_camera_intrinsics()
    if intrinsics is None:
        raise FileNotFoundError(storage.CAMERA_INTRINSICS_PATH)
    camera_matrix = np.asarray(intrinsics["rectified_K"], dtype=np.float64)
    T_cube_tag = geometry.transform_map_from_json(cube_calibration)

    recording_paths = storage.recording_paths()
    if not recording_paths:
        raise FileNotFoundError("No cube calibration recordings are available")
    total_frames = sum(_video_frame_count(path) for path in recording_paths)
    candidates: list[geometry.SnoutCandidate] = []
    global_frame_index = 0
    print(f"\nPreparing snout calibration views from {len(recording_paths)} clip(s)")
    for path in recording_paths:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        local_frame_index = 0
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                observations = geometry.detect_tags(frame, camera_matrix)
                T_camera_cube, rms, used_ids = geometry.estimate_joint_cube_pose(
                    observations,
                    T_cube_tag,
                    camera_matrix,
                )
                if T_camera_cube is not None:
                    candidates.append(
                        geometry.make_snout_candidate(
                            global_frame_index,
                            T_camera_cube,
                            used_ids,
                            rms,
                            source_path=str(path.resolve()),
                            source_frame_index=local_frame_index,
                        )
                    )
                local_frame_index += 1
                global_frame_index += 1
                if global_frame_index % PROGRESS_UPDATE_FRAMES == 0:
                    print_progress(
                        "Estimating cube poses",
                        global_frame_index,
                        total_frames,
                    )
        finally:
            capture.release()
    print_progress("Estimating cube poses", global_frame_index, total_frames)
    print()

    print("Ordering valid frames by viewpoint diversity...")
    ordered = geometry.order_snout_candidates(candidates)
    multi_count = sum(len(candidate.used_tag_ids) >= 2 for candidate in ordered)
    print(
        f"Prepared {len(ordered)} snout candidates "
        f"({multi_count} multi-tag, {len(ordered) - multi_count} single-tag)."
    )
    return ordered, camera_matrix


def _load_candidate_frame(
    captures: dict[str, cv2.VideoCapture],
    candidate: geometry.SnoutCandidate,
) -> np.ndarray:
    source_path = candidate.source_path or str(storage.RECORDING_PATH.resolve())
    frame_index = (
        candidate.frame_index
        if candidate.source_frame_index is None
        else candidate.source_frame_index
    )
    capture = captures.get(source_path)
    if capture is None:
        capture = cv2.VideoCapture(source_path)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {source_path}")
        captures[source_path] = capture
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = capture.read()
    if not success:
        raise RuntimeError(f"Cannot read frame {frame_index} from {source_path}")
    return frame


def run_snout_calibration(cube_calibration: dict[str, Any]) -> bool:
    candidates, camera_matrix = collect_snout_candidates(cube_calibration)
    show_snout_guide()
    captures: dict[str, cv2.VideoCapture] = {}

    clicks: list[tuple[geometry.SnoutCandidate, tuple[float, float]]] = []
    current_pixel: dict[str, tuple[int, int] | None] = {"value": None}

    def on_mouse(event: int, x: int, y: int, flags: int, parameter: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            current_pixel["value"] = (x, y)

    cv2.namedWindow(SNOUT_WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(SNOUT_WINDOW, on_mouse)
    try:
        for candidate_index, candidate in enumerate(candidates, start=1):
            frame = _load_candidate_frame(captures, candidate)
            observations = geometry.detect_tags(frame, camera_matrix)
            axis_pixels = geometry.project_points(
                geometry.cube_axis_points(),
                candidate.T_camera_cube,
                camera_matrix,
            )
            current_pixel["value"] = None

            while True:
                output = frame.copy()
                display.draw_tag_observations(output, observations)
                display.draw_axes(output, axis_pixels)
                display.draw_click(output, current_pixel["value"])
                display.draw_text_lines(
                    output,
                    [
                        "Click the snout center; click again to adjust",
                        "Space: accept or skip | Close window: exit",
                        (
                            f"View {candidate_index}/{len(candidates)} | "
                            f"Marked {len(clicks)}/{geometry.TARGET_SNOUT_CLICKS} | "
                            f"Tags {','.join(str(tag_id) for tag_id in candidate.used_tag_ids)} | "
                            f"RMS {candidate.reprojection_rms_px:.2f} px"
                        ),
                    ],
                )
                cv2.imshow(SNOUT_WINDOW, output)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    if current_pixel["value"] is not None:
                        clicks.append(
                            (
                                candidate,
                                (
                                    float(current_pixel["value"][0]),
                                    float(current_pixel["value"][1]),
                                ),
                            )
                        )
                    break
                if display.window_was_closed(SNOUT_WINDOW):
                    raise SystemExit(0)

            if len(clicks) == geometry.TARGET_SNOUT_CLICKS:
                break
    finally:
        for capture in captures.values():
            capture.release()
        cv2.destroyAllWindows()

    if len(clicks) < geometry.TARGET_SNOUT_CLICKS:
        print(
            f"Snout calibration failed: {len(clicks)} valid clicks, "
            f"{geometry.TARGET_SNOUT_CLICKS} required."
        )
        return False

    result = geometry.solve_snout_position(clicks, camera_matrix)
    storage.save_snout_calibration(result)
    point = np.asarray(result["p_snout_cube_mm"], dtype=np.float64)
    print(f"Saved {storage.SNOUT_CALIBRATION_PATH}")
    print(f"Snout position [mm]: {point.tolist()}")
    print(f"Reprojection RMS: {result['reprojection_rms_px']:.4f}px")
    print(f"View span: {result['view_span_deg']:.2f}deg")
    return True


def run_cube_menu(
    cube_calibration: dict[str, Any],
    save_live_video: bool = False,
) -> None:
    while True:
        action = choose_menu(
            "Cube calibration is available.",
            [
                ("live", "Open live cube visualization"),
                ("snout", "Start snout calibration"),
                ("exit", "Exit"),
            ],
        )
        if action == "exit":
            return
        if action == "live":
            run_live(show_snout=False)
            continue
        if run_snout_calibration(cube_calibration):
            run_live(show_snout=True, save_video=save_live_video)
            return


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the tagged cube and Cyranose snout position."
    )
    parser.add_argument(
        "--save-live-video",
        action="store_true",
        help=(
            "Record the final annotated snout live demonstration to "
            "calibration/data/live_demos."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    storage.ensure_data_dir()
    show_overview()
    if storage.load_snout_calibration() is not None:
        run_live(show_snout=True, save_video=args.save_live_video)
        return

    cube_calibration = storage.load_cube_calibration()
    if cube_calibration is None:
        cube_calibration = run_pre_cube_workflow()
    run_cube_menu(cube_calibration, save_live_video=args.save_live_video)


if __name__ == "__main__":
    main()
