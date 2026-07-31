from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = ROOT_DIR / "calibration"
sys.path.insert(0, str(CALIBRATION_DIR))

import display
import geometry
import storage
from camera import RectifiedColorCamera, RectifiedColorFrame


WINDOW_NAME = "Cyranose Pose Tracking"
SESSION_PREFIX = "cyranose_pose_session"
CSV_NAME = "cyranose_pose_tracking.csv"
VIDEO_NAME = "rectified_rgb.mp4"
VIDEO_CODEC = "mp4v"
TRACKED_TAG_IDS = tuple(range(7))
MM_PER_CM = 10.0


def pose_field_names(prefix: str) -> list[str]:
    return [
        f"{prefix}_x_cm",
        f"{prefix}_y_cm",
        f"{prefix}_z_cm",
        f"{prefix}_rvec_x_rad",
        f"{prefix}_rvec_y_rad",
        f"{prefix}_rvec_z_rad",
    ]


def csv_field_names() -> list[str]:
    fields = [
        "host_timestamp_utc",
        "elapsed_s",
        "camera_timestamp_ms",
        "frame_number",
    ]
    for tag_id in TRACKED_TAG_IDS:
        prefix = f"tag_{tag_id}"
        fields.extend([f"{prefix}_visible", f"{prefix}_rms_px"])
        fields.extend(pose_field_names(f"{prefix}_cam"))
        fields.extend(pose_field_names(f"{prefix}_desk"))
    fields.extend(
        [
            "cube_pose_valid",
            "cube_used_tag_ids",
            "cube_rms_px",
        ]
    )
    fields.extend(pose_field_names("cube_cam"))
    fields.extend(pose_field_names("cube_desk"))
    fields.extend(
        [
            "snout_position_valid",
            "snout_cam_x_cm",
            "snout_cam_y_cm",
            "snout_cam_z_cm",
            "snout_desk_x_cm",
            "snout_desk_y_cm",
            "snout_desk_z_cm",
        ]
    )
    return fields


CSV_FIELDS = csv_field_names()


def write_pose_fields(row: dict[str, Any], prefix: str, transform: np.ndarray) -> None:
    transform = np.asarray(transform, dtype=np.float64)
    rvec, _ = cv2.Rodrigues(transform[:3, :3])
    position_cm = transform[:3, 3] / MM_PER_CM
    row.update(
        {
            f"{prefix}_x_cm": float(position_cm[0]),
            f"{prefix}_y_cm": float(position_cm[1]),
            f"{prefix}_z_cm": float(position_cm[2]),
            f"{prefix}_rvec_x_rad": float(rvec[0, 0]),
            f"{prefix}_rvec_y_rad": float(rvec[1, 0]),
            f"{prefix}_rvec_z_rad": float(rvec[2, 0]),
        }
    )


def write_point_fields(row: dict[str, Any], prefix: str, point_mm: np.ndarray) -> None:
    point_cm = np.asarray(point_mm, dtype=np.float64).reshape(3) / MM_PER_CM
    row.update(
        {
            f"{prefix}_x_cm": float(point_cm[0]),
            f"{prefix}_y_cm": float(point_cm[1]),
            f"{prefix}_z_cm": float(point_cm[2]),
        }
    )


def build_csv_row(
    host_timestamp_utc: str,
    elapsed_s: float,
    camera_frame: RectifiedColorFrame,
    observations: list[geometry.TagObservation],
    camera_matrix: np.ndarray,
    T_cube_tag: dict[int, np.ndarray],
    p_snout_cube_mm: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray | None, float, tuple[int, ...]]:
    row: dict[str, Any] = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "host_timestamp_utc": host_timestamp_utc,
            "elapsed_s": round(float(elapsed_s), 6),
            "camera_timestamp_ms": round(camera_frame.camera_timestamp_ms, 3),
            "frame_number": camera_frame.frame_number,
            "cube_pose_valid": False,
            "snout_position_valid": False,
        }
    )
    for tag_id in TRACKED_TAG_IDS:
        row[f"tag_{tag_id}_visible"] = False

    qualified = {
        observation.tag_id: observation
        for observation in observations
        if (
            observation.tag_id in TRACKED_TAG_IDS
            and observation.qualified
            and observation.T_camera_tag is not None
        )
    }
    desk_observation = qualified.get(geometry.DESK_TAG_ID)
    T_desk_camera = None
    if desk_observation is not None:
        T_desk_camera = geometry.invert_transform(desk_observation.T_camera_tag)

    for tag_id, observation in qualified.items():
        prefix = f"tag_{tag_id}"
        row[f"{prefix}_visible"] = True
        row[f"{prefix}_rms_px"] = float(observation.reprojection_rms_px)
        write_pose_fields(row, f"{prefix}_cam", observation.T_camera_tag)
        if T_desk_camera is not None:
            T_desk_tag = T_desk_camera @ observation.T_camera_tag
            write_pose_fields(row, f"{prefix}_desk", T_desk_tag)

    T_camera_cube, cube_rms, used_ids = geometry.estimate_joint_cube_pose(
        observations,
        T_cube_tag,
        camera_matrix,
    )
    if T_camera_cube is not None:
        row["cube_pose_valid"] = True
        row["cube_used_tag_ids"] = ";".join(str(tag_id) for tag_id in used_ids)
        row["cube_rms_px"] = float(cube_rms)
        write_pose_fields(row, "cube_cam", T_camera_cube)

        p_camera_snout = geometry.transform_points(
            T_camera_cube,
            np.asarray(p_snout_cube_mm, dtype=np.float64).reshape(1, 3),
        )[0]
        row["snout_position_valid"] = True
        write_point_fields(row, "snout_cam", p_camera_snout)

        if T_desk_camera is not None:
            T_desk_cube = T_desk_camera @ T_camera_cube
            write_pose_fields(row, "cube_desk", T_desk_cube)
            p_desk_snout = geometry.transform_points(
                T_desk_camera,
                p_camera_snout.reshape(1, 3),
            )[0]
            write_point_fields(row, "snout_desk", p_desk_snout)

    return row, T_camera_cube, cube_rms, used_ids


def load_calibration() -> tuple[dict[str, Any], dict[int, np.ndarray], np.ndarray]:
    intrinsics = storage.load_camera_intrinsics()
    if intrinsics is None:
        raise FileNotFoundError(storage.CAMERA_INTRINSICS_PATH)
    cube_calibration = storage.load_cube_calibration()
    if cube_calibration is None:
        raise FileNotFoundError(storage.CUBE_CALIBRATION_PATH)
    snout_calibration = storage.load_snout_calibration()
    if snout_calibration is None:
        raise FileNotFoundError(storage.SNOUT_CALIBRATION_PATH)

    T_cube_tag = geometry.transform_map_from_json(cube_calibration)
    if 4 not in T_cube_tag:
        raise ValueError("Cube calibration does not contain face ID 4")
    p_snout_cube_mm = np.asarray(
        snout_calibration["p_snout_cube_mm"],
        dtype=np.float64,
    ).reshape(3)
    return intrinsics, T_cube_tag, p_snout_cube_mm


def session_paths(start_time: datetime) -> tuple[Path, Path]:
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT_DIR / f"{SESSION_PREFIX}_{timestamp}"
    return output_dir, output_dir / CSV_NAME


def show_guide(csv_path: Path, video_path: Path | None) -> None:
    video_text = (
        f"Saving clean rectified RGB video to: {video_path}\n"
        if video_path is not None
        else "Video recording is disabled.\n"
    )
    print(
        "\nCALIBRATED CYRANOSE POSE TRACKING\n\n"
        "The script records qualified tag poses, the jointly estimated cube pose,\n"
        "and the calibrated snout position in camera and desk coordinates.\n\n"
        "Desk coordinates are available only while qualified desk tag ID 6 is visible.\n"
        "Positions are stored in centimeters; Rodrigues rotations are stored in radians.\n\n"
        f"Recording immediately to: {csv_path}\n\n"
        f"{video_text}\n"
        "Press Space to finish recording.\n"
        "Close the window to stop and preserve all rows already written.\n"
    )


def run_tracking(save_video: bool = False) -> None:
    intrinsics, T_cube_tag, p_snout_cube_mm = load_calibration()
    camera_matrix = np.asarray(intrinsics["rectified_K"], dtype=np.float64)
    output_dir, csv_path = session_paths(datetime.now())
    video_path = output_dir / VIDEO_NAME if save_video else None

    video_writer: cv2.VideoWriter | None = None
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        with RectifiedColorCamera(intrinsics) as camera:
            output_dir.mkdir(parents=True, exist_ok=False)
            show_guide(csv_path, video_path)
            session_start = time.perf_counter()
            rows_written = 0

            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
                writer.writeheader()
                csv_file.flush()

                while True:
                    camera_frame = camera.read_frame()
                    if save_video and video_writer is None:
                        height, width = camera_frame.image.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
                        video_writer = cv2.VideoWriter(
                            str(video_path),
                            fourcc,
                            float(intrinsics["fps"]),
                            (width, height),
                        )
                        if not video_writer.isOpened():
                            raise RuntimeError(f"Cannot open video writer: {video_path}")

                    host_timestamp = datetime.now(timezone.utc).isoformat(
                        timespec="milliseconds"
                    )
                    elapsed_s = time.perf_counter() - session_start
                    observations = geometry.detect_tags(
                        camera_frame.image,
                        camera_matrix,
                    )
                    row, T_camera_cube, cube_rms, used_ids = build_csv_row(
                        host_timestamp,
                        elapsed_s,
                        camera_frame,
                        observations,
                        camera_matrix,
                        T_cube_tag,
                        p_snout_cube_mm,
                    )
                    writer.writerow(row)
                    if video_writer is not None:
                        video_writer.write(camera_frame.image)
                    csv_file.flush()
                    rows_written += 1

                    output = camera_frame.image.copy()
                    display.draw_tag_observations(output, observations)
                    desk_valid = bool(row[f"tag_{geometry.DESK_TAG_ID}_visible"])
                    lines = [
                        "LIVE RECORDING: CUBE POSE + CALIBRATED SNOUT",
                        "Space: finish | Close window: exit",
                        (
                            f"Frames {rows_written} | "
                            f"Desk {'valid' if desk_valid else 'unavailable'}"
                        ),
                    ]
                    if T_camera_cube is None:
                        lines.append(
                            "Cube pose unavailable | No qualified calibrated tag visible"
                        )
                    else:
                        axis_pixels = geometry.project_points(
                            geometry.cube_axis_points(),
                            T_camera_cube,
                            camera_matrix,
                        )
                        display.draw_axes(output, axis_pixels)
                        snout_point = p_snout_cube_mm
                        plane_point = geometry.snout_projection_on_id4(
                            snout_point,
                            T_cube_tag[4],
                        )
                        projected = geometry.project_points(
                            np.vstack([snout_point, plane_point]),
                            T_camera_cube,
                            camera_matrix,
                        )
                        display.draw_snout_geometry(
                            output,
                            projected[0],
                            projected[1],
                        )
                        lines.append(
                            f"Cube tags {','.join(str(tag_id) for tag_id in used_ids)} | "
                            f"RMS {cube_rms:.2f} px"
                        )
                    display.draw_text_lines(output, lines)
                    cv2.imshow(WINDOW_NAME, output)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(" "):
                        return
                    if display.window_was_closed(WINDOW_NAME):
                        raise SystemExit(0)
    finally:
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record calibrated tag, cube, and snout poses while showing the live view."
        )
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save clean rectified RGB frames as rectified_rgb.mp4.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_tracking(save_video=args.save_video)


if __name__ == "__main__":
    main()
