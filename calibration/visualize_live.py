from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from h264_video import H264VideoWriter

import display
import geometry
import storage
from camera import RectifiedColorCamera


WINDOW_NAME = "Cube Pose Live"
LIVE_DEMO_DIR = storage.DATA_DIR / "live_demos"


def default_video_path(show_snout: bool) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = "snout_live_demo" if show_snout else "cube_live_demo"
    return LIVE_DEMO_DIR / f"{prefix}_{timestamp}.mp4"


def show_live_guide(show_snout: bool, video_path: Path | None) -> None:
    recording_text = (
        f"\nThe annotated live view will be recorded to:\n{video_path}\n"
        if video_path is not None
        else "\nVideo recording is disabled.\n"
    )
    if show_snout:
        print(
            "\nLIVE DEMONSTRATION: CUBE POSE AND CALIBRATED SNOUT\n\n"
            "The cube pose is recomputed in every frame from the currently visible\n"
            "calibrated tags.\n\n"
            "The stored snout position is transformed using the live cube pose.\n"
            "It should remain rigidly attached to the physical snout as the cube moves.\n\n"
            "Magenta point: calibrated snout position\n"
            "Yellow segment: orthogonal connection to the calibrated ID 4 face\n"
            "Red X: right   Green Y: front   Blue Z: top\n\n"
            f"{recording_text}\n"
            "Press Space to finish the live demonstration.\n"
            "Close the window to exit."
        )
        return
    print(
        "\nLIVE DEMONSTRATION: CALIBRATED CUBE POSE\n\n"
        "The cube pose is recomputed in every frame from the currently visible\n"
        "calibrated tags.\n\n"
        "Move the cube through several viewpoints. The axes should remain rigidly\n"
        "attached to the cube without jumping, flipping, or changing orientation\n"
        "between visible faces.\n\n"
        "Red X: right   Green Y: front   Blue Z: top\n\n"
        f"{recording_text}\n"
        "Press Space to finish the live demonstration.\n"
        "Close the window to exit."
    )


def run_live(
    show_snout: bool,
    save_video: bool = False,
    video_path: Path | None = None,
) -> Path | None:
    intrinsics = storage.load_camera_intrinsics()
    if intrinsics is None:
        raise FileNotFoundError(storage.CAMERA_INTRINSICS_PATH)
    cube_calibration = storage.load_cube_calibration()
    if cube_calibration is None:
        raise FileNotFoundError(storage.CUBE_CALIBRATION_PATH)
    snout_calibration = storage.load_snout_calibration() if show_snout else None
    if show_snout and snout_calibration is None:
        raise FileNotFoundError(storage.SNOUT_CALIBRATION_PATH)

    if save_video:
        video_path = video_path or default_video_path(show_snout)
        video_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        video_path = None

    show_live_guide(show_snout, video_path)
    camera_matrix = np.asarray(intrinsics["rectified_K"], dtype=np.float64)
    T_cube_tag = geometry.transform_map_from_json(cube_calibration)
    video_writer: H264VideoWriter | None = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        with RectifiedColorCamera(intrinsics) as camera:
            while True:
                frame = camera.read()
                observations = geometry.detect_tags(frame, camera_matrix)
                T_camera_cube, pose_rms, used_ids = geometry.estimate_joint_cube_pose(
                    observations,
                    T_cube_tag,
                    camera_matrix,
                )

                output = frame.copy()
                display.draw_tag_observations(output, observations)
                title = (
                    "LIVE: CUBE POSE + CALIBRATED SNOUT"
                    if show_snout
                    else "LIVE: CALIBRATED CUBE POSE"
                )
                lines = [
                    title,
                    "Space: finish | Close window: exit",
                    "X red | Y green | Z blue",
                ]
                if video_path is not None:
                    lines.append("VIDEO RECORDING: annotated live view")
                if T_camera_cube is None:
                    lines.append(
                        "Live pose unavailable | No qualified calibrated tag visible"
                    )
                else:
                    axis_pixels = geometry.project_points(
                        geometry.cube_axis_points(),
                        T_camera_cube,
                        camera_matrix,
                    )
                    display.draw_axes(output, axis_pixels)
                    lines.append(
                        f"Tags {','.join(str(tag_id) for tag_id in used_ids)} | "
                        f"RMS {pose_rms:.2f} px"
                    )

                    if snout_calibration is not None:
                        snout_point = np.asarray(
                            snout_calibration["p_snout_cube_mm"],
                            dtype=np.float64,
                        )
                        plane_point = geometry.snout_projection_on_id4(
                            snout_point,
                            T_cube_tag[4],
                        )
                        projected = geometry.project_points(
                            np.vstack([snout_point, plane_point]),
                            T_camera_cube,
                            camera_matrix,
                        )
                        display.draw_snout_geometry(output, projected[0], projected[1])

                display.draw_text_lines(output, lines)
                if video_path is not None and video_writer is None:
                    height, width = output.shape[:2]
                    video_writer = H264VideoWriter(
                        video_path,
                        float(intrinsics["fps"]),
                        (width, height),
                    )
                if video_writer is not None:
                    video_writer.write(output)
                cv2.imshow(WINDOW_NAME, output)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    break
                if display.window_was_closed(WINDOW_NAME):
                    raise SystemExit(0)
    finally:
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
    if video_path is not None:
        print(f"Saved annotated live demonstration to: {video_path}")
    return video_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show the calibrated cube/snout pose in the live camera view."
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Record the annotated live view to calibration/data/live_demos.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_live(
        show_snout=storage.SNOUT_CALIBRATION_PATH.exists(),
        save_video=args.save_video,
    )


if __name__ == "__main__":
    main()
