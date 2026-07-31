import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


# Cube tags mounted on Cyranose.
CYRANOSE_TAG_IDS = {0, 1, 2, 3, 4, 5}

# Desk / paper reference tag.
DESK_TAG_ID = 6

# IMPORTANT:
# Measure from the left black border edge to the right black border edge.
# Put the measurement in meters.
#
# Example:
# 50 mm = 0.050
DEFAULT_TAG_SIZE_M = 0.050  # <-- CHANGE THIS TO YOUR MEASURED TAG SIZE

# If all tags are the same printed size, this is fine.
TAG_SIZE_M_BY_ID = {
    0: DEFAULT_TAG_SIZE_M,
    1: DEFAULT_TAG_SIZE_M,
    2: DEFAULT_TAG_SIZE_M,
    3: DEFAULT_TAG_SIZE_M,
    4: DEFAULT_TAG_SIZE_M,
    5: DEFAULT_TAG_SIZE_M,
    6: DEFAULT_TAG_SIZE_M,
}

TAG_DICTIONARY = cv2.aruco.DICT_APRILTAG_36h11

FRAME_WIDTH = 848
FRAME_HEIGHT = 480
FPS = 30


def make_camera_matrix_and_distortion(profile):
    color_stream = profile.get_stream(
        rs.stream.color
    ).as_video_stream_profile()

    intrinsics = color_stream.get_intrinsics()

    camera_matrix = np.array(
        [
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    distortion_coeffs = np.array(
        intrinsics.coeffs,
        dtype=np.float64,
    )

    print("Camera intrinsics:")
    print(camera_matrix)
    print("Distortion coefficients:")
    print(distortion_coeffs)

    return camera_matrix, distortion_coeffs


def tag_object_points(tag_size_m):
    half = tag_size_m / 2.0

    # Corners in tag-local coordinates.
    # Order matches OpenCV AprilTag/ArUco corner order:
    # top-left, top-right, bottom-right, bottom-left.
    return np.array(
        [
            [-half, -half, 0],
            [half, -half, 0],
            [half, half, 0],
            [-half, half, 0],
        ],
        dtype=np.float64,
    )


def pose_to_matrix(rvec, tvec):
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = tvec.reshape(3)

    return transform


def main():
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"apriltag_pose_session_{session_timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "apriltag_pose_tracking.csv"

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        rs.format.bgr8,
        FPS,
    )

    profile = pipeline.start(config)

    camera_matrix, distortion_coeffs = make_camera_matrix_and_distortion(
        profile
    )

    dictionary = cv2.aruco.getPredefinedDictionary(TAG_DICTIONARY)
    detector_parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(
        dictionary,
        detector_parameters,
    )

    csv_fields = [
        "host_timestamp_utc",
        "elapsed_s",
        "camera_timestamp_ms",
        "frame_number",
        "recording",
        "tag_id",
        "role",
        "tag_size_m",
        "t_cam_x_m",
        "t_cam_y_m",
        "t_cam_z_m",
        "rvec_x",
        "rvec_y",
        "rvec_z",
        "desk_tag_visible",
        "t_desk_x_m",
        "t_desk_y_m",
        "t_desk_z_m",
    ]

    session_start = time.perf_counter()
    recording = False

    print("\nAprilTag pose estimation started.")
    print("Press R to start/stop recording.")
    print("Press Q to quit.")
    print(f"Saving to: {csv_path.resolve()}\n")

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
            writer.writeheader()

            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                if not color_frame:
                    continue

                image = np.asanyarray(color_frame.get_data())
                grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                corners, ids, _ = detector.detectMarkers(grayscale)

                elapsed_s = time.perf_counter() - session_start
                host_timestamp = datetime.now(
                    timezone.utc
                ).isoformat(timespec="milliseconds")

                camera_timestamp_ms = color_frame.get_timestamp()
                frame_number = color_frame.get_frame_number()

                detected_poses = {}
                desk_transform_cam_to_tag = None

                if ids is not None:
                    for tag_id, tag_corners in zip(ids.flatten(), corners):
                        tag_id = int(tag_id)

                        if tag_id not in TAG_SIZE_M_BY_ID:
                            continue

                        tag_size_m = TAG_SIZE_M_BY_ID[tag_id]
                        object_points = tag_object_points(tag_size_m)
                        image_points = tag_corners.reshape(4, 2).astype(
                            np.float64
                        )

                        success, rvec, tvec = cv2.solvePnP(
                            object_points,
                            image_points,
                            camera_matrix,
                            distortion_coeffs,
                            flags=cv2.SOLVEPNP_ITERATIVE,
                        )

                        if not success:
                            continue

                        transform_cam_tag = pose_to_matrix(rvec, tvec)

                        detected_poses[tag_id] = {
                            "rvec": rvec,
                            "tvec": tvec,
                            "transform_cam_tag": transform_cam_tag,
                            "tag_size_m": tag_size_m,
                            "corners": tag_corners,
                        }

                        if tag_id == DESK_TAG_ID:
                            desk_transform_cam_to_tag = transform_cam_tag

                # If desk tag is visible, use it as the coordinate frame.
                # This lets us express cube tag positions relative to the desk tag.
                if desk_transform_cam_to_tag is not None:
                    transform_tag_to_cam = np.linalg.inv(
                        desk_transform_cam_to_tag
                    )
                else:
                    transform_tag_to_cam = None

                visible_ids = sorted(detected_poses.keys())

                for tag_id, pose in detected_poses.items():
                    rvec = pose["rvec"]
                    tvec = pose["tvec"]
                    tag_size_m = pose["tag_size_m"]

                    role = (
                        "desk_reference"
                        if tag_id == DESK_TAG_ID
                        else "cyranose_cube_face"
                        if tag_id in CYRANOSE_TAG_IDS
                        else "other"
                    )

                    # Draw tag outline and 3D axes.
                    cv2.aruco.drawDetectedMarkers(
                        image,
                        [pose["corners"]],
                        np.array([[tag_id]], dtype=np.int32),
                    )

                    cv2.drawFrameAxes(
                        image,
                        camera_matrix,
                        distortion_coeffs,
                        rvec,
                        tvec,
                        tag_size_m * 0.5,
                    )

                    t_cam = tvec.reshape(3)

                    t_desk_x = ""
                    t_desk_y = ""
                    t_desk_z = ""

                    if transform_tag_to_cam is not None:
                        transform_desk_tag = (
                            transform_tag_to_cam
                            @ pose["transform_cam_tag"]
                        )

                        t_desk = transform_desk_tag[:3, 3]
                        t_desk_x = float(t_desk[0])
                        t_desk_y = float(t_desk[1])
                        t_desk_z = float(t_desk[2])

                    if recording:
                        writer.writerow(
                            {
                                "host_timestamp_utc": host_timestamp,
                                "elapsed_s": round(elapsed_s, 6),
                                "camera_timestamp_ms": round(
                                    camera_timestamp_ms,
                                    3,
                                ),
                                "frame_number": frame_number,
                                "recording": recording,
                                "tag_id": tag_id,
                                "role": role,
                                "tag_size_m": tag_size_m,
                                "t_cam_x_m": float(t_cam[0]),
                                "t_cam_y_m": float(t_cam[1]),
                                "t_cam_z_m": float(t_cam[2]),
                                "rvec_x": float(rvec[0]),
                                "rvec_y": float(rvec[1]),
                                "rvec_z": float(rvec[2]),
                                "desk_tag_visible": (
                                    desk_transform_cam_to_tag
                                    is not None
                                ),
                                "t_desk_x_m": t_desk_x,
                                "t_desk_y_m": t_desk_y,
                                "t_desk_z_m": t_desk_z,
                            }
                        )

                if recording:
                    csv_file.flush()

                status_text = (
                    "RECORDING - press R to stop"
                    if recording
                    else "NOT RECORDING - press R to start"
                )

                cv2.putText(
                    image,
                    status_text,
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255) if recording else (255, 255, 255),
                    2,
                )

                cv2.putText(
                    image,
                    f"Visible tag IDs: {visible_ids}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                if DESK_TAG_ID in visible_ids:
                    desk_status = "Desk tag visible"
                else:
                    desk_status = "Desk tag NOT visible"

                cv2.putText(
                    image,
                    desk_status,
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0)
                    if DESK_TAG_ID in visible_ids
                    else (0, 0, 255),
                    2,
                )

                cv2.imshow("RealSense AprilTag Pose", image)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("r"):
                    recording = not recording

                    if recording:
                        print(f"[{elapsed_s:.2f}s] Recording started.")
                    else:
                        print(f"[{elapsed_s:.2f}s] Recording stopped.")

                elif key == ord("q"):
                    break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    print("\nPose tracking complete.")
    print(f"CSV saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()