import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


# -----------------------------
# Measurements
# -----------------------------

TAG_SIZE_M = 0.077
CUBE_SIDE_M = 0.077

DESK_TAG_ID = 6

# Cube coordinate system:
# +X = front / toward snout
# +Y = right
# +Z = top
FACE_CENTER_CUBE = {
    4: np.array([+CUBE_SIDE_M / 2, 0, 0]),  # front / toward snout
    0: np.array([-CUBE_SIDE_M / 2, 0, 0]),  # back
    3: np.array([0, +CUBE_SIDE_M / 2, 0]),  # right
    1: np.array([0, -CUBE_SIDE_M / 2, 0]),  # left
    2: np.array([0, 0, +CUBE_SIDE_M / 2]),  # top
    5: np.array([0, 0, -CUBE_SIDE_M / 2]),  # bottom
}

CYRANOSE_TAG_IDS = set(FACE_CENTER_CUBE.keys())

# Tag local axes expressed in cube coordinates.
# This is the part that may need adjustment if the projected snout dot looks wrong.
# Columns are: tag local x-axis, tag local y-axis, tag local z-axis in cube coords.
# Assumption: tag local +z points outward from the cube face.
R_CUBE_TAG = {
    # front face, outward +X
    4: np.column_stack([
        np.array([0, -1, 0]),   # tag +x
        np.array([0, 0, -1]),   # tag +y
        np.array([1, 0, 0]),    # tag +z/outward
    ]),

    # back face, outward -X
    0: np.column_stack([
        np.array([0, 1, 0]),
        np.array([0, 0, -1]),
        np.array([-1, 0, 0]),
    ]),

    # right face, outward +Y
    3: np.column_stack([
        np.array([1, 0, 0]),
        np.array([0, 0, -1]),
        np.array([0, 1, 0]),
    ]),

    # left face, outward -Y
    1: np.column_stack([
        np.array([-1, 0, 0]),
        np.array([0, 0, -1]),
        np.array([0, -1, 0]),
    ]),

    # top face, outward +Z
    2: np.column_stack([
        np.array([0, 1, 0]),
        np.array([-1, 0, 0]),
        np.array([0, 0, 1]),
    ]),

    # bottom face, outward -Z
    5: np.column_stack([
        np.array([0, 1, 0]),
        np.array([1, 0, 0]),
        np.array([0, 0, -1]),
    ]),
}

# Snout estimate in cube coordinates.
# Snout is 29 cm forward from the front face.
snout_x = 0.29 + CUBE_SIDE_M / 2

# Snout to center of top face was measured as 34.25 cm.
distance_snout_to_top_center = 0.3425
top_center_z = CUBE_SIDE_M / 2

horizontal_dx = snout_x
vertical_dz = np.sqrt(
    max(distance_snout_to_top_center**2 - horizontal_dx**2, 0)
)

# Assume snout is below the cube top.
snout_z = top_center_z - vertical_dz

# Assume snout is centered left/right.
SNOUT_CUBE = np.array([snout_x, 0.0, snout_z])

print("Estimated snout position in cube coordinates:")
print(f"x forward = {SNOUT_CUBE[0]:.4f} m")
print(f"y right   = {SNOUT_CUBE[1]:.4f} m")
print(f"z up      = {SNOUT_CUBE[2]:.4f} m")


# -----------------------------
# Camera setup
# -----------------------------

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

    return camera_matrix, distortion_coeffs


def tag_object_points(tag_size_m):
    half = tag_size_m / 2.0

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


def snout_in_tag_coordinates(tag_id):
    """
    Convert the snout point from cube coordinates into this tag face's
    local coordinate system.
    """
    face_center = FACE_CENTER_CUBE[tag_id]
    r_cube_tag = R_CUBE_TAG[tag_id]

    # p_cube = R_cube_tag @ p_tag + face_center
    # therefore:
    # p_tag = R_cube_tag.T @ (p_cube - face_center)
    return r_cube_tag.T @ (SNOUT_CUBE - face_center)


def main():
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"cyranose_pose_session_{session_timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cyranose_snout_pose.csv"

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
        "visible_cube_faces",
        "num_snout_estimates",
        "snout_cam_x_m",
        "snout_cam_y_m",
        "snout_cam_z_m",
        "snout_desk_x_m",
        "snout_desk_y_m",
        "snout_desk_z_m",
        "delta_from_baseline_cm",
        "delta_x_cm",
        "delta_y_cm",
        "delta_z_cm",
    ]

    session_start = time.perf_counter()
    recording = False
    baseline = None
    baseline_frame_name = None

    print("\nRunning Cyranose snout pose estimate.")
    print("This version works with 1+ visible cube face.")
    print("Press B to set baseline position.")
    print("Press R to start/stop recording CSV.")
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

                desk_transform_cam_desk = None
                snout_estimates_camera = []
                snout_estimates_image = []
                visible_cube_ids = []

                if ids is not None:
                    for tag_id, tag_corners in zip(ids.flatten(), corners):
                        tag_id = int(tag_id)

                        if tag_id not in CYRANOSE_TAG_IDS and tag_id != DESK_TAG_ID:
                            continue

                        object_points = tag_object_points(TAG_SIZE_M)
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

                        cv2.aruco.drawDetectedMarkers(
                            image,
                            [tag_corners],
                            np.array([[tag_id]], dtype=np.int32),
                        )

                        cv2.drawFrameAxes(
                            image,
                            camera_matrix,
                            distortion_coeffs,
                            rvec,
                            tvec,
                            TAG_SIZE_M * 0.45,
                        )

                        if tag_id == DESK_TAG_ID:
                            desk_transform_cam_desk = transform_cam_tag

                        if tag_id in CYRANOSE_TAG_IDS:
                            visible_cube_ids.append(tag_id)

                            p_snout_tag = snout_in_tag_coordinates(tag_id)

                            rotation_matrix, _ = cv2.Rodrigues(rvec)
                            p_snout_cam = (
                                rotation_matrix @ p_snout_tag
                                + tvec.reshape(3)
                            )

                            snout_estimates_camera.append(p_snout_cam)

                            projected, _ = cv2.projectPoints(
                                p_snout_tag.reshape(1, 3),
                                rvec,
                                tvec,
                                camera_matrix,
                                distortion_coeffs,
                            )

                            sx = int(projected[0, 0, 0])
                            sy = int(projected[0, 0, 1])
                            snout_estimates_image.append((sx, sy))

                            # Draw individual snout estimate from this face.
                            cv2.circle(
                                image,
                                (sx, sy),
                                6,
                                (255, 0, 255),
                                -1,
                            )

                            cv2.putText(
                                image,
                                f"snout est from ID {tag_id}",
                                (sx + 8, sy),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                (255, 0, 255),
                                1,
                            )

                visible_cube_ids = sorted(visible_cube_ids)

                snout_cam = None
                snout_desk = None
                active_position = None
                active_frame_name = None

                if snout_estimates_camera:
                    snout_cam = np.mean(
                        np.array(snout_estimates_camera),
                        axis=0,
                    )

                    # Draw average projected snout location.
                    if snout_estimates_image:
                        avg_px = int(
                            np.mean([p[0] for p in snout_estimates_image])
                        )
                        avg_py = int(
                            np.mean([p[1] for p in snout_estimates_image])
                        )

                        cv2.circle(
                            image,
                            (avg_px, avg_py),
                            10,
                            (0, 0, 255),
                            2,
                        )

                        cv2.putText(
                            image,
                            "AVG SNIFF POINT",
                            (avg_px + 12, avg_py),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 0, 255),
                            2,
                        )

                    if desk_transform_cam_desk is not None:
                        transform_desk_cam = np.linalg.inv(
                            desk_transform_cam_desk
                        )

                        snout_cam_h = np.array(
                            [
                                snout_cam[0],
                                snout_cam[1],
                                snout_cam[2],
                                1.0,
                            ]
                        )

                        snout_desk_h = transform_desk_cam @ snout_cam_h
                        snout_desk = snout_desk_h[:3]

                        active_position = snout_desk
                        active_frame_name = "desk"
                    else:
                        active_position = snout_cam
                        active_frame_name = "camera"

                delta_norm_cm = ""
                delta_x_cm = ""
                delta_y_cm = ""
                delta_z_cm = ""

                if baseline is not None and active_position is not None:
                    delta = active_position - baseline
                    delta_cm = delta * 100.0
                    delta_norm_cm = float(np.linalg.norm(delta_cm))
                    delta_x_cm = float(delta_cm[0])
                    delta_y_cm = float(delta_cm[1])
                    delta_z_cm = float(delta_cm[2])

                # Display status.
                cv2.putText(
                    image,
                    f"Visible cube faces: {visible_cube_ids}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )

                desk_text = (
                    "Desk tag 6 visible"
                    if desk_transform_cam_desk is not None
                    else "Desk tag 6 NOT visible"
                )
                desk_color = (
                    (0, 255, 0)
                    if desk_transform_cam_desk is not None
                    else (0, 0, 255)
                )

                cv2.putText(
                    image,
                    desk_text,
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    desk_color,
                    2,
                )

                if active_position is not None:
                    cv2.putText(
                        image,
                        (
                            f"Snout {active_frame_name} xyz m: "
                            f"{active_position[0]:.3f}, "
                            f"{active_position[1]:.3f}, "
                            f"{active_position[2]:.3f}"
                        ),
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                    )

                    if baseline is not None:
                        cv2.putText(
                            image,
                            (
                                f"Delta from baseline: "
                                f"{delta_norm_cm:.1f} cm "
                                f"(dx={delta_x_cm:.1f}, "
                                f"dy={delta_y_cm:.1f}, "
                                f"dz={delta_z_cm:.1f})"
                            ),
                            (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2,
                        )
                    else:
                        cv2.putText(
                            image,
                            "Press B to set baseline for cm movement check",
                            (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            2,
                        )

                else:
                    cv2.putText(
                        image,
                        "Need at least 1 cube face visible",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                    )

                record_text = (
                    "RECORDING"
                    if recording
                    else "not recording"
                )
                cv2.putText(
                    image,
                    f"{record_text} | B baseline | R record | Q quit",
                    (20, image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255) if recording else (255, 255, 255),
                    2,
                )

                if recording and active_position is not None:
                    writer.writerow(
                        {
                            "host_timestamp_utc": host_timestamp,
                            "elapsed_s": round(elapsed_s, 6),
                            "camera_timestamp_ms": round(
                                camera_timestamp_ms,
                                3,
                            ),
                            "frame_number": frame_number,
                            "visible_cube_faces": ";".join(
                                str(x) for x in visible_cube_ids
                            ),
                            "num_snout_estimates": len(
                                snout_estimates_camera
                            ),
                            "snout_cam_x_m": float(snout_cam[0])
                            if snout_cam is not None
                            else "",
                            "snout_cam_y_m": float(snout_cam[1])
                            if snout_cam is not None
                            else "",
                            "snout_cam_z_m": float(snout_cam[2])
                            if snout_cam is not None
                            else "",
                            "snout_desk_x_m": float(snout_desk[0])
                            if snout_desk is not None
                            else "",
                            "snout_desk_y_m": float(snout_desk[1])
                            if snout_desk is not None
                            else "",
                            "snout_desk_z_m": float(snout_desk[2])
                            if snout_desk is not None
                            else "",
                            "delta_from_baseline_cm": delta_norm_cm,
                            "delta_x_cm": delta_x_cm,
                            "delta_y_cm": delta_y_cm,
                            "delta_z_cm": delta_z_cm,
                        }
                    )
                    csv_file.flush()

                cv2.imshow("Cyranose Snout Pose Single Face", image)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("b"):
                    if active_position is not None:
                        baseline = active_position.copy()
                        baseline_frame_name = active_frame_name
                        print(
                            f"Baseline set in {baseline_frame_name} frame:",
                            np.round(baseline, 4),
                        )
                    else:
                        print("Cannot set baseline: no snout position.")

                elif key == ord("r"):
                    recording = not recording
                    print(
                        "Recording started."
                        if recording
                        else "Recording stopped."
                    )

                elif key == ord("q"):
                    break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    print("\nDone.")
    print(f"CSV saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()