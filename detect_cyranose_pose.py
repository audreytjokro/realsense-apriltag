import time
from datetime import datetime, timezone

import cv2
import numpy as np
import pyrealsense2 as rs


# -----------------------------
# Your measurements
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

# Snout is 29 cm forward from the front cube face.
# Front face center is at +CUBE_SIDE/2, so x = 0.29 + CUBE_SIDE/2.
snout_x = 0.29 + CUBE_SIDE_M / 2

# You measured snout to center of top face as 34.25 cm.
# Estimate vertical offset from that.
distance_snout_to_top_center = 0.3425
top_center_x = 0.0
top_center_z = CUBE_SIDE_M / 2

horizontal_dx = snout_x - top_center_x
vertical_dz = np.sqrt(
    max(distance_snout_to_top_center**2 - horizontal_dx**2, 0)
)

# Assume snout is below the top face center.
snout_z = top_center_z - vertical_dz

# Assume snout is centered left/right.
SNOUT_CUBE = np.array([snout_x, 0.0, snout_z])

print("Estimated snout position in cube coordinates:")
print(f"x forward = {SNOUT_CUBE[0]:.4f} m")
print(f"y right   = {SNOUT_CUBE[1]:.4f} m")
print(f"z up      = {SNOUT_CUBE[2]:.4f} m")


# -----------------------------
# RealSense / AprilTag setup
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


def estimate_rigid_transform(points_cube, points_camera):
    """
    Estimate R, t such that:
        point_camera ≈ R @ point_cube + t
    """
    cube_mean = points_cube.mean(axis=0)
    cam_mean = points_camera.mean(axis=0)

    cube_centered = points_cube - cube_mean
    cam_centered = points_camera - cam_mean

    H = cube_centered.T @ cam_centered
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = cam_mean - R @ cube_mean

    return R, t


def main():
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

    print("\nRunning Cyranose snout pose estimate.")
    print("Need desk tag ID 6 visible.")
    print("Need at least 3 cube faces visible for this first version.")
    print("Press Q to quit.\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = detector.detectMarkers(grayscale)

            detected_tag_centers_camera = {}
            desk_transform_cam_desk = None

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

                    detected_tag_centers_camera[tag_id] = tvec.reshape(3)

                    if tag_id == DESK_TAG_ID:
                        desk_transform_cam_desk = transform_cam_tag

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
                        TAG_SIZE_M * 0.5,
                    )

            visible_cube_ids = sorted(
                tag_id
                for tag_id in detected_tag_centers_camera
                if tag_id in CYRANOSE_TAG_IDS
            )

            snout_camera = None
            snout_desk = None

            if len(visible_cube_ids) >= 3:
                points_cube = np.array(
                    [FACE_CENTER_CUBE[tag_id] for tag_id in visible_cube_ids]
                )

                points_camera = np.array(
                    [
                        detected_tag_centers_camera[tag_id]
                        for tag_id in visible_cube_ids
                    ]
                )

                R_cube_to_cam, t_cube_to_cam = estimate_rigid_transform(
                    points_cube,
                    points_camera,
                )

                snout_camera = R_cube_to_cam @ SNOUT_CUBE + t_cube_to_cam

                if desk_transform_cam_desk is not None:
                    transform_desk_cam = np.linalg.inv(
                        desk_transform_cam_desk
                    )

                    snout_camera_h = np.array(
                        [
                            snout_camera[0],
                            snout_camera[1],
                            snout_camera[2],
                            1.0,
                        ]
                    )

                    snout_desk_h = transform_desk_cam @ snout_camera_h
                    snout_desk = snout_desk_h[:3]

            cv2.putText(
                image,
                f"Visible cube faces: {visible_cube_ids}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            if desk_transform_cam_desk is not None:
                desk_text = "Desk tag 6 visible"
                desk_color = (0, 255, 0)
            else:
                desk_text = "Desk tag 6 NOT visible"
                desk_color = (0, 0, 255)

            cv2.putText(
                image,
                desk_text,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                desk_color,
                2,
            )

            if snout_camera is not None:
                cv2.putText(
                    image,
                    (
                        "Snout in camera frame: "
                        f"x={snout_camera[0]:.3f}, "
                        f"y={snout_camera[1]:.3f}, "
                        f"z={snout_camera[2]:.3f} m"
                    ),
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

                print(
                    "Snout camera xyz:",
                    np.round(snout_camera, 4),
                    end="",
                )

                if snout_desk is not None:
                    print(
                        " | snout desk xyz:",
                        np.round(snout_desk, 4),
                    )
                else:
                    print(" | desk coords unavailable")

            else:
                cv2.putText(
                    image,
                    "Need >=3 cube faces for snout estimate",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Cyranose Snout Pose", image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()