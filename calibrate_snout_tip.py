import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


TAG_SIZE_M = 0.077
CYRANOSE_TAG_IDS = {0, 1, 2, 3, 4, 5}
DESK_TAG_ID = 6

TAG_DICTIONARY = cv2.aruco.DICT_APRILTAG_36h11
FRAME_WIDTH = 848
FRAME_HEIGHT = 480
FPS = 30

OUTPUT_PATH = Path("snout_calibration.json")


clicked_point = None


def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"Clicked snout pixel: ({x}, {y})")


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


def get_camera_matrix_and_distortion(color_intrinsics):
    camera_matrix = np.array(
        [
            [color_intrinsics.fx, 0, color_intrinsics.ppx],
            [0, color_intrinsics.fy, color_intrinsics.ppy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    distortion_coeffs = np.array(
        color_intrinsics.coeffs,
        dtype=np.float64,
    )

    return camera_matrix, distortion_coeffs


def get_depth_median(depth_frame, x, y, radius=3):
    depths = []

    width = depth_frame.get_width()
    height = depth_frame.get_height()

    for yy in range(max(0, y - radius), min(height, y + radius + 1)):
        for xx in range(max(0, x - radius), min(width, x + radius + 1)):
            d = depth_frame.get_distance(xx, yy)
            if d > 0:
                depths.append(d)

    if not depths:
        return None

    return float(np.median(depths))


def main():
    global clicked_point

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        rs.format.bgr8,
        FPS,
    )

    config.enable_stream(
        rs.stream.depth,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        rs.format.z16,
        FPS,
    )

    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)

    dictionary = cv2.aruco.getPredefinedDictionary(TAG_DICTIONARY)
    detector_parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_parameters)

    cv2.namedWindow("Click yellow snout tip")
    cv2.setMouseCallback("Click yellow snout tip", mouse_callback)

    calibration = {}

    print("\nSnout calibration started.")
    print("Put the yellow snout tip clearly in view.")
    print("Make sure at least one cube tag is visible.")
    print("Click exactly on the yellow snout tip.")
    print("Press S to save calibration.")
    print("Press Q to quit.\n")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            color_intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
            camera_matrix, distortion_coeffs = get_camera_matrix_and_distortion(
                color_intrinsics
            )

            corners, ids, _ = detector.detectMarkers(grayscale)

            detected_tag_transforms = {}

            if ids is not None:
                for tag_id, tag_corners in zip(ids.flatten(), corners):
                    tag_id = int(tag_id)

                    if tag_id not in CYRANOSE_TAG_IDS:
                        continue

                    object_points = tag_object_points(TAG_SIZE_M)
                    image_points = tag_corners.reshape(4, 2).astype(np.float64)

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
                    detected_tag_transforms[tag_id] = transform_cam_tag

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

            if clicked_point is not None:
                x, y = clicked_point
                cv2.circle(image, (x, y), 8, (0, 0, 255), -1)
                cv2.putText(
                    image,
                    "clicked snout tip",
                    (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                )

                depth_m = get_depth_median(depth_frame, x, y, radius=4)

                if depth_m is not None:
                    snout_cam = np.array(
                        rs.rs2_deproject_pixel_to_point(
                            color_intrinsics,
                            [x, y],
                            depth_m,
                        ),
                        dtype=np.float64,
                    )

                    snout_cam_h = np.array(
                        [snout_cam[0], snout_cam[1], snout_cam[2], 1.0],
                        dtype=np.float64,
                    )

                    for tag_id, transform_cam_tag in detected_tag_transforms.items():
                        transform_tag_cam = np.linalg.inv(transform_cam_tag)
                        snout_tag_h = transform_tag_cam @ snout_cam_h
                        snout_tag = snout_tag_h[:3]

                        calibration[str(tag_id)] = {
                            "snout_in_tag_frame_m": snout_tag.tolist(),
                            "clicked_pixel": [int(x), int(y)],
                            "depth_m": float(depth_m),
                        }

                    cv2.putText(
                        image,
                        f"Calibrated visible faces: {sorted(calibration.keys())}",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                    cv2.putText(
                        image,
                        f"Depth at click: {depth_m:.3f} m",
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        image,
                        "No valid depth at clicked point",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

            cv2.putText(
                image,
                "Click snout tip | S save | Q quit",
                (20, image.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Click yellow snout tip", image)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                with OUTPUT_PATH.open("w", encoding="utf-8") as f:
                    json.dump(calibration, f, indent=2)

                print(f"\nSaved calibration to {OUTPUT_PATH.resolve()}")
                print(json.dumps(calibration, indent=2))

            elif key == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    print("\nDone.")


if __name__ == "__main__":
    main()