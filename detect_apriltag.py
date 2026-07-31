import cv2
import numpy as np
import pyrealsense2 as rs


# All six faces belong to the same Cyranose-mounted cube.
CYRANOSE_TAG_IDS = {0, 1, 2, 3, 4, 5}


def main() -> None:
    pipeline = rs.pipeline()
    config = rs.config()

    # Use the stream profile that already worked for your D405.
    config.enable_stream(
        rs.stream.color,
        848,
        480,
        rs.format.bgr8,
        30,
    )

    pipeline.start(config)

    # Change this only if you printed a different AprilTag family.
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_APRILTAG_36h11
    )

    detector_parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(
        dictionary,
        detector_parameters,
    )

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            corners, ids, rejected = detector.detectMarkers(grayscale)

            detected_cyranose_faces = []

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(image, corners, ids)

                for tag_id, tag_corners in zip(ids.flatten(), corners):
                    points = tag_corners.reshape(4, 2)

                    center_x = int(points[:, 0].mean())
                    center_y = int(points[:, 1].mean())

                    is_cyranose_face = tag_id in CYRANOSE_TAG_IDS

                    if is_cyranose_face:
                        detected_cyranose_faces.append(int(tag_id))
                        label = f"Cyranose face {tag_id}"
                    else:
                        label = f"Other tag {tag_id}"

                    cv2.circle(
                        image,
                        (center_x, center_y),
                        6,
                        (0, 255, 0),
                        -1,
                    )

                    cv2.putText(
                        image,
                        f"{label}: ({center_x}, {center_y})",
                        (center_x + 10, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )

                if detected_cyranose_faces:
                    print(
                        "Cyranose cube visible. "
                        f"Detected face IDs: {detected_cyranose_faces}"
                    )
                else:
                    print("Tags detected, but no Cyranose cube faces found.")

            else:
                print("No AprilTags detected.")

            cv2.imshow("RealSense AprilTag Detection", image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()