import cv2
import numpy as np
import pyrealsense2 as rs


def main() -> None:
    pipeline = rs.pipeline()
    config = rs.config()

    # Ask the RealSense for a normal RGB video stream.
    config.enable_stream(
        rs.stream.color,
        640,
        480,
        rs.format.bgr8,
        30,
    )

    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())

            cv2.imshow("RealSense Color Feed", image)

            # Press q to close.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()