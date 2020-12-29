import pyrealsense2 as rs
import numpy as np
import cv2


def main():
    # RealSense L515 Setting
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    align_to = rs.stream.color
    align = rs.align(align_to)

    # Start streaming
    pipeline.start(config)

    window_name = "RealSense Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    resize_factor = 0.3

    # Main Loop
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            images = np.hstack((color_image, depth_colormap))
            height = images.shape[0]
            width = images.shape[1]
            images_resized = cv2.resize(images, (int(width * resize_factor), int(height * resize_factor)))

            cv2.imshow(window_name, images_resized)
            key = cv2.waitKey(1)
            if key != -1:
                pipeline.stop()
    finally:
        pipeline.stop()

if __name__ == '__main__':
    main()
