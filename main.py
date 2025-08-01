import pyrealsense2 as rs
import numpy as np
import cv2
import time

# 相机参数
HFOV = 87.0  # 水平视场角（单位：度）
VFOV = 58.0  # 垂直视场角（单位：度）
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

def compute_angle_offset(center_x, center_y):
    offset_x = center_x - FRAME_WIDTH // 2
    offset_y = center_y - FRAME_HEIGHT // 2

    angle_x = (offset_x / FRAME_WIDTH) * HFOV
    angle_y = (offset_y / FRAME_HEIGHT) * VFOV

    return angle_x, angle_y

def detect_outer_rectangle(gray_img):
    blurred = cv2.GaussianBlur(gray_img, (7, 7), 1.5)
    edges = cv2.Canny(blurred, 50, 150)
    cv2.imshow("Edges", edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_rect = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5000:
            continue

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(approx)
            aspect_ratio = float(rect_w) / rect_h

            if 0.5 < aspect_ratio < 2.0 and area > max_area:
                max_area = area
                best_rect = approx

    if best_rect is not None:
        x, y, w, h = cv2.boundingRect(best_rect)
        return best_rect, x, y, w, h
    else:
        return None, None, None, None, None

def detect(color_image, depth_frame):
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    best_rect, x, y, w, h = detect_outer_rectangle(gray)

    # 图像中心点（黄色十字）
    img_center_x = FRAME_WIDTH // 2
    img_center_y = FRAME_HEIGHT // 2
    cv2.circle(color_image, (img_center_x, img_center_y), 4, (0, 255, 255), -1)
    cv2.line(color_image, (img_center_x - 10, img_center_y), (img_center_x + 10, img_center_y), (0, 255, 255), 1)
    cv2.line(color_image, (img_center_x, img_center_y - 10), (img_center_x, img_center_y + 10), (0, 255, 255), 1)

    if best_rect is not None and x is not None:
        center_x = x + w // 2
        center_y = y + h // 2

        distance = depth_frame.get_distance(center_x, center_y)
        angle_x, angle_y = compute_angle_offset(center_x, center_y)
        print(f"中心坐标center_x: {center_x}, center_y: {center_y}       偏移角度: X方向偏移{angle_x:.2f}, y方向偏移{angle_y:.2f}")

        # 可视化目标
        cv2.drawContours(color_image, [best_rect], -1, (0, 255, 0), 4)
        cv2.circle(color_image, (center_x, center_y), 2, (0, 0, 255), -1)
        cv2.putText(color_image, f"Angle X: {angle_x:.1f}, Y: {angle_y:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return color_image


if __name__ == "__main__":
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, 30)

    pipeline.start(config)

    try:
        prev_time = time.time()

        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            result = detect(color_image, depth_frame)

            # FPS 显示
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time
            cv2.putText(result, f"FPS: {fps:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # 显示图像
            #cv2.namedWindow('RealSense View', cv2.WINDOW_NORMAL)
            #cv2.resizeWindow('RealSense View', 960, 540)
            cv2.imshow('RealSense View', result)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break

        cv2.destroyAllWindows()

    finally:
        pipeline.stop()
