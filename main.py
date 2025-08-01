import pyrealsense2 as rs
import numpy as np
import cv2
import time
import serial  

# 相机参数
HFOV = 87.0  # 水平视场角
VFOV = 58.0  # 垂直视场角
FRAME_WIDTH = 1280 #图像宽度
FRAME_HEIGHT = 720 #图像高度

def compute_angle_offset(center_x, center_y):
    offset_x = center_x - FRAME_WIDTH // 2  #x方向偏移量
    offset_y = center_y - FRAME_HEIGHT // 2 #y方向偏移量

    angle_x = (offset_x / FRAME_WIDTH) * HFOV  #x方向偏移角度  angle_x>0 说明云台要向右偏移
    angle_y = (offset_y / FRAME_HEIGHT) * VFOV #y方向偏移角度  angle_y>0 说明云台要向下偏移

    return angle_x, angle_y

def detect_a4_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 59, 20)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cv2.imshow("mask", mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 5000:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                return approx
    return None

def detect(color_image, depth_frame):
    # 图像中心点
    img_center_y = FRAME_HEIGHT // 2
    img_center_x = FRAME_WIDTH // 2
    cv2.circle(color_image, (img_center_x, img_center_y), 4, (0, 255, 255), -1)
    cv2.line(color_image, (img_center_x - 10, img_center_y), (img_center_x + 10, img_center_y), (0, 255, 255), 1)
    cv2.line(color_image, (img_center_x, img_center_y - 10), (img_center_x, img_center_y + 10), (0, 255, 255), 1)

    best_rect = detect_a4_contour(color_image)

    if best_rect is not None:
        x, y, w, h = cv2.boundingRect(best_rect)
        center_x = x + w // 2
        center_y = y + h // 2

        distance = depth_frame.get_distance(center_x, center_y)
        angle_x, angle_y = compute_angle_offset(center_x, center_y) #目标中心点相对于图像中心点的偏移角度 angle_x，angle_y
        print(f"中心坐标 center_x: {center_x}, center_y: {center_y}       偏移角度: X方向 {angle_x:.2f}°, Y方向 {angle_y:.2f}°")

        # 可视化目标
        cv2.drawContours(color_image, [best_rect], -1, (0, 255, 0), 2)
        # 画出目标中心点
        cv2.circle(color_image, (center_x, center_y), 4, (0, 0, 255), -1)
        # 连线
        cv2.line(color_image, (img_center_x, img_center_y), (center_x, center_y), (255, 0, 0), 2)
        cv2.putText(color_image, f"Angle X: {angle_x:.1f}, Y: {angle_y:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(color_image, f"Distance: {distance*100:.1f} cm",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
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
            cv2.putText(result, f"FPS: {fps:.0f}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # 显示图像
            cv2.imshow('image', result)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break

        cv2.destroyAllWindows()

    finally:
        pipeline.stop()
