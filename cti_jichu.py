import cv2
import numpy as np
import pyrealsense2 as rs
import enum

# ------------ 参数设置 -------------
A4_REAL_WIDTH = 21.0  # A4纸宽度 (cm)
MIN_CONTOUR_AREA = 1200  # 最小轮廓面积
DEBUG_MODE = False
CALIBRATION_MODE = False  # 不再使用焦距法，改用深度值
contoutrtype = None

# ------------ 模式设置 -------------
class ContourType(enum.Enum):
    Nothing = 1
    Basic_squares = 2
    Basic_circle = 3
    Basic_triangle = 4

# ------------ RealSense 初始化 -------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

# 对齐深度到彩色图
align_to = rs.stream.color
align = rs.align(align_to)

# ------------ 检测A4纸函数 -------------
def detect_a4_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 59, 20
    )
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                return approx, mask
    return None, mask

# ------------ A4中心点距离获取函数 -------------
def get_center_depth(depth_frame, contour):
    if contour is None:
        return None
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    depth = depth_frame.get_distance(cX, cY)
    return depth * 100  # 转换为 cm

# ------------ 主循环 -------------
print("按以下键切换检测模式：")
print("1 - 正方形   2 - 圆形   3 - 三角形   ESC - 退出")

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())

    # 显示提示
    cv2.putText(color_image, "1:Square 2:Circle 3:Triangle", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('1'):
        contoutrtype = ContourType.Basic_squares
        print("切换到：正方形检测")
    elif key == ord('2'):
        contoutrtype = ContourType.Basic_circle
        print("切换到：圆形检测")
    elif key == ord('3'):
        contoutrtype = ContourType.Basic_triangle
        print("切换到：三角形检测")

    # 检测A4纸
    contour, mask = detect_a4_contour(color_image)
    text = "未识别到A4纸"
    if contour is not None:
        cv2.drawContours(color_image, [contour], -1, (255, 0, 0), 2)

        # 获取中心点深度
        distance_cm = get_center_depth(depth_frame, contour)
        if distance_cm:
            text = f"距离: {distance_cm:.2f} cm"

        # 后续形状识别略（可复用你已有的透视变换 + 形状识别函数）

    cv2.putText(color_image, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.imshow("RealSense A4 Detection", color_image)

# 清理
pipeline.stop()
cv2.destroyAllWindows()
