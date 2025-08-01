import cv2
import numpy as np
import enum

DEBUG_MODE = False  # 调试模式显示中间过程
CALIBRATION_MODE = True  # 标定模式

A4_REAL_WIDTH = 21.0  # A4纸实际宽�?(cm)
MIN_CONTOUR_AREA = 1200  # 最小轮廓面积阈值（过滤噪声�?
FOCAL_length = 257.14  # 摄像头焦�?

class ContourType(enum.Enum):
    Nothing = 1,
    Basic_squares = 2,
    Basic_circle = 3,
    Basic_triangle = 4,
    Extension = 5

contoutrtype = ContourType.Nothing

# 创建视频捕获对象
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

### 标定函数
def calibrate_camera(img, known_distance, known_width):
    """摄像头标定：计算焦距f"""
    frame = img
    contour, _ = detect_a4_contour(frame)
    if contour is None:
        print("标定失败：未检测到A4�?")
        return None
    if DEBUG_MODE:
        print(f"contour:{contour}")
    # 计算像素宽度
    pixel_width = calculate_pixel_width(contour)
    # 计算焦距f = (像素宽度 * 已知距离) / 实际宽度
    focal_length = (pixel_width * known_distance) / known_width
    print(f"标定完成: 焦距f={focal_length:.2f}像素")
    return focal_length

def calculate_pixel_width(contour):
    """使用最小外接矩形计算短�?"""
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    distance = min(width, height)
    return distance  # 直接返回短边长度

### a4边框识别和测�?
def detect_a4_contour(image):
    """检测A4纸轮�?"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 自适应阈值处�? - 增强黑色边框
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  
        cv2.THRESH_BINARY_INV, 59, 20
    )
    thresh_copy = thresh.copy()
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if DEBUG_MODE:
        print(f"检测到 {len(contours)} 个轮�?")
    # 筛选A4纸轮廓（最大轮�?+四边形判断）
    if contours:
        # 按面积降序排�?
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if DEBUG_MODE:
            print(f"检测到contours  {len(contours)} 个轮�?")
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if DEBUG_MODE:
                print(f"检测到 A4 轮廓 {area:.2f} 面积")
            if area < MIN_CONTOUR_AREA:
                continue

            # 多边形逼近
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if DEBUG_MODE:
                print(f"len(approx){len(approx)}")
            # 四边形检�?
            if len(approx) == 4:
                if DEBUG_MODE:
                    print(f"approx{approx}")
                return approx, thresh_copy
    return None, thresh_copy

def calculate_distance(focal_length, pixel_width):
    """计算目标物距离D"""
    return (focal_length * A4_REAL_WIDTH) / pixel_width

### 透视变换
def perspective_transform(image, contour):
    """将A4纸区域透视变换为标准矩�?"""
    # 将轮廓点排序为（左上、右上、右下、左下）
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # 计算轮廓点中�?
    center = np.mean(pts, axis=0)

    # 根据点与中心的相对位置排�?
    for point in pts:
        if point[0] < center[0] and point[1] < center[1]:
            rect[0] = point  # 左上
        elif point[0] > center[0] and point[1] < center[1]:
            rect[1] = point  # 右上
        elif point[0]> center[0] and point[1] > center[1]:
            rect[2] = point  # 右下
        else:
            rect[3] = point  # 左下

    # 计算目标矩形尺寸（保持A4比例�?
    width = max(
        np.linalg.norm(rect[0] - rect[1]),
        np.linalg.norm(rect[2] - rect[3])
    )
    height = max(
        np.linalg.norm(rect[0] - rect[3]),
        np.linalg.norm(rect[1] - rect[2])
    )

    # 创建目标�?
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)

    # 应用透视变换
    warped = cv2.warpPerspective(image, M, (int(width), int(height)))

    return warped

### 形状检测函�?
def detect_squares_in_a4(warped):
    inverted_gray = cv2.bitwise_not(warped)
    _, binary_img = cv2.threshold(inverted_gray, 160, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    squares = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4 and is_square(approx):
            side_length = calculate_side_length(approx)
            if side_length < 2:
                continue
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            squares.append({
                "contour": approx,
                "side_length": side_length,
                "center": (cX, cY)
            })
    if len(squares) == 1:
        largest_square = max(squares, key=lambda x: x["side_length"])
        pixel_diameter = largest_square["side_length"]
        actual_diameter = (pixel_diameter / warped.shape[1]) * A4_REAL_WIDTH
        return warped, actual_diameter
    return warped, None

def is_square(contour, angle_threshold=15, aspect_threshold=0.15):
    """验证是否为正方形（基于角度和边长比例�?"""
    sides = []
    for i in range(4):
        pt1 = contour[i][0]
        pt2 = contour[(i+1)%4][0]
        sides.append(np.linalg.norm(pt2 - pt1))
    max_side, min_side = max(sides), min(sides)
    aspect_ratio = abs(max_side - min_side) / ((max_side + min_side)/2)
    if aspect_ratio > aspect_threshold:
        return False
    for i in range(4):
        pt1 = contour[i][0]
        pt2 = contour[(i+1)%4][0]
        pt3 = contour[(i+2)%4][0]
        angle = calculate_angle(pt1, pt2, pt3)
        if not (85 <= angle <= 95):
            return False
    return True

def calculate_angle(p1, p2, p3):
    """计算三点形成的角�?"""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(cos_theta))

def calculate_side_length(contour):
    """计算正方形边长（像素单位�?"""
    sides = []
    for i in range(4):
        pt1 = contour[i][0]
        pt2 = contour[(i+1)%4][0]
        sides.append(np.linalg.norm(pt2 - pt1))
    avg_side = np.mean(sides)
    rect = cv2.minAreaRect(contour)
    min_side = min(rect[1])
    return (avg_side + min_side) / 2

def detect_triangles_in_a4(warped):
    """检测三角形"""
    inverted = cv2.bitwise_not(warped)
    _, binary_img = cv2.threshold(inverted, 160, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    triangles = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 3 and is_equilateral_triangle(approx):
            side_length = calculate_triangle_side(approx)
            triangles.append({
                "contour": approx,
                "side_length": side_length,
            })
    if len(triangles) == 1:
        largest_triangle = max(triangles, key=lambda t: cv2.contourArea(t["contour"]))
        pixel_side = largest_triangle["side_length"]
        cm_side = (pixel_side / warped.shape[1]) * A4_REAL_WIDTH
        return warped, cm_side
    return warped, None

def is_equilateral_triangle(contour, angle_threshold=15, side_threshold=0.15):
    """验证等边三角形（基于角度和边长）"""
    points = contour.reshape(3, 2)
    sides = [
        np.linalg.norm(points[1] - points[0]),
        np.linalg.norm(points[2] - points[1]),
        np.linalg.norm(points[0] - points[2])
    ]
    max_side, min_side = max(sides), min(sides)
    aspect_ratio = abs(max_side - min_side) / ((max_side + min_side)/2)
    if aspect_ratio > side_threshold:
        return False
    for i in range(3):
        a, b, c = points[i], points[(i+1)%3], points[(i+2)%3]
        angle = calculate_angle(a, b, c)
        if not (55 <= angle <= 65):
            return False
    return True

def calculate_triangle_side(contour):
    """计算三角形边长（三种方法融合�?"""
    points = contour.reshape(3, 2)
    sides_direct = [
        np.linalg.norm(points[1] - points[0]),
        np.linalg.norm(points[2] - points[1]),
        np.linalg.norm(points[0] - points[2])
    ]
    avg_side = np.mean(sides_direct)
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    circum_radius = radius
    theoretical_side = circum_radius * np.sqrt(3)
    area = cv2.contourArea(contour)
    area_side = (4 * area / np.sqrt(3)) ** 0.5
    return (avg_side * 0.6 + theoretical_side * 0.3 + area_side * 0.1)

def detect_circle_in_a4(warped):
    """检测圆�?"""
    inverted_gray = cv2.bitwise_not(warped)
    _, binary_img = cv2.threshold(inverted_gray, 160, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    selected_contours = []
    MIN_AREA = 30
    MAX_AREA = 6000
    MIN_CIRCULARITY = 0.7
    MAX_CIRCULARITY = 1.3

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if (MIN_AREA <= area <= MAX_AREA and
            MIN_CIRCULARITY <= circularity <= MAX_CIRCULARITY):
            (x, y), radius = cv2.minEnclosingCircle(contour)
            diameter = 2 * radius
            pixel_diameter = diameter * circularity
            actual_diameter = (pixel_diameter / warped.shape[1]) * A4_REAL_WIDTH
            return warped, actual_diameter
    return warped, None

### 主循�?
print("程序启动 - 按以下键选择检测模�?:")
print("1: 正方形检�?")
print("2: 圆形检�?")
print("3: 三角形检�?")
print("c: 进入校准模式")
print("ESC: 退出程�?")

calibration_active = False
calibration_distance = 150  # 默认校准距离150cm

while True:
    # 读取图像
    ret, img_cv2 = cap.read()
    if not ret:
        print("无法获取摄像头图�?")
        break
    
    # 显示操作提示
    cv2.putText(img_cv2, "1:Square 2:Circle 3:Triangle c:Calibrate", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 检测键盘输�?
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC键退�?
        break
    elif key == ord('1'):
        contoutrtype = ContourType.Basic_squares
        calibration_active = False
        print("切换到正方形检测模�?")
    elif key == ord('2'):
        contoutrtype = ContourType.Basic_circle
        calibration_active = False
        print("切换到圆形检测模�?")
    elif key == ord('3'):
        contoutrtype = ContourType.Basic_triangle
        calibration_active = False
        print("切换到三角形检测模�?")
    elif key == ord('c'):
        calibration_active = True
        print("进入校准模式，请将A4纸放置在距离摄像�?150cm�?")
    
    # 处理图像
    text = "Noting"
    
    if calibration_active:
        # 执行校准
        focal_length = calibrate_camera(img_cv2, known_distance=calibration_distance, known_width=A4_REAL_WIDTH)
        if focal_length is not None:
            FOCAL_length = focal_length
            print(f"校准完成，新焦距: {FOCAL_length:.2f}")
            calibration_active = False
            text = f"Calibrated! New Focal: {FOCAL_length:.2f}"
        else:
            text = "Calibration Failed - No A4 detected"
    elif contoutrtype != ContourType.Nothing:
        # 检测A4纸轮�?
        contour, threth__ = detect_a4_contour(img_cv2)
        if contour is not None:
            imgcv2_copy = img_cv2.copy()
            threth__copy = threth__.copy()

            # 计算像素宽度和距�?
            pixel_width = calculate_pixel_width(contour)
            distance = calculate_distance(FOCAL_length, pixel_width)
            
            # 透视变换
            warped = perspective_transform(threth__copy, contour)
            warped_copy = warped.copy()

            # 根据模式检测不同形�?
            if contoutrtype == ContourType.Basic_squares:
                warped_img, actual_diameter = detect_squares_in_a4(warped_copy)
                if actual_diameter is not None:
                    print(f"正方形边�?: {actual_diameter:.2f} cm")
                    text = f"Base,D:{distance:.2f},squares,diameter:{actual_diameter:.2f}"
            elif contoutrtype == ContourType.Basic_triangle:
                warped_img, actual_diameter = detect_triangles_in_a4(warped_copy)
                if actual_diameter is not None:
                    print(f"三角形边�?: {actual_diameter:.2f} cm")
                    text = f"Base,D:{distance:.2f},triangles,diameter:{actual_diameter:.2f}"
            elif contoutrtype == ContourType.Basic_circle:
                warped_img, actual_diameter = detect_circle_in_a4(warped_copy)
                if actual_diameter is not None:
                    print(f"圆形直径: {actual_diameter:.2f} cm")
                    text = f"Base,D:{distance:.2f},circle,diameter:{actual_diameter:.2f}"
    
    # 显示结果
    cv2.putText(img_cv2, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow("Shape Detection", img_cv2)

# 释放资源
cap.release()
cv2.destroyAllWindows()