"""
装甲板 PnP 实时识别
功能：从视频中检测装甲板，求解其位姿和距离
"""

import cv2
import numpy as np
from pathlib import Path

# ============ 装甲板尺寸定义 ============
ARMOR_WIDTH = 140   # 小装甲板宽度（mm）
ARMOR_HEIGHT = 125  # 小装甲板高度（mm）
BIG_ARMOR_WIDTH = 230   # 大装甲板宽度（mm）
BIG_ARMOR_HEIGHT = 125  # 大装甲板高度（mm）

# ============ 相机标定参数 ============
# 内参矩阵 K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
# 需要实际标定获得，这里使用示例值
CAMERA_MATRIX = np.array([
    [1000, 0, 640],
    [0, 1000, 480],
    [0, 0, 1]
], dtype=np.float32)

# 畸变系数 [k1, k2, p1, p2, k3]
DIST_COEFFS = np.array([0, 0, 0, 0, 0], dtype=np.float32)

# ============ 其他参数 ============
min_area = 100      # 轮廓最小面积阈值
top_k = 4           # 只处理面积最大的前4个轮廓


# ============ 3D 点定义 ============
def get_armor_3d_points(width=ARMOR_WIDTH, height=ARMOR_HEIGHT):
    """获取装甲板四个顶点的3D坐标"""
    w2 = width / 2
    h2 = height / 2
    points_3d = np.array([
        [-w2, -h2, 0],  # 左上
        [ w2, -h2, 0],  # 右上
        [ w2,  h2, 0],  # 右下
        [-w2,  h2, 0]   # 左下
    ], dtype=np.float32)
    return points_3d


# ============ PnP 求解 ============
def solve_armor_pnp(points_2d, width=ARMOR_WIDTH, height=ARMOR_HEIGHT):
    """求解 PnP 问题，获取装甲板位姿"""
    points_3d = get_armor_3d_points(width, height)
    object_points = points_3d.reshape(-1, 1, 3)
    image_points = np.array(points_2d, dtype=np.float32).reshape(-1, 1, 2)

    success, rvec, tvec = cv2.solvePnP(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=CAMERA_MATRIX,
        distCoeffs=DIST_COEFFS,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return False, None, None, None

    distance = np.linalg.norm(tvec)
    return True, rvec, tvec, distance


def get_euler_angles(rvec):
    """将旋转向量转换为欧拉角（roll, pitch, yaw）"""
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = 0

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


# ============ 绘制 PnP 信息 ============
def draw_pnp_info(img, points_2d, rvec, tvec, distance):
    """在图像上绘制 PnP 结果"""
    # 绘制坐标轴
    axis_length = 100  # mm
    axis_3d = np.array([
        [0, 0, 0],
        [axis_length, 0, 0],  # X轴
        [0, axis_length, 0],  # Y轴
        [0, 0, axis_length]   # Z轴
    ], dtype=np.float32).reshape(-1, 1, 3)

    axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)
    axis_2d = axis_2d.astype(int)

    origin = tuple(axis_2d[0].ravel())
    cv2.line(img, origin, tuple(axis_2d[1].ravel()), (0, 0, 255), 3)  # X轴 - 红色
    cv2.line(img, origin, tuple(axis_2d[2].ravel()), (0, 255, 0), 3)  # Y轴 - 绿色
    cv2.line(img, origin, tuple(axis_2d[3].ravel()), (255, 0, 0), 3)  # Z轴 - 蓝色

    # 显示距离
    text = f"Distance: {distance/1000:.2f} m"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示欧拉角
    roll, pitch, yaw = get_euler_angles(rvec)
    angle_text = f"Yaw: {yaw:.1f} Pitch: {pitch:.1f} Roll: {roll:.1f}"
    cv2.putText(img, angle_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return img


# ============ 装甲板检测 ============
def detect_armor(frame):
    """检测装甲板并返回四个顶点坐标"""
    img = frame.copy()

    # 图像预处理
    b, g, r = cv2.split(img)
    _, b_bin = cv2.threshold(b, 168, 255, cv2.THRESH_BINARY)
    Gaussian = b_bin.copy()

    # 轮廓检测
    contours, _ = cv2.findContours(Gaussian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    contours = contours[:top_k]

    whole_h = img.shape[0]
    width_array, height_array, point_array, boxes_array, contour_areas = [], [], [], [], []

    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        if w == 0:
            continue

        width_array.append(w)
        height_array.append(h)
        point_array.append([x, y])
        contour_areas.append(cv2.contourArea(cont))

        rect = cv2.minAreaRect(cont)
        box = cv2.boxPoints(rect).astype(np.int32)
        boxes_array.append(box)
        cv2.polylines(img, [box], True, (0, 255, 0), 2)

    # 轮廓配对
    armor_points = None
    if len(width_array) >= 2:
        point_near = [0, 0]
        min_val = float('inf')

        for i in range(len(width_array) - 1):
            for j in range(i + 1, len(width_array)):
                value = abs(contour_areas[i] - contour_areas[j])
                if value < min_val:
                    min_val = value
                    point_near[0] = i
                    point_near[1] = j

        try:
            box1 = boxes_array[point_near[0]]
            box2 = boxes_array[point_near[1]]

            def get_top_bottom_points(box):
                sorted_box = sorted(box, key=lambda p: p[1])
                top_points = sorted(sorted_box[:2], key=lambda p: p[0])
                bottom_points = sorted(sorted_box[2:], key=lambda p: p[0])
                return top_points[0], top_points[1], bottom_points[0], bottom_points[1]

            tl1, tr1, bl1, br1 = get_top_bottom_points(box1)
            tl2, tr2, bl2, br2 = get_top_bottom_points(box2)

            center1 = np.mean(box1, axis=0)
            center2 = np.mean(box2, axis=0)

            if center1[0] < center2[0]:
                point1 = (tl1 + tr1) / 2
                point2 = (bl1 + br1) / 2
                point3 = (tl2 + tr2) / 2
                point4 = (bl2 + br2) / 2
            else:
                point1 = (tl2 + tr2) / 2
                point2 = (bl2 + br2) / 2
                point3 = (tl1 + tr1) / 2
                point4 = (bl1 + br1) / 2

            point1, point2, point3, point4 = map(lambda p: (int(p[0]), int(p[1])),
                                                [point1, point2, point3, point4])

            # 绘制装甲板边框
            x = np.array([point1, point2, point4, point3], np.int32)
            box = x.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(img, [box], True, (0, 255, 0), 2)

            # 绘制中心点
            armor_center = [int((point1[0] + point2[0] + point3[0] + point4[0]) / 4),
                           int((point1[1] + point2[1] + point3[1] + point4[1]) / 4)]
            cv2.circle(img, armor_center, 5, (0, 0, 255), -1)

            # 绘制对角线
            cv2.line(img, point1, point4, (0, 255, 0), 2)
            cv2.line(img, point2, point3, (0, 255, 0), 2)

            armor_points = [point1, point2, point3, point4]  # 左上、左下、右上、右下

        except Exception as e:
            print(f"Error: {e}")

    return img, armor_points


# ============ 主程序 ============
if __name__ == "__main__":
    # 视频路径
    project_root = Path(__file__).resolve().parents[1]
    video_path = project_root / "images" / "blue.mp4"
    print(f"Loading video from: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 检测装甲板
        result_img, armor_points = detect_armor(frame)

        # 如果检测到装甲板，求解 PnP
        if armor_points is not None:
            # 重排点顺序以匹配 PnP 格式：左上、右上、右下、左下
            points_2d = [armor_points[0], armor_points[2], armor_points[3], armor_points[1]]

            # 求解 PnP
            success, rvec, tvec, distance = solve_armor_pnp(points_2d)

            if success:
                # 绘制 PnP 信息
                result_img = draw_pnp_info(result_img, points_2d, rvec, tvec, distance)

        # 显示结果
        cv2.imshow("Armor Detection with PnP", result_img)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
