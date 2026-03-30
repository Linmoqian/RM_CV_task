# 装甲板识别程序
# 功能：从视频中检测蓝色装甲板，绘制轮廓和四边形边框

import cv2
from pathlib import Path
import numpy as np
from display import merge_display_images
from datect_armor import detect_armor
from pnp import solve_armor_pnp, draw_pnp_info
# ============ 配置参数 ============
project_root = Path(__file__).resolve().parents[1]
video_path = project_root / "images" / "blue.mp4"
print(f"Loading video from: {video_path}")

min_area = 100      # 灯条轮廓最小面积阈值
top_k = 4           # 只处理面积最大的前4个轮廓

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = frame.copy()
    armor_crop = None #截取装甲板区域

    b, g, r = cv2.split(img)

    # 蓝色通道二值化
    _, b_bin = cv2.threshold(b, 168, 255, cv2.THRESH_BINARY)
    Gaussian = b_bin.copy()

    # 检测装甲板，获取顶点坐标和截取区域
    armor_points, armor_crop = detect_armor(Gaussian, img, min_area, top_k, frame)

    # 如果检测到装甲板，计算距离并显示
    if armor_points is not None:
        # 调整顶点顺序以匹配 PnP 格式：左上、右上、右下、左下
        points_2d = [armor_points[0], armor_points[2], armor_points[3], armor_points[1]]

        # 求解 PnP
        success, rvec, tvec, distance = solve_armor_pnp(points_2d)

        if success:
            # 绘制 PnP 信息（距离、角度、坐标轴）
            img = draw_pnp_info(img, points_2d, rvec, tvec, distance)

    # ----- 合并并显示结果 -----
    merged_img = merge_display_images(Gaussian, img, armor_crop)
    cv2.imshow("Armor Detection", merged_img)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
