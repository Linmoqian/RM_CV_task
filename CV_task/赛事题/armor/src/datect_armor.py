import cv2
from pathlib import Path
import numpy as np

def detect_armor(Gaussian, img, min_area, top_k, frame):
    """
    检测装甲板

    Returns:
        armor_points: 检测到的装甲板四个顶点 [左上, 左下, 右上, 右下]，未检测到返回 None
        armor_crop: 截取的装甲板区域，未检测到返回 None
    """
# ----- 轮廓检测 -----
    contours, _ = cv2.findContours(Gaussian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    contours = contours[:top_k]

    # ----- 灯条筛选与信息保存 -----
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

    # ----- 轮廓配对 -----
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

            # 绘制装甲板
            x = np.array([point1, point2, point4, point3], np.int32)
            box = x.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(img, [box], True, (0, 255, 0), 2)

            armor_center = [int((point1[0] + point2[0] + point3[0] + point4[0]) / 4),
                           int((point1[1] + point2[1] + point3[1] + point4[1]) / 4)]
            cv2.circle(img, armor_center, 5, (0, 0, 255), -1)
            cv2.line(img, point1, point4, (0, 255, 0), 2)
            cv2.line(img, point2, point3, (0, 255, 0), 2)

            # 截取装甲板区域
            all_points = np.array([point1, point2, point3, point4])
            x_min, y_min = np.min(all_points, axis=0).astype(int)
            x_max, y_max = np.max(all_points, axis=0).astype(int)

            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(frame.shape[1], x_max + margin)
            y_max = min(frame.shape[0], y_max + margin)

            armor_crop = frame[y_min:y_max, x_min:x_max].copy()

            # 返回装甲板顶点坐标和截取区域
            armor_points = [point1, point2, point3, point4]  # 左上、左下、右上、右下
            return armor_points, armor_crop

        except Exception as e:
            print(f"Error: {e}")
            return None, None

    # 如果没有检测到装甲板，返回 None
    return None, None