"""
鼠标框选图片程序
功能：
1. 鼠标框选区域
2. 拖动时显示框线和像素信息
3. 完成后显示框选区域
4. 输出框中心坐标
"""

import cv2
import numpy as np
import os

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, '..', '..', 'images', '修喵.png')

# 读取图片
image = cv2.imread(image_path)
if image is None:
    print("\033[31m错误：无法读取图片\033[0m")
    exit(1)

# 全局变量
drawing = False
start_point = (-1, -1)
end_point = (-1, -1)
roi_selected = False

def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数"""
    global drawing, start_point, end_point, roi_selected

    # 限制坐标在图片范围内
    h, w = image.shape[:2]
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))

    if event == cv2.EVENT_LBUTTONDOWN:
        # 左键按下，开始绘制
        drawing = True
        start_point = (x, y)
        end_point = (x, y)
        roi_selected = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # 更新终点
            end_point = (x, y)

            # 复制原图进行绘制
            display = image.copy()

            # 绘制矩形框
            cv2.rectangle(display, start_point, end_point, (0, 255, 0), 2)

            # 获取当前像素RGB值 (BGR转RGB显示)
            b, g, r = image[y, x]
            pixel_info = f"Pos:({x},{y}) RGB:({r},{g},{b})"

            # 显示像素信息
            cv2.putText(display, pixel_info, (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            cv2.imshow('Select ROI', display)

    elif event == cv2.EVENT_LBUTTONUP:
        # 左键释放，完成绘制
        drawing = False
        end_point = (x, y)
        roi_selected = True

        # 计算框选区域
        x1, y1 = start_point
        x2, y2 = end_point
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        # 确保区域有效
        if x_max > x_min and y_max > y_min:
            # 提取ROI
            roi = image[y_min:y_max, x_min:x_max]

            # 计算中心点坐标
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            # 输出中心点坐标
            print(f"\033[32m框选区域中心坐标: ({center_x}, {center_y})\033[0m")

            # 显示框选的图像
            cv2.imshow('Selected ROI', roi)

            # 保存框选图片
            save_path = os.path.join(script_dir, 'cat_roi.png')
            cv2.imwrite(save_path, roi)
            print(f"\033[32m框选图片已保存: {save_path}\033[0m")

# 创建窗口
cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Select ROI', 800, 600)
cv2.setMouseCallback('Select ROI', mouse_callback)

# 显示原图
cv2.imshow('Select ROI', image)

print("\033[36m操作说明:\033[0m")
print("  鼠标拖动框选区域")
print("  按 \033[33mESC\033[0m 退出程序")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC退出
        break

cv2.destroyAllWindows()
