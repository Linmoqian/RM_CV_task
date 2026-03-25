"""
OpenCV色彩分割与边缘提取
功能：
1. 红色色彩分割，二值化输出
2. Canny边缘提取，浅蓝色显示
3. 窗口显示原图、二值图、边缘图
"""

import cv2
import numpy as np
import os

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, '..', '..', 'images', 'image.png')

# 读取原图
image = cv2.imread(image_path)
if image is None:
    print("\033[31m错误：无法读取图片\033[0m")
    exit(1)

# ==================== 1. 色彩分割：红色二值化 ====================
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 红色在HSV中有两个范围（色环首尾相接）
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# 创建两个红色掩码并合并
mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)

# 形态学操作去噪（减少毛刺）
kernel = np.ones((3, 3), np.uint8)
mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

# 二值图：红色为白(255)，其余为黑(0)
binary_img = mask_red.copy()

# ==================== 2. 边缘提取 ====================
# Canny边缘检测
edges = cv2.Canny(binary_img, 50, 150)

# 创建黑色背景，用浅蓝色绘制边缘
edge_img = np.zeros_like(image)
edge_img[edges == 255] = [255, 255, 0]  # BGR: 浅蓝色 (255,255,0)

# ==================== 3. 窗口显示 ====================
# 调整图像大小为1280x720
display_size = (1280, 720)
image_resized = cv2.resize(image, display_size)
binary_resized = cv2.resize(binary_img, display_size)
edge_resized = cv2.resize(edge_img, display_size)

# 显示窗口
cv2.namedWindow('原图', cv2.WINDOW_NORMAL)
cv2.namedWindow('二值图', cv2.WINDOW_NORMAL)
cv2.namedWindow('边缘图', cv2.WINDOW_NORMAL)

cv2.resizeWindow('原图', 1280, 720)
cv2.resizeWindow('二值图', 1280, 720)
cv2.resizeWindow('边缘图', 1280, 720)

cv2.imshow('原图', image_resized)
cv2.imshow('二值图', binary_resized)
cv2.imshow('边缘图', edge_resized)

# ==================== 4. 保存边缘图像 ====================
output_path = os.path.join(script_dir, 'edge_output.png')
cv2.imwrite(output_path, edge_img)
print(f"\033[32m边缘图像已保存: {output_path}\033[0m")

print("\033[36m按任意键退出...\033[0m")
cv2.waitKey(0)
cv2.destroyAllWindows()
