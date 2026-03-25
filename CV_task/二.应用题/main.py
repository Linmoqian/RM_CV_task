import cv2
import numpy as np
import os

# 获取脚本所在目录，计算图片绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, '..', '..', 'images', '3个装甲板.png')
image = cv2.imread(image_path)

# 转换到HSV颜色空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 蓝色HSV范围（根据图片实际检测）
lower_blue = np.array([90, 50, 75])
upper_blue = np.array([130, 255, 255])

# 创建蓝色掩码
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 形态学操作
kernel_small = np.ones((5, 5), np.uint8)
kernel_large = np.ones((15, 15), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)  # 去噪声
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)  # 连接相邻区域

# 查找轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制边界框
result = image.copy()
for contour in contours:
    if cv2.contourArea(contour) > 500:  # 过滤小面积
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('Blue Detection', result)
cv2.imshow('Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
