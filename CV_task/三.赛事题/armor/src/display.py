import cv2
import numpy as np
# ============ 图像合并函数 ============
def merge_display_images(gaussian, result, crop):
    """
    将三个图像合并在一个窗口中显示
    布局：
    ┌─────────────┬─────────────┐
    │ Blue Channel│ Parallelogram│
    ├─────────────┴─────────────┤
    │     Armor Crop (居中)      │
    └───────────────────────────┘
    """
    h, w = result.shape[:2]

    # 确保 Gaussian 是彩色图像（3通道）
    if len(gaussian.shape) == 2:
        gaussian = cv2.cvtColor(gaussian, cv2.COLOR_GRAY2BGR)

    # 处理 crop 图像
    if crop is None:
        # 如果没有检测到装甲板，创建一个占位图像
        crop = np.zeros((h // 2, w // 2, 3), dtype=np.uint8)
        cv2.putText(crop, "No Armor Detected", (20, h // 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        # 调整 crop 大小以匹配布局
        crop = cv2.resize(crop, (w // 2, h // 2))

    # 调整 Gaussian 和 result 大小为相同尺寸
    gaussian = cv2.resize(gaussian, (w // 2, h // 2))
    result_small = cv2.resize(result, (w // 2, h // 2))

    # 上半部分：Blue Channel (左) + Parallelogram (右)
    top_row = np.hstack([gaussian, result_small])

    # 下半部分：Armor Crop (居中，两侧填充黑色)
    crop_with_padding = cv2.copyMakeBorder(crop, 0, 0, w // 4, w // 4,
                                           cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # 垂直拼接上下两部分
    merged = np.vstack([top_row, crop_with_padding])

    # 添加标签
    cv2.putText(merged, "Blue Channel", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(merged, "Parallelogram", (w // 2 + 10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(merged, "Armor Crop", (10, h // 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return merged