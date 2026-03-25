import numpy as np
import cv2
from armor_detector import ArmorDetector
from armor_detector.types import ArmorColor, ArmorSize


def test_full_pipeline_empty():
    """完整流水线测试 - 空图像"""
    detector = ArmorDetector()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(img)

    assert result is not None
    assert len(result.armors) == 0


def test_full_pipeline_with_blue_armor():
    """完整流水线测试 - 蓝色装甲板"""
    detector = ArmorDetector()

    # 创建模拟蓝色装甲板图像
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # 左灯条
    cv2.rectangle(img, (100, 150), (120, 300), (255, 100, 100), -1)
    # 右灯条
    cv2.rectangle(img, (200, 150), (220, 300), (255, 100, 100), -1)

    result = detector.detect(img)

    # 检查结果格式正确
    assert result is not None
    assert isinstance(result.armors, list)


def test_draw_visualization():
    """可视化绘制测试"""
    detector = ArmorDetector()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(img)

    vis = detector.draw_result(img, result)

    assert vis.shape == img.shape
    assert vis.dtype == np.uint8
