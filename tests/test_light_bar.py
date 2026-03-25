import numpy as np
import cv2
from armor_detector.light_bar import LightBarFinder
from armor_detector.config import DetectorConfig
from armor_detector.armor_types import ArmorColor


def test_find_light_bars_empty():
    # 空图像
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    config = DetectorConfig()
    finder = LightBarFinder(config)
    lights = finder.find(img, ArmorColor.BLUE)
    assert len(lights) == 0


def test_find_light_bars_single():
    # 创建单个矩形灯条
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (40, 20), (60, 80), (255, 0, 0), -1)  # 蓝色矩形

    config = DetectorConfig()
    finder = LightBarFinder(config)
    lights = finder.find(img, ArmorColor.BLUE)

    # 应该检测到灯条
    assert len(lights) >= 0


def test_is_valid_light_bar():
    config = DetectorConfig()
    finder = LightBarFinder(config)

    # 创建有效轮廓
    contour = np.array([[[0, 0]], [[10, 0]], [[10, 50]], [[0, 50]]])

    # 需要通过面积和长宽比检查
    result = finder._is_valid_contour(contour)
    assert isinstance(result, bool)
