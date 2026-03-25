import numpy as np
import cv2
from armor_detector.color_segmenter import ColorSegmenter
from armor_detector.config import DetectorConfig


def test_segment_red():
    # 创建红色测试图像
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 2] = 255  # 红色通道

    config = DetectorConfig()
    segmenter = ColorSegmenter(config)
    mask = segmenter.segment_red(img)

    assert mask is not None
    assert mask.shape == (100, 100)


def test_segment_blue():
    # 创建蓝色测试图像
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # 蓝色通道

    config = DetectorConfig()
    segmenter = ColorSegmenter(config)
    mask = segmenter.segment_blue(img)

    assert mask is not None
    assert mask.shape == (100, 100)


def test_segment_both():
    # 创建双色测试图像
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :50, 2] = 255  # 左半红色
    img[:, 50:, 0] = 255  # 右半蓝色

    config = DetectorConfig()
    segmenter = ColorSegmenter(config)
    red_mask, blue_mask = segmenter.segment_both(img)

    assert red_mask is not None
    assert blue_mask is not None
