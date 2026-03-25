import numpy as np
from armor_detector.detector import ArmorDetector
from armor_detector.config import DetectorConfig


def test_detector_init():
    detector = ArmorDetector()
    assert detector is not None


def test_detector_with_config():
    config = DetectorConfig(light_min_area=100)
    detector = ArmorDetector(config)
    assert detector.config.light_min_area == 100


def test_detect_empty():
    detector = ArmorDetector()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(img)
    assert result is not None
    assert len(result.armors) == 0


def test_draw_result():
    detector = ArmorDetector()
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(img)
    vis = detector.draw_result(img, result)
    assert vis is not None
    assert vis.shape == img.shape
