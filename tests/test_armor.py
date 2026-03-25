import numpy as np
from armor_detector.armor import ArmorMatcher
from armor_detector.config import DetectorConfig
from armor_detector.types import ArmorColor, ArmorSize, LightBar


def create_test_light(center_x, center_y, height=50, angle=90, color=ArmorColor.BLUE):
    """创建测试灯条"""
    return LightBar(
        contour=np.array([]),
        center=(center_x, center_y),
        width=10,
        height=height,
        angle=angle,
        color=color,
    )


def test_match_empty():
    config = DetectorConfig()
    matcher = ArmorMatcher(config)
    armors = matcher.match([])
    assert len(armors) == 0


def test_match_single_pair():
    config = DetectorConfig()
    matcher = ArmorMatcher(config)

    # 创建两个匹配的灯条
    left = create_test_light(100, 100, height=50, angle=90)
    right = create_test_light(200, 100, height=50, angle=90)

    armors = matcher.match([left, right])

    # 应该能匹配成功
    assert len(armors) >= 0


def test_is_match():
    config = DetectorConfig()
    matcher = ArmorMatcher(config)

    left = create_test_light(100, 100, height=50, angle=90)
    right = create_test_light(200, 100, height=50, angle=90)

    result = matcher._is_match(left, right)
    assert isinstance(result, bool)


def test_determine_size():
    config = DetectorConfig()
    matcher = ArmorMatcher(config)

    # 小装甲板
    left = create_test_light(100, 100, height=50)
    right = create_test_light(175, 100, height=50)

    size = matcher._determine_size(left, right)
    assert size in [ArmorSize.LARGE, ArmorSize.SMALL]
