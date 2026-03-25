import numpy as np
from armor_detector.types import ArmorColor, ArmorSize, LightBar, Armor, DetectResult


def test_armor_color_enum():
    assert ArmorColor.RED.value == "red"
    assert ArmorColor.BLUE.value == "blue"


def test_armor_size_enum():
    assert ArmorSize.LARGE.value == "large"
    assert ArmorSize.SMALL.value == "small"


def test_light_bar_dataclass():
    contour = np.array([[[0, 0]], [[10, 0]], [[10, 30]], [[0, 30]]])
    light = LightBar(
        contour=contour,
        center=(5.0, 15.0),
        width=10.0,
        height=30.0,
        angle=90.0,
        color=ArmorColor.BLUE,
    )
    assert light.center == (5.0, 15.0)
    assert light.color == ArmorColor.BLUE


def test_armor_dataclass():
    light = LightBar(
        contour=np.array([]),
        center=(0, 0),
        width=10,
        height=30,
        angle=90,
        color=ArmorColor.RED,
    )
    armor = Armor(
        left_light=light,
        right_light=light,
        corners=[(0, 0), (50, 0), (50, 40), (0, 40)],
        center=(25, 20),
        color=ArmorColor.RED,
        size=ArmorSize.SMALL,
        confidence=0.9,
    )
    assert armor.color == ArmorColor.RED
    assert armor.size == ArmorSize.SMALL


def test_detect_result_dataclass():
    armor = Armor(
        left_light=None,
        right_light=None,
        corners=[],
        center=(0, 0),
        color=ArmorColor.BLUE,
        size=ArmorSize.LARGE,
        confidence=0.8,
    )
    result = DetectResult(armors=[armor], frame_id=1, timestamp=0.5)
    assert len(result.armors) == 1
    assert result.frame_id == 1
