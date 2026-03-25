from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class ArmorColor(Enum):
    """装甲板颜色"""
    RED = "red"
    BLUE = "blue"


class ArmorSize(Enum):
    """装甲板尺寸"""
    LARGE = "large"
    SMALL = "small"


@dataclass
class LightBar:
    """灯条"""
    contour: np.ndarray
    center: Tuple[float, float]
    width: float
    height: float
    angle: float
    color: ArmorColor


@dataclass
class Armor:
    """装甲板"""
    left_light: Optional[LightBar]
    right_light: Optional[LightBar]
    corners: List[Tuple[float, float]]
    center: Tuple[float, float]
    color: ArmorColor
    size: ArmorSize
    confidence: float


@dataclass
class DetectResult:
    """检测结果"""
    armors: List[Armor]
    frame_id: int
    timestamp: float
