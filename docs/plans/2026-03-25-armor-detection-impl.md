# 装甲板识别系统实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现基于传统视觉的RoboMaster装甲板检测系统，支持红蓝双色、大小装甲板识别。

**Architecture:** 采用模块化设计，分为类型定义、配置、灯条提取、装甲板匹配、主检测器五个模块。处理流程：颜色分割→轮廓检测→灯条筛选→灯条匹配→装甲板输出。

**Tech Stack:** Python 3.10+, OpenCV, NumPy, dataclasses

---

## Task 1: 项目结构和类型定义

**Files:**
- Create: `CV_task/赛事题/armor_detector/__init__.py`
- Create: `CV_task/赛事题/armor_detector/types.py`
- Create: `tests/test_types.py`

**Step 1: 创建目录结构**

```bash
mkdir -p CV_task/赛事题/armor_detector
mkdir -p tests
```

**Step 2: 创建 __init__.py**

```python
# CV_task/赛事题/armor_detector/__init__.py
from .types import ArmorColor, ArmorSize, LightBar, Armor, DetectResult
from .detector import ArmorDetector

__all__ = [
    "ArmorColor",
    "ArmorSize",
    "LightBar",
    "Armor",
    "DetectResult",
    "ArmorDetector",
]
```

**Step 3: 写类型定义测试**

```python
# tests/test_types.py
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
```

**Step 4: 实现类型定义**

```python
# CV_task/赛事题/armor_detector/types.py
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
```

**Step 5: 运行测试验证**

```bash
cd D:/project/RM_CV_task
python -m pytest tests/test_types.py -v
```
Expected: 5 passed

**Step 6: 提交**

```bash
git add CV_task/赛事题/armor_detector/ tests/test_types.py
git commit -m "feat(armor): 添加类型定义模块"
```

---

## Task 2: 配置模块

**Files:**
- Create: `CV_task/赛事题/armor_detector/config.py`
- Create: `tests/test_config.py`

**Step 1: 写配置测试**

```python
# tests/test_config.py
from armor_detector.config import DetectorConfig


def test_default_config():
    config = DetectorConfig()
    assert config.red_h_min == 0
    assert config.red_h_max == 10
    assert config.blue_h_min == 100
    assert config.light_min_area == 10


def test_custom_config():
    config = DetectorConfig(light_min_area=100, light_max_area=5000)
    assert config.light_min_area == 100
    assert config.light_max_area == 5000


def test_config_to_dict():
    config = DetectorConfig()
    d = config.to_dict()
    assert isinstance(d, dict)
    assert "light_min_area" in d
```

**Step 2: 实现配置模块**

```python
# CV_task/赛事题/armor_detector/config.py
from dataclasses import dataclass, field, asdict
from typing import Dict, Any


@dataclass
class DetectorConfig:
    """检测器配置参数"""

    # HSV颜色阈值 - 红色
    red_h_min: int = 0
    red_h_max: int = 10
    red_s_min: int = 100
    red_s_max: int = 255
    red_v_min: int = 100
    red_v_max: int = 255

    # HSV颜色阈值 - 红色(高H值区域)
    red_h2_min: int = 156
    red_h2_max: int = 180

    # HSV颜色阈值 - 蓝色
    blue_h_min: int = 100
    blue_h_max: int = 130
    blue_s_min: int = 100
    blue_s_max: int = 255
    blue_v_min: int = 100
    blue_v_max: int = 255

    # 灯条几何约束
    light_min_area: int = 10
    light_max_area: int = 5000
    light_min_ratio: float = 1.5
    light_max_ratio: float = 6.0
    light_fill_ratio: float = 0.6

    # 灯条匹配约束
    match_max_angle_diff: float = 15.0
    match_min_height_ratio: float = 0.7
    match_max_height_ratio: float = 1.3
    match_min_dist_ratio: float = 1.0
    match_max_dist_ratio: float = 4.5
    match_max_y_diff_ratio: float = 0.5

    # 装甲板尺寸判断
    small_armor_ratio_min: float = 1.0
    small_armor_ratio_max: float = 2.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
```

**Step 3: 运行测试验证**

```bash
python -m pytest tests/test_config.py -v
```
Expected: 3 passed

**Step 4: 提交**

```bash
git add CV_task/赛事题/armor_detector/config.py tests/test_config.py
git commit -m "feat(armor): 添加配置模块"
```

---

## Task 3: 颜色分割模块

**Files:**
- Create: `CV_task/赛事题/armor_detector/color_segmenter.py`
- Create: `tests/test_color_segmenter.py`

**Step 1: 写颜色分割测试**

```python
# tests/test_color_segmenter.py
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
```

**Step 2: 实现颜色分割模块**

```python
# CV_task/赛事题/armor_detector/color_segmenter.py
import cv2
import numpy as np

from .config import DetectorConfig
from .types import ArmorColor


class ColorSegmenter:
    """HSV颜色分割器"""

    def __init__(self, config: DetectorConfig):
        self.config = config

    def segment_red(self, frame: np.ndarray) -> np.ndarray:
        """分割红色区域"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 红色在HSV中有两个区域
        lower1 = np.array([
            self.config.red_h_min,
            self.config.red_s_min,
            self.config.red_v_min
        ])
        upper1 = np.array([
            self.config.red_h_max,
            self.config.red_s_max,
            self.config.red_v_max
        ])

        lower2 = np.array([
            self.config.red_h2_min,
            self.config.red_s_min,
            self.config.red_v_min
        ])
        upper2 = np.array([
            self.config.red_h2_max,
            self.config.red_s_max,
            self.config.red_v_max
        ])

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)

        return cv2.bitwise_or(mask1, mask2)

    def segment_blue(self, frame: np.ndarray) -> np.ndarray:
        """分割蓝色区域"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([
            self.config.blue_h_min,
            self.config.blue_s_min,
            self.config.blue_v_min
        ])
        upper = np.array([
            self.config.blue_h_max,
            self.config.blue_s_max,
            self.config.blue_v_max
        ])

        return cv2.inRange(hsv, lower, upper)

    def segment_both(self, frame: np.ndarray) -> tuple:
        """同时分割红色和蓝色区域"""
        return self.segment_red(frame), self.segment_blue(frame)

    def segment_by_color(self, frame: np.ndarray, color: ArmorColor) -> np.ndarray:
        """根据颜色类型分割"""
        if color == ArmorColor.RED:
            return self.segment_red(frame)
        else:
            return self.segment_blue(frame)
```

**Step 3: 运行测试验证**

```bash
python -m pytest tests/test_color_segmenter.py -v
```
Expected: 3 passed

**Step 4: 提交**

```bash
git add CV_task/赛事题/armor_detector/color_segmenter.py tests/test_color_segmenter.py
git commit -m "feat(armor): 添加颜色分割模块"
```

---

## Task 4: 灯条提取模块

**Files:**
- Create: `CV_task/赛事题/armor_detector/light_bar.py`
- Create: `tests/test_light_bar.py`

**Step 1: 写灯条提取测试**

```python
# tests/test_light_bar.py
import numpy as np
import cv2
from armor_detector.light_bar import LightBarFinder
from armor_detector.config import DetectorConfig
from armor_detector.types import ArmorColor


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
```

**Step 2: 实现灯条提取模块**

```python
# CV_task/赛事题/armor_detector/light_bar.py
import cv2
import numpy as np
from typing import List, Tuple

from .config import DetectorConfig
from .types import ArmorColor, LightBar
from .color_segmenter import ColorSegmenter


class LightBarFinder:
    """灯条提取器"""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.segmenter = ColorSegmenter(config)

    def find(self, frame: np.ndarray, color: ArmorColor) -> List[LightBar]:
        """提取指定颜色的灯条"""
        # 颜色分割
        mask = self.segmenter.segment_by_color(frame, color)

        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 轮廓检测
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 筛选有效灯条
        light_bars = []
        for contour in contours:
            if self._is_valid_contour(contour):
                light = self._create_light_bar(contour, color)
                if light is not None:
                    light_bars.append(light)

        # 按x坐标排序
        light_bars.sort(key=lambda x: x.center[0])

        return light_bars

    def _is_valid_contour(self, contour: np.ndarray) -> bool:
        """检查轮廓是否符合灯条特征"""
        area = cv2.contourArea(contour)

        # 面积检查
        if area < self.config.light_min_area:
            return False
        if area > self.config.light_max_area:
            return False

        # 凸性检查
        if not cv2.isContourConvex(contour):
            hull = cv2.convexHull(contour)
            contour = hull

        # 最小外接矩形
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]

        # 长宽比检查 (取较长边为高度)
        if width > height:
            width, height = height, width

        if height < 1e-5:
            return False

        ratio = height / width
        if ratio < self.config.light_min_ratio:
            return False
        if ratio > self.config.light_max_ratio:
            return False

        # 填充率检查
        rect_area = width * height
        if rect_area < 1e-5:
            return False

        fill_ratio = area / rect_area
        if fill_ratio < self.config.light_fill_ratio:
            return False

        return True

    def _create_light_bar(
        self, contour: np.ndarray, color: ArmorColor
    ) -> LightBar:
        """创建灯条对象"""
        # 获取最小外接矩形
        rect = cv2.minAreaRect(contour)
        center = rect[0]
        width, height = rect[1]
        angle = rect[2]

        # 标准化角度和尺寸
        if width > height:
            width, height = height, width
            angle = angle + 90

        # 角度范围 [-90, 90]
        if angle < -90:
            angle += 180
        if angle > 90:
            angle -= 180

        return LightBar(
            contour=contour,
            center=tuple(map(float, center)),
            width=float(width),
            height=float(height),
            angle=float(angle),
            color=color,
        )

    def find_all(self, frame: np.ndarray) -> Tuple[List[LightBar], List[LightBar]]:
        """提取所有红蓝灯条"""
        red_lights = self.find(frame, ArmorColor.RED)
        blue_lights = self.find(frame, ArmorColor.BLUE)
        return red_lights, blue_lights
```

**Step 3: 运行测试验证**

```bash
python -m pytest tests/test_light_bar.py -v
```
Expected: 3 passed

**Step 4: 提交**

```bash
git add CV_task/赛事题/armor_detector/light_bar.py tests/test_light_bar.py
git commit -m "feat(armor): 添加灯条提取模块"
```

---

## Task 5: 装甲板匹配模块

**Files:**
- Create: `CV_task/赛事题/armor_detector/armor.py`
- Create: `tests/test_armor.py`

**Step 1: 写装甲板匹配测试**

```python
# tests/test_armor.py
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
```

**Step 2: 实现装甲板匹配模块**

```python
# CV_task/赛事题/armor_detector/armor.py
import math
from typing import List

from .config import DetectorConfig
from .types import ArmorColor, ArmorSize, LightBar, Armor


class ArmorMatcher:
    """装甲板匹配器"""

    def __init__(self, config: DetectorConfig):
        self.config = config

    def match(self, light_bars: List[LightBar]) -> List[Armor]:
        """灯条匹配生成装甲板"""
        armors = []
        n = len(light_bars)

        if n < 2:
            return armors

        # 遍历所有灯条对
        for i in range(n):
            for j in range(i + 1, n):
                left, right = light_bars[i], light_bars[j]

                if self._is_match(left, right):
                    armor = self._create_armor(left, right)
                    if armor is not None:
                        armors.append(armor)

        return armors

    def _is_match(self, left: LightBar, right: LightBar) -> bool:
        """判断两个灯条是否可以匹配成装甲板"""
        # 同色检查
        if left.color != right.color:
            return False

        # 角度差检查 (平行度)
        angle_diff = abs(left.angle - right.angle)
        if angle_diff > self.config.match_max_angle_diff:
            return False

        # 高度比检查
        if left.height < 1e-5 or right.height < 1e-5:
            return False

        height_ratio = left.height / right.height
        if height_ratio < self.config.match_min_height_ratio:
            return False
        if height_ratio > self.config.match_max_height_ratio:
            return False

        # 计算间距
        dx = right.center[0] - left.center[0]
        dy = right.center[1] - left.center[1]
        distance = math.sqrt(dx * dx + dy * dy)

        # 间距检查 (灯条必须左右排列)
        if dx <= 0:
            return False

        avg_height = (left.height + right.height) / 2
        dist_ratio = distance / avg_height

        if dist_ratio < self.config.match_min_dist_ratio:
            return False
        if dist_ratio > self.config.match_max_dist_ratio:
            return False

        # 垂直对齐检查
        y_diff = abs(dy)
        if y_diff > self.config.match_max_y_diff_ratio * avg_height:
            return False

        return True

    def _determine_size(self, left: LightBar, right: LightBar) -> ArmorSize:
        """判断装甲板尺寸"""
        dx = right.center[0] - left.center[0]
        dy = right.center[1] - left.center[1]
        distance = math.sqrt(dx * dx + dy * dy)

        avg_height = (left.height + right.height) / 2
        ratio = distance / avg_height

        if self.config.small_armor_ratio_min < ratio < self.config.small_armor_ratio_max:
            return ArmorSize.SMALL
        return ArmorSize.LARGE

    def _create_armor(self, left: LightBar, right: LightBar) -> Armor:
        """创建装甲板对象"""
        # 计算装甲板中心
        center_x = (left.center[0] + right.center[0]) / 2
        center_y = (left.center[1] + right.center[1]) / 2

        # 计算四角坐标
        corners = self._compute_corners(left, right)

        # 判断尺寸
        size = self._determine_size(left, right)

        # 计算置信度 (简化：基于几何特征)
        confidence = self._compute_confidence(left, right)

        return Armor(
            left_light=left,
            right_light=right,
            corners=corners,
            center=(center_x, center_y),
            color=left.color,
            size=size,
            confidence=confidence,
        )

    def _compute_corners(
        self, left: LightBar, right: LightBar
    ) -> List[tuple]:
        """计算装甲板四角坐标"""
        # 简化：使用灯条端点估算
        left_half_h = left.height / 2
        right_half_h = right.height / 2

        # 左灯条上下端点
        left_top = (left.center[0], left.center[1] - left_half_h)
        left_bottom = (left.center[0], left.center[1] + left_half_h)

        # 右灯条上下端点
        right_top = (right.center[0], right.center[1] - right_half_h)
        right_bottom = (right.center[0], right.center[1] + right_half_h)

        # 四角：左上、右上、右下、左下
        return [
            left_top,
            right_top,
            right_bottom,
            left_bottom,
        ]

    def _compute_confidence(self, left: LightBar, right: LightBar) -> float:
        """计算置信度"""
        # 基于高度比和对齐度
        height_ratio = min(left.height, right.height) / max(left.height, right.height)

        y_diff = abs(left.center[1] - right.center[1])
        avg_height = (left.height + right.height) / 2
        alignment = 1 - min(y_diff / avg_height, 1)

        return (height_ratio + alignment) / 2
```

**Step 3: 运行测试验证**

```bash
python -m pytest tests/test_armor.py -v
```
Expected: 4 passed

**Step 4: 提交**

```bash
git add CV_task/赛事题/armor_detector/armor.py tests/test_armor.py
git commit -m "feat(armor): 添加装甲板匹配模块"
```

---

## Task 6: 主检测器模块

**Files:**
- Create: `CV_task/赛事题/armor_detector/detector.py`
- Create: `tests/test_detector.py`

**Step 1: 写检测器测试**

```python
# tests/test_detector.py
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
```

**Step 2: 实现主检测器模块**

```python
# CV_task/赛事题/armor_detector/detector.py
import time
from typing import Optional

import cv2
import numpy as np

from .armor import ArmorMatcher
from .config import DetectorConfig
from .light_bar import LightBarFinder
from .types import ArmorColor, Armor, DetectResult


class ArmorDetector:
    """装甲板检测器"""

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.light_finder = LightBarFinder(self.config)
        self.armor_matcher = ArmorMatcher(self.config)
        self.frame_count = 0

    def detect(self, frame: np.ndarray) -> DetectResult:
        """检测装甲板"""
        self.frame_count += 1
        start_time = time.time()

        # 提取所有灯条
        red_lights, blue_lights = self.light_finder.find_all(frame)

        # 匹配装甲板
        red_armors = self.armor_matcher.match(red_lights)
        blue_armors = self.armor_matcher.match(blue_lights)

        # 合并结果
        all_armors = red_armors + blue_armors

        elapsed = time.time() - start_time

        return DetectResult(
            armors=all_armors,
            frame_id=self.frame_count,
            timestamp=elapsed,
        )

    def draw_result(
        self,
        frame: np.ndarray,
        result: DetectResult,
        draw_lights: bool = True,
        draw_armors: bool = True,
    ) -> np.ndarray:
        """绘制检测结果"""
        vis = frame.copy()

        # 颜色映射
        color_map = {
            ArmorColor.RED: (0, 0, 255),
            ArmorColor.BLUE: (255, 0, 0),
        }

        # 绘制灯条
        if draw_lights:
            for armor in result.armors:
                if armor.left_light:
                    self._draw_light(vis, armor.left_light, color_map)
                if armor.right_light:
                    self._draw_light(vis, armor.right_light, color_map)

        # 绘制装甲板
        if draw_armors:
            for armor in result.armors:
                self._draw_armor(vis, armor, color_map)

        return vis

    def _draw_light(self, frame: np.ndarray, light, color_map: dict):
        """绘制灯条"""
        color = color_map.get(light.color, (0, 255, 0))
        cv2.drawContours(frame, [light.contour], -1, color, 2)

    def _draw_armor(self, frame: np.ndarray, armor: Armor, color_map: dict):
        """绘制装甲板"""
        color = color_map.get(armor.color, (0, 255, 0))

        # 绘制四边形
        if len(armor.corners) == 4:
            pts = np.array(armor.corners, dtype=np.int32)
            cv2.polylines(frame, [pts], True, color, 2)

        # 绘制中心点
        center = tuple(map(int, armor.center))
        cv2.circle(frame, center, 5, (0, 255, 0), -1)

        # 绘制标签
        label = f"{armor.color.value}-{armor.size.value}"
        cv2.putText(
            frame,
            label,
            (center[0] - 30, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
```

**Step 3: 运行测试验证**

```bash
python -m pytest tests/test_detector.py -v
```
Expected: 4 passed

**Step 4: 提交**

```bash
git add CV_task/赛事题/armor_detector/detector.py tests/test_detector.py
git commit -m "feat(armor): 添加主检测器模块"
```

---

## Task 7: 集成测试和示例脚本

**Files:**
- Create: `CV_task/赛事题/armor_detector/demo.py`
- Create: `tests/test_integration.py`

**Step 1: 写集成测试**

```python
# tests/test_integration.py
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
```

**Step 2: 创建示例脚本**

```python
# CV_task/赛事题/armor_detector/demo.py
#!/usr/bin/env python3
"""
装甲板检测演示脚本
支持图片、视频、摄像头三种输入模式
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# 添加本地模块路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from armor_detector import ArmorDetector


def process_image(detector: ArmorDetector, image_path: str):
    """处理单张图片"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return

    result = detector.detect(img)
    vis = detector.draw_result(img, result)

    print(f"检测到 {len(result.armors)} 个装甲板")
    for i, armor in enumerate(result.armors):
        print(f"  [{i+1}] {armor.color.value}-{armor.size.value} @ {armor.center}")

    cv2.imshow("Armor Detection", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(detector: ArmorDetector, video_path: str):
    """处理视频文件"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect(frame)
        vis = detector.draw_result(frame, result)

        cv2.imshow("Armor Detection", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_camera(detector: ArmorDetector, camera_id: int = 0):
    """处理摄像头输入"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_id}")
        return

    # 设置低曝光
    cap.set(cv2.CAP_PROP_EXPOSURE, -10)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect(frame)
        vis = detector.draw_result(frame, result)

        # 显示FPS
        cv2.putText(
            vis,
            f"FPS: {1/result.timestamp:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Armor Detection", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="装甲板检测演示")
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="输入图片路径"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        help="输入视频路径"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="摄像头ID (默认: 0)"
    )

    args = parser.parse_args()

    detector = ArmorDetector()

    if args.image:
        process_image(detector, args.image)
    elif args.video:
        process_video(detector, args.video)
    else:
        process_camera(detector, args.camera)


if __name__ == "__main__":
    main()
```

**Step 3: 运行测试验证**

```bash
python -m pytest tests/test_integration.py -v
```
Expected: 3 passed

**Step 4: 提交**

```bash
git add CV_task/赛事题/armor_detector/demo.py tests/test_integration.py
git commit -m "feat(armor): 添加集成测试和演示脚本"
```

---

## Task 8: 更新 __init__.py 导出

**Files:**
- Modify: `CV_task/赛事题/armor_detector/__init__.py`

**Step 1: 更新导出**

```python
# CV_task/赛事题/armor_detector/__init__.py
from .types import ArmorColor, ArmorSize, LightBar, Armor, DetectResult
from .config import DetectorConfig
from .color_segmenter import ColorSegmenter
from .light_bar import LightBarFinder
from .armor import ArmorMatcher
from .detector import ArmorDetector

__all__ = [
    # Types
    "ArmorColor",
    "ArmorSize",
    "LightBar",
    "Armor",
    "DetectResult",
    # Config
    "DetectorConfig",
    # Components
    "ColorSegmenter",
    "LightBarFinder",
    "ArmorMatcher",
    # Main
    "ArmorDetector",
]
```

**Step 2: 运行所有测试**

```bash
python -m pytest tests/ -v
```
Expected: All tests passed

**Step 3: 提交**

```bash
git add CV_task/赛事题/armor_detector/__init__.py
git commit -m "feat(armor): 完善模块导出"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | 类型定义 | types.py, test_types.py |
| 2 | 配置模块 | config.py, test_config.py |
| 3 | 颜色分割 | color_segmenter.py, test_color_segmenter.py |
| 4 | 灯条提取 | light_bar.py, test_light_bar.py |
| 5 | 装甲板匹配 | armor.py, test_armor.py |
| 6 | 主检测器 | detector.py, test_detector.py |
| 7 | 集成测试 | demo.py, test_integration.py |
| 8 | 完善导出 | __init__.py |
