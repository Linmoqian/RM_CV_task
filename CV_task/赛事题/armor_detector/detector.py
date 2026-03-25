import time
from typing import Optional

import cv2
import numpy as np

from .armor import ArmorMatcher
from .config import DetectorConfig
from .light_bar import LightBarFinder
from .armor_types import ArmorColor, Armor, DetectResult


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
