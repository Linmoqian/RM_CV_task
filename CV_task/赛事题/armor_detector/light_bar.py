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
