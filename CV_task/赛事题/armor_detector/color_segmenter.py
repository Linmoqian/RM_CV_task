import cv2
import numpy as np

from .config import DetectorConfig
from .armor_types import ArmorColor


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
