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
