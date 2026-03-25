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
