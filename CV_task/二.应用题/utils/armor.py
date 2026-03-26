"""
装甲板类模块
用于表示和操作RoboMaster比赛中的装甲板属性
"""

from dataclasses import dataclass
from typing import Tuple
import math


@dataclass
class Rect:
    """
    矩形结构体，表示装甲板的基本属性

    Attributes:
        id: 装甲板数字ID (1-6)
        color: 装甲板颜色 (0=蓝色, 1=红色)
        point: 左上角坐标 (x, y)
        width: 宽度
        height: 高度
    """
    id: int
    color: int
    point: Tuple[int, int]
    width: int
    height: int


class Armor:
    """
    装甲板类，提供装甲板的各种计算和输出功能

    Attributes:
        rect: Rect结构体，包含装甲板的基本属性
    """

    # 颜色映射字典
    COLOR_MAP = {0: "蓝", 1: "红"}

    def __init__(self, rect: Rect):
        """
        初始化Armor对象

        Args:
            rect: Rect结构体，包含装甲板属性
        """
        self.rect = rect

    def Central_Point(self) -> Tuple[int, int]:
        """
        计算装甲板中心坐标

        Returns:
            中心点坐标元组 (cx, cy)
        """
        x, y = self.rect.point
        cx = x + self.rect.width // 2
        cy = y + self.rect.height // 2
        return (cx, cy)

    def Diagonal(self) -> float:
        """
        计算装甲板对角线长度

        Returns:
            对角线长度，保留两位小数
        """
        diagonal = math.sqrt(self.rect.width ** 2 + self.rect.height ** 2)
        return round(diagonal, 2)

    def Armor_Point(self) -> Tuple[Tuple[int, int], ...]:
        """
        输出装甲板4点坐标，从左上角开始顺时针排列

        Returns:
            四个角点坐标的元组: (左上, 右上, 右下, 左下)
        """
        x, y = self.rect.point
        w, h = self.rect.width, self.rect.height

        top_left = (x, y)           # 左上角
        top_right = (x + w, y)      # 右上角
        bottom_right = (x + w, y + h)  # 右下角
        bottom_left = (x, y + h)    # 左下角

        return (top_left, top_right, bottom_right, bottom_left)

    def Armor_Color(self) -> str:
        """
        输出装甲板颜色

        Returns:
            颜色名称字符串 ("蓝" 或 "红")
        """
        return self.COLOR_MAP.get(self.rect.color, "未知")


def main():
    """
    主函数：处理输入并输出装甲板信息
    """
    # 读取输入
    print("\033[36m请输入装甲板ID和颜色 (ID:1-6, 颜色:0=蓝/1=红):\033[0m", end=" ")
    id_color = input().split()
    armor_id = int(id_color[0])
    armor_color = int(id_color[1])

    print("\033[36m请输入左上角坐标和宽高 (x y width height):\033[0m", end=" ")
    coords = input().split()
    x, y = int(coords[0]), int(coords[1])
    width, height = int(coords[2]), int(coords[3])

    # 创建Rect和Armor对象
    rect = Rect(id=armor_id, color=armor_color, point=(x, y), width=width, height=height)
    armor = Armor(rect)

    # 输出第一行：ID和颜色
    print(f"ID：{armor.rect.id} 颜色：{armor.Armor_Color()}")

    # 输出第二行：中心坐标和对角线长度
    cx, cy = armor.Central_Point()
    print(f"({cx},{cy}) 长度：{armor.Diagonal():.2f}")

    # 输出第三行：4点坐标
    points = armor.Armor_Point()
    points_str = " ".join(f"({p[0]},{p[1]})" for p in points)
    print(points_str)


if __name__ == "__main__":
    main()
