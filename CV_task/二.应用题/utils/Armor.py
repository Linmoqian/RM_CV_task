import cv2
import math
import numpy as np

class Armor:
    def __init__(self,ID,color=None,position=None,width=0,height=0): 
        self.ID = ID
        self.color = color
        self.width = float(width)
        self.height = float(height)
        self.position = position

    def detect(self, image):
        """
        识别装甲板并返回装甲板的坐标和颜色
        """
        result = {
            "id": self.ID,
            "color": self.Armor_Color(),
            "center": self.Central_point(),
            "diagonal": self.Diagonal(),
            "points": self.Armor_Point(),
            "width": self.width,
            "height": self.height,
        }
        return result

    def Central_point(self):
        """
        计算装甲板中心坐标
        """

    def Diagonal(self):
        """
        计算装甲板对角线长度
        """


    def Armor_Point(self):
        """
        输出装甲板4点坐标
        """

        return [(int(round(x)), int(round(y))) for x, y in pts]
    
    def Armor_Color(self):
        """
        获取装甲板颜色
        """
        return self.color

if __name__ == "__main__":
    