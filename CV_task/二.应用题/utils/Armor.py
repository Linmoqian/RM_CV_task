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
        if image is None or image.size == 0:
            return None

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 红色掩码（红色在 HSV 两段区间）
        lower_red_1 = np.array([0, 80, 80], dtype=np.uint8)
        upper_red_1 = np.array([10, 255, 255], dtype=np.uint8)
        lower_red_2 = np.array([160, 80, 80], dtype=np.uint8)
        upper_red_2 = np.array([179, 255, 255], dtype=np.uint8)

        mask_red = cv2.inRange(hsv, lower_red_1, upper_red_1) | cv2.inRange(hsv, lower_red_2, upper_red_2)

        # 蓝色掩码
        lower_blue = np.array([90, 80, 80], dtype=np.uint8)
        upper_blue = np.array([130, 255, 255], dtype=np.uint8)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        kernel = np.ones((3, 3), np.uint8)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel, iterations=1)

        best = self._find_best_rect(mask_red, "red")
        best_blue = self._find_best_rect(mask_blue, "blue")

        if best is None or (best_blue is not None and best_blue["score"] > best["score"]):
            best = best_blue

        if best is None:
            return None

        self.color = best["color"]
        self.position = best["center"]
        self.width = best["width"]
        self.height = best["height"]
        self.points = best["points"]

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
        if self.position is None:
            return None
        return int(round(self.position[0])), int(round(self.position[1]))

    def Diagonal(self):
        """
        计算装甲板对角线长度
        """
        if self.width <= 0 or self.height <= 0:
            return 0.0
        return math.hypot(self.width, self.height)

    def Armor_Point(self):
        """
        输出装甲板4点坐标
        """
        if self.points is not None:
            return [(int(round(x)), int(round(y))) for x, y in self.points]

        if self.position is None or self.width <= 0 or self.height <= 0:
            return []

        cx, cy = self.position
        hw = self.width / 2.0
        hh = self.height / 2.0

        pts = [
            (cx - hw, cy - hh),  # 左上
            (cx + hw, cy - hh),  # 右上
            (cx + hw, cy + hh),  # 右下
            (cx - hw, cy + hh),  # 左下
        ]
        return [(int(round(x)), int(round(y))) for x, y in pts]
    
    def Armor_Color(self):
        """
        获取装甲板颜色
        """
        return self.color

if __name__ == "__main__":
    armor = Armor(ID=1, color='red', position=(100, 200), width=50, height=30)
    # image = cv2.imread('images/3个装甲板.png')
    # cv2.imshow('Armor Detection', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print("Armor ID:", armor.ID, "Color:", armor.color, "Position:", armor.position, "Width:", armor.width, "Height:", armor.height)