import cv2

class Armor:
    def __init__(self,ID,color,position,width,height): 
        self.ID = ID
        self.color = color
        self.position = position
        self.width = width
        self.height = height

    def detect(self, image,):
        """
        Detect armor in the given image.
        """
        pass

    def Central_point():
        """
        计算装甲板中心坐标
        """
        pass

    def Diagonal() :
        """
        计算装甲板对角线长度
        """
        pass

    def Armor_Point():
        """
        输出装甲板4点坐标
        """
        pass

    def Armor_Color():
        """
        输出装甲板颜色
        """
        pass

if __name__ == "__main__":
    armor = Armor(ID=1, color='red', position=(100, 200), width=50, height=30)
    image = cv2.imread('images/3个装甲板.png')
    cv2.imshow('Armor Detection', image)
    cv2.waitKey(0)  # 一直显示，直到按任意键
    cv2.destroyAllWindows()
    armor.detect(image)