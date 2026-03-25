#!/usr/bin/env python3
"""
图像处理模块
提供模块化的图像处理框架，支持自定义处理器链
"""

import abc
from typing import Any, Callable
import cv2
import numpy as np


class BaseProcessor(abc.ABC):
    """图像处理器基类"""

    @abc.abstractmethod
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        处理图像帧

        Args:
            frame: 输入图像 (BGR格式)

        Returns:
            处理后的图像
        """
        pass

    @property
    def name(self) -> str:
        """处理器名称"""
        return self.__class__.__name__


class ResizeProcessor(BaseProcessor):
    """图像缩放处理器"""

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height

    def process(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, (self.width, self.height))


class GrayProcessor(BaseProcessor):
    """灰度转换处理器"""

    def process(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class BlurProcessor(BaseProcessor):
    """高斯模糊处理器"""

    def __init__(self, kernel_size: int = 5):
        self.kernel_size = kernel_size

    def process(self, frame: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(frame, (self.kernel_size, self.kernel_size), 0)


class CannyProcessor(BaseProcessor):
    """Canny边缘检测处理器"""

    def __init__(self, threshold1: int = 50, threshold2: int = 150):
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def process(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.threshold1, self.threshold2)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


class ThresholdProcessor(BaseProcessor):
    """二值化处理器"""

    def __init__(self, threshold: int = 127, mode: int = cv2.THRESH_BINARY):
        self.threshold = threshold
        self.mode = mode

    def process(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.threshold, 255, self.mode)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


class ColorFilterProcessor(BaseProcessor):
    """颜色过滤处理器"""

    def __init__(self, lower: tuple = (0, 0, 0), upper: tuple = (180, 255, 255)):
        self.lower = np.array(lower)
        self.upper = np.array(upper)

    def process(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        return cv2.bitwise_and(frame, frame, mask=mask)


class DrawInfoProcessor(BaseProcessor):
    """绘制信息处理器"""

    def __init__(self, text: str = "", position: tuple = (10, 30)):
        self.text = text
        self.position = position

    def process(self, frame: np.ndarray) -> np.ndarray:
        result = frame.copy()
        cv2.putText(result, self.text, self.position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return result


class FunctionProcessor(BaseProcessor):
    """函数式处理器 - 用函数快速创建处理器"""

    def __init__(self, func: Callable[[np.ndarray], np.ndarray], name: str = None):
        self._func = func
        self._name = name or func.__name__

    def process(self, frame: np.ndarray) -> np.ndarray:
        return self._func(frame)

    @property
    def name(self) -> str:
        return self._name


class ProcessorChain:
    """处理器链 - 串联多个处理器"""

    def __init__(self):
        self.processors: list[BaseProcessor] = []

    def add(self, processor: BaseProcessor) -> 'ProcessorChain':
        """添加处理器"""
        self.processors.append(processor)
        return self

    def add_func(self, func: Callable[[np.ndarray], np.ndarray],
                 name: str = None) -> 'ProcessorChain':
        """添加函数作为处理器"""
        self.processors.append(FunctionProcessor(func, name))
        return self

    def process(self, frame: np.ndarray) -> np.ndarray:
        """依次执行所有处理器"""
        result = frame
        for processor in self.processors:
            result = processor.process(result)
        return result

    def clear(self):
        """清空处理器链"""
        self.processors.clear()

    def __len__(self):
        return len(self.processors)

    def __iter__(self):
        return iter(self.processors)


# ============ 预定义的常用处理器 ============

def create_red_filter() -> ColorFilterProcessor:
    """创建红色过滤器"""
    return ColorFilterProcessor(lower=(0, 100, 100), upper=(10, 255, 255))


def create_blue_filter() -> ColorFilterProcessor:
    """创建蓝色过滤器"""
    return ColorFilterProcessor(lower=(100, 100, 100), upper=(130, 255, 255))


def create_green_filter() -> ColorFilterProcessor:
    """创建绿色过滤器"""
    return ColorFilterProcessor(lower=(35, 100, 100), upper=(85, 255, 255))


# ============ 示例用法 ============

if __name__ == '__main__':
    # 示例：创建处理链
    chain = ProcessorChain()
    chain.add(ResizeProcessor(640, 480))
    chain.add(GrayProcessor())
    chain.add_func(lambda f: cv2.flip(f, 1), "水平翻转")

    # 使用摄像头测试
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = chain.process(frame)
        cv2.imshow('Processed', processed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
