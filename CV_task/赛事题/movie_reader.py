#!/usr/bin/env python3
"""
ROS2摄像头帧订阅节点
订阅 /camera/image_raw 话题，支持自定义图像处理
"""

import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# 添加本地模块路径
sys.path.append(str(Path(__file__).parent))
from image_processor import ProcessorChain, BaseProcessor


class CameraSubscriber(Node):
    """摄像头订阅节点 - 支持图像处理链"""

    def __init__(self, processor_chain: ProcessorChain = None):
        super().__init__('camera_subscriber')

        # 声明参数
        self.declare_parameter('topic_name', '/camera/image_raw')
        self.declare_parameter('show_window', True)
        self.declare_parameter('show_original', False)

        # 获取参数
        self.topic_name = self.get_parameter('topic_name').value
        self.show_window = self.get_parameter('show_window').value
        self.show_original = self.get_parameter('show_original').value

        # CvBridge转换器
        self.bridge = CvBridge()

        # 图像处理链
        self.processor_chain = processor_chain or ProcessorChain()

        # 创建订阅者
        self.subscription = self.create_subscription(
            Image,
            self.topic_name,
            self.image_callback,
            10
        )

        # 帧计数
        self.frame_count = 0

        self.get_logger().info(
            f'摄像头订阅节点已启动 | 话题: {self.topic_name} | '
            f'处理器数量: {len(self.processor_chain)}'
        )

    def set_processor(self, processor: BaseProcessor):
        """设置单个处理器"""
        self.processor_chain.clear().add(processor)

    def add_processor(self, processor: BaseProcessor):
        """添加处理器到链"""
        self.processor_chain.add(processor)

    def image_callback(self, msg: Image):
        """图像回调函数"""
        try:
            # ROS2 Image消息转换为OpenCV格式
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.frame_count += 1

            # 执行图像处理
            processed = self.processor_chain.process(frame)

            # 显示图像窗口
            if self.show_window:
                if self.show_original:
                    cv2.imshow('Original', frame)
                cv2.imshow('Processed', processed)
                cv2.waitKey(1)

            # 每100帧打印一次信息
            if self.frame_count % 100 == 0:
                self.get_logger().info(f'已处理 {self.frame_count} 帧')

        except Exception as e:
            self.get_logger().error(f'图像处理失败: {e}')

    def destroy(self):
        """清理资源"""
        if self.show_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def create_default_processor() -> ProcessorChain:
    """创建默认处理器链"""
    from image_processor import ResizeProcessor, DrawInfoProcessor

    chain = ProcessorChain()
    chain.add(ResizeProcessor(640, 480))
    chain.add(DrawInfoProcessor("RM CV Task", (10, 30)))
    return chain


def main(args=None):
    rclpy.init(args=args)

    # 创建处理链
    processor_chain = create_default_processor()

    # 创建节点
    node = CameraSubscriber(processor_chain)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
