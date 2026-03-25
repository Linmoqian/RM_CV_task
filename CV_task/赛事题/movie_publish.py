#!/usr/bin/env python3
"""
ROS2摄像头帧发布节点
发布本地摄像头的图像帧到 /camera/image_raw 话题
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraPublisher(Node):
    """摄像头发布节点"""

    def __init__(self):
        super().__init__('camera_publisher')

        # 声明参数
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('frame_rate', 30)
        self.declare_parameter('topic_name', '/camera/image_raw')

        # 获取参数
        self.camera_id = self.get_parameter('camera_id').value
        self.frame_rate = self.get_parameter('frame_rate').value
        self.topic_name = self.get_parameter('topic_name').value

        # 创建发布者
        self.publisher = self.create_publisher(Image, self.topic_name, 10)

        # CvBridge转换器
        self.bridge = CvBridge()

        # 初始化摄像头
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            self.get_logger().error(f'无法打开摄像头 {self.camera_id}')
            return

        # 设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 创建定时器
        timer_period = 1.0 / self.frame_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            f'摄像头发布节点已启动 | 设备: {self.camera_id} | '
            f'帧率: {self.frame_rate} | 话题: {self.topic_name}'
        )

    def timer_callback(self):
        """定时器回调函数"""
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('读取摄像头帧失败')
            return

        # OpenCV BGR格式转换为ROS2 Image消息
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'

        self.publisher.publish(msg)

    def destroy(self):
        """清理资源"""
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
