#!/usr/bin/env python3
"""
ROS2 摄像头发布节点（无cv_bridge版本）
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
import cv2


class CameraPublisher(Node):
    """摄像头发布节点"""

    def __init__(self):
        super().__init__('camera_publisher')

        self.declare_parameter('camera_id', 0)
        self.declare_parameter('frame_rate', 30)
        self.declare_parameter('topic_name', '/camera/image_raw')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)

        self.camera_id = self.get_parameter('camera_id').value
        self.frame_rate = self.get_parameter('frame_rate').value
        self.topic_name = self.get_parameter('topic_name').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.publisher = self.create_publisher(Image, self.topic_name, qos)

        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            self.get_logger().error(f'无法打开摄像头 {self.camera_id}')
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        timer_period = 1.0 / self.frame_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.frame_count = 0
        self.get_logger().info(
            f'摄像头发布节点已启动\n'
            f'  设备: {self.camera_id}\n'
            f'  分辨率: {self.width}x{self.height}\n'
            f'  帧率: {self.frame_rate}\n'
            f'  话题: {self.topic_name}'
        )

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('读取摄像头帧失败')
            return

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        msg.height = frame.shape[0]
        msg.width = frame.shape[1]
        msg.encoding = 'bgr8'
        msg.is_bigendian = False
        msg.step = frame.shape[1] * 3
        msg.data = frame.tobytes()

        self.publisher.publish(msg)

        self.frame_count += 1
        if self.frame_count % 100 == 0:
            self.get_logger().info(f'已发布 {self.frame_count} 帧')

    def destroy(self):
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
