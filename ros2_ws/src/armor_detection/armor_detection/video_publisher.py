#!/usr/bin/env python3
"""
ROS2 视频发布节点（无cv_bridge版本）
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
import cv2
import numpy as np


class VideoPublisher(Node):
    """视频文件发布节点"""

    def __init__(self):
        super().__init__('video_publisher')

        self.declare_parameter('video_path', '')
        self.declare_parameter('frame_rate', 30.0)
        self.declare_parameter('loop', True)
        self.declare_parameter('topic_name', '/camera/image_raw')

        self.video_path = self.get_parameter('video_path').value
        self.frame_rate = self.get_parameter('frame_rate').value
        self.loop = self.get_parameter('loop').value
        self.topic_name = self.get_parameter('topic_name').value

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.publisher = self.create_publisher(Image, self.topic_name, qos)

        if not self.video_path:
            self.get_logger().error('请设置 video_path 参数')
            return

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f'无法打开视频: {self.video_path}')
            return

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or self.frame_rate
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.frame_count = 0
        self.get_logger().info(
            f'视频发布节点已启动\n'
            f'  视频路径: {self.video_path}\n'
            f'  帧率: {self.fps:.1f}\n'
            f'  总帧数: {self.total_frames}\n'
            f'  话题: {self.topic_name}'
        )

    def timer_callback(self):
        ret, frame = self.cap.read()

        if not ret:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_count = 0
                ret, frame = self.cap.read()
                if not ret:
                    return
            else:
                self.get_logger().info('视频播放完成')
                self.timer.cancel()
                return

        # OpenCV BGR转ROS Image消息
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
            self.get_logger().info(f'已发布 {self.frame_count}/{self.total_frames} 帧')

    def destroy(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
