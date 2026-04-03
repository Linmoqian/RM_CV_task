#!/usr/bin/env python3
"""
ROS2 装甲板识别处理节点（无cv_bridge版本）
"""

import sys
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
from ultralytics import YOLO

# 添加armor模块路径
ARMOR_SRC = Path("/home/lin/projects/RM_CV_task/CV_task/赛事题/armor/src")
if str(ARMOR_SRC) not in sys.path:
    sys.path.insert(0, str(ARMOR_SRC))

from datect_armor import detect_armor
from pnp import solve_armor_pnp, draw_pnp_info


def imgmsg_to_cv2(msg):
    """ROS Image消息转OpenCV"""
    if msg.encoding == 'rgb8':
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    elif msg.encoding == 'bgr8':
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    elif msg.encoding == 'mono8':
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
    else:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")


def cv2_to_imgmsg(img, encoding='bgr8'):
    """OpenCV转ROS Image消息"""
    msg = Image()
    msg.height = img.shape[0]
    msg.width = img.shape[1]
    msg.encoding = encoding
    msg.step = img.strides[0]
    msg.data = img.tobytes()
    return msg


class ArmorProcessorNode(Node):
    """装甲板识别处理节点"""

    def __init__(self):
        super().__init__('armor_processor')

        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('output_topic', '/armor/result')
        self.declare_parameter('digit_topic', '/armor/digit')
        self.declare_parameter('show_window', False)
        self.declare_parameter('min_area', 100)
        self.declare_parameter('top_k', 4)

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.digit_topic = self.get_parameter('digit_topic').value
        self.show_window = self.get_parameter('show_window').value
        self.min_area = self.get_parameter('min_area').value
        self.top_k = self.get_parameter('top_k').value

        # 加载模型
        model_path = Path("/home/lin/projects/RM_CV_task/CV_task/runs/classify/runs/classify/digit_classifier/weights/best.pt")
        self.get_logger().info(f'加载分类模型: {model_path}')
        self.classifier = YOLO(str(model_path))
        self.class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.subscription = self.create_subscription(Image, self.input_topic, self.image_callback, qos)
        self.image_publisher = self.create_publisher(Image, self.output_topic, qos)
        self.digit_publisher = self.create_publisher(String, self.digit_topic, qos)

        self.frame_count = 0
        self.get_logger().info(f'装甲板处理节点已启动，订阅: {self.input_topic}')

    def image_callback(self, msg: Image):
        try:
            frame = imgmsg_to_cv2(msg)
            self.frame_count += 1

            result_img, digit_result = self.process_frame(frame)

            result_msg = cv2_to_imgmsg(result_img, 'bgr8')
            result_msg.header = msg.header
            self.image_publisher.publish(result_msg)

            if digit_result:
                digit_msg = String()
                digit_msg.data = f"{digit_result['label']}:{digit_result['confidence']:.4f}:{digit_result['distance']:.2f}"
                self.digit_publisher.publish(digit_msg)
                self.get_logger().info(f"检测到: {digit_result['label']} ({digit_result['confidence']:.1%})")

            if self.show_window:
                cv2.imshow('Armor Detection', result_img)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'处理失败: {e}')

    def process_frame(self, frame):
        img = frame.copy()
        armor_crop = None
        digit_result = None

        b = cv2.split(img)[0]
        _, b_bin = cv2.threshold(b, 168, 255, cv2.THRESH_BINARY)

        armor_points, armor_crop = detect_armor(b_bin, img, self.min_area, self.top_k, frame)

        if armor_points is not None:
            points_2d = [armor_points[0], armor_points[2], armor_points[3], armor_points[1]]
            success, rvec, tvec, distance = solve_armor_pnp(points_2d)

            if success:
                img = draw_pnp_info(img, points_2d, rvec, tvec, distance)

            if armor_crop is not None and armor_crop.size > 0:
                # 预处理
                gray = cv2.cvtColor(armor_crop, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

                results = self.classifier.predict(processed, verbose=False)
                if results:
                    probs = results[0].probs
                    class_id = probs.top1
                    confidence = probs.top1conf.item()
                    label = self.class_names[class_id]

                    h, w = armor_crop.shape[:2]
                    cv2.putText(armor_crop, f"{label} ({int(confidence * 100)}%)", (10, h - 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    digit_result = {'label': label, 'confidence': confidence, 'distance': distance if success else 0.0}

        return self.merge_display(b_bin, img, armor_crop), digit_result

    def merge_display(self, binary, result, crop=None):
        h, w = result.shape[:2]
        if len(binary.shape) == 2:
            binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        if crop is None:
            crop = np.zeros((h // 2, w // 2, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (w // 2, h // 2))
        binary = cv2.resize(binary, (w // 2, h // 2))
        result_small = cv2.resize(result, (w // 2, h // 2))
        top_row = np.hstack([binary, result_small])
        crop_padded = cv2.copyMakeBorder(crop, 0, 0, w // 4, w // 4, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return np.vstack([top_row, crop_padded])

    def destroy(self):
        if self.show_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ArmorProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
