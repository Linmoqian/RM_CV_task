#!/usr/bin/env python3
"""
装甲板检测演示脚本
支持图片、视频、摄像头三种输入模式
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# 添加本地模块路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from armor_detector import ArmorDetector


def process_image(detector: ArmorDetector, image_path: str):
    """处理单张图片"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return

    result = detector.detect(img)
    vis = detector.draw_result(img, result)

    print(f"检测到 {len(result.armors)} 个装甲板")
    for i, armor in enumerate(result.armors):
        print(f"  [{i+1}] {armor.color.value}-{armor.size.value} @ {armor.center}")

    cv2.imshow("Armor Detection", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(detector: ArmorDetector, video_path: str):
    """处理视频文件"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect(frame)
        vis = detector.draw_result(frame, result)

        cv2.imshow("Armor Detection", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_camera(detector: ArmorDetector, camera_id: int = 0):
    """处理摄像头输入"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_id}")
        return

    # 设置低曝光
    cap.set(cv2.CAP_PROP_EXPOSURE, -10)

    print("摄像头已启动，按 'q' 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect(frame)
        vis = detector.draw_result(frame, result)

        # 显示FPS
        if result.timestamp > 0:
            fps = 1.0 / result.timestamp
            cv2.putText(
                vis,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        # 显示检测数量
        cv2.putText(
            vis,
            f"Armors: {len(result.armors)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Armor Detection - Camera", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="装甲板检测演示")
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="输入图片路径"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        help="输入视频路径"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="摄像头ID (默认: 0)"
    )

    args = parser.parse_args()

    detector = ArmorDetector()

    if args.image:
        process_image(detector, args.image)
    elif args.video:
        process_video(detector, args.video)
    else:
        process_camera(detector, args.camera)


if __name__ == "__main__":
    main()
