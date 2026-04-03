"""
RoboMaster 能量机关视觉识别 - 轮廓检测
"""

import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent


def separate_red(frame):
    """分离红色通道"""
    b, g, r = cv2.split(frame)

    _, r_thresh = cv2.threshold(r, 200, 255, cv2.THRESH_BINARY)
    _, b_thresh = cv2.threshold(b, 100, 255, cv2.THRESH_BINARY_INV)
    _, g_thresh = cv2.threshold(g, 100, 255, cv2.THRESH_BINARY_INV)

    mask = cv2.bitwise_and(r_thresh, cv2.bitwise_and(b_thresh, g_thresh))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=2)

    return cv2.bitwise_and(frame, frame, mask=mask), mask


def find_target_contour(mask):
    """发现目标轮廓"""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return None

    best_id = -1
    best_area = -1

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 20 or area > 10000:
            continue
        # 无父轮廓且无子轮廓
        if hierarchy[0][i][3] >= 0 or hierarchy[0][i][2] >= 0:
            continue
        if area > best_area:
            best_area = area
            best_id = i

    return contours[best_id] if best_id >= 0 else None


def main():
    video_path = ROOT / "images" / "3_red.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"无法打开: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (320, 240))
        result, mask = separate_red(frame)

        contour = find_target_contour(mask)
        if contour is not None:
            cv2.drawContours(result, [contour], -1, (0, 255, 255), 1)

            m = cv2.moments(contour)
            if m["m00"] != 0:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
                cv2.circle(result, (cx, cy), 2, (255, 0, 0), -1)

        display = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), result])
        cv2.imshow("Original | Mask | Result", display)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
