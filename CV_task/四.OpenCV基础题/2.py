"""
摄像头录制程序
功能：
1. 打开摄像头并显示画面
2. 显示视频大小和帧率
3. 滑动条调节曝光和亮度
4. 录制视频保存到本地
"""

import cv2
import os
from datetime import datetime

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("\033[31m错误：无法打开摄像头\033[0m")
    exit(1)

# 获取摄像头参数
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

print(f"\033[32m摄像头已打开\033[0m")
print(f"分辨率: {width}x{height}, FPS: {fps}")

# 初始化参数
exposure_val = 0
brightness_val = 128
is_recording = False
writer = None

# 滑动条回调函数
def on_exposure(val):
    """曝光调节回调"""
    global exposure_val
    exposure_val = val - 10  # 转换为实际曝光值范围
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure_val)

def on_brightness(val):
    """亮度调节回调"""
    global brightness_val
    brightness_val = val
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness_val)

# 创建窗口和滑动条
window_name = 'Camera Record'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

# 创建滑动条
cv2.createTrackbar('Exposure', window_name, 10, 20, on_exposure)
cv2.createTrackbar('Brightness', window_name, 128, 255, on_brightness)

# 提示信息
print("\033[36m操作说明:\033[0m")
print("  按 \033[33m空格键\033[0m 开始/停止录制")
print("  按 \033[33mESC\033[0m 退出程序")

frame_count = 0
start_time = datetime.now()

while True:
    ret, frame = cap.read()
    if not ret:
        print("\033[31m读取帧失败\033[0m")
        break

    # 计算实时FPS
    frame_count += 1
    elapsed = (datetime.now() - start_time).total_seconds()
    current_fps = frame_count / elapsed if elapsed > 0 else 0

    # 在画面上显示信息
    info_text = f"Size: {width}x{height} | FPS: {current_fps:.1f}"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 显示录制状态
    if is_recording:
        cv2.putText(frame, "REC", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.circle(frame, (80, 55), 8, (0, 0, 255), -1)
        writer.write(frame)

    # 显示参数信息
    param_text = f"Exposure: {exposure_val} | Brightness: {brightness_val}"
    cv2.putText(frame, param_text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    cv2.imshow(window_name, frame)

    # 按键检测
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC退出
        break
    elif key == ord(' '):  # 空格开始/停止录制
        if not is_recording:
            # 开始录制
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(script_dir, f'record_{timestamp}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            is_recording = True
            print(f"\033[32m开始录制: {output_path}\033[0m")
        else:
            # 停止录制
            is_recording = False
            if writer:
                writer.release()
                print(f"\033[32m录制完成\033[0m")

# 清理资源
cap.release()
if writer:
    writer.release()
cv2.destroyAllWindows()
print("\033[32m程序已退出\033[0m")
