"""
LeNet-5 手写数字推理脚本
使用 OpenCV 创建画板，实时识别手写数字
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class Colors:
    """终端颜色定义"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


class LeNet5(nn.Module):
    """LeNet-5 模型定义"""

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = self.activation(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class DigitRecognizer:
    """手写数字识别器"""

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(SCRIPT_DIR, 'lenet5_best.pth')
        elif not os.path.isabs(model_path):
            model_path = os.path.join(SCRIPT_DIR, model_path)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"{Colors.CYAN}使用设备: {self.device}{Colors.RESET}")

        # 加载模型
        self.model = LeNet5().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        print(f"{Colors.GREEN}模型加载成功: {model_path}{Colors.RESET}")

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])

        # 画板参数
        self.canvas_size = 400
        self.canvas = np.ones((self.canvas_size, self.canvas_size), dtype=np.uint8) * 255
        self.drawing = False
        self.brush_size = 8  # 细一些的笔画
        self.last_x = -1
        self.last_y = -1
        self.need_predict = False  # 是否需要预测

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """预处理图像"""
        # 反转颜色：白底黑字 -> 黑底白字 (与 MNIST 格式一致)
        image = 255 - image

        # 转换为 PIL Image 并应用变换
        from PIL import Image
        image = Image.fromarray(image)
        image = self.transform(image)
        return image.unsqueeze(0).to(self.device)

    def predict(self, image: np.ndarray) -> tuple:
        """预测数字，返回预测结果和所有概率"""
        with torch.no_grad():
            tensor = self.preprocess(image)
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = probs.max(1)
            all_probs = probs[0].cpu().numpy()  # 获取所有数字的概率
            return predicted.item(), confidence.item(), all_probs

    def draw_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.last_x, self.last_y = x, y
            cv2.circle(self.canvas, (x, y), self.brush_size, 0, -1)
            self.need_predict = True
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # 用线段连接相邻点，避免断触
            cv2.line(self.canvas, (self.last_x, self.last_y), (x, y), 0, self.brush_size * 2)
            self.last_x, self.last_y = x, y
            self.need_predict = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.last_x, self.last_y = -1, -1
            self.need_predict = True

    def clear_canvas(self):
        """清空画板"""
        self.canvas = np.ones((self.canvas_size, self.canvas_size), dtype=np.uint8) * 255
        self.last_x, self.last_y = -1, -1

    def draw_probability_panel(self, probs: np.ndarray, predicted: int) -> np.ndarray:
        """绘制概率面板"""
        panel_width = 200
        panel_height = self.canvas_size
        panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 240

        bar_height = 30
        bar_max_width = 140
        start_y = 35

        # 标题
        cv2.putText(panel, 'Probability', (45, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

        for i in range(10):
            y = start_y + i * 38
            prob = probs[i]
            bar_width = int(prob * bar_max_width)

            # 背景条
            cv2.rectangle(panel, (50, y), (50 + bar_max_width, y + bar_height), (200, 200, 200), -1)

            # 概率条 - 高亮预测结果
            color = (0, 200, 0) if i == predicted else (100, 150, 255)
            if bar_width > 0:
                cv2.rectangle(panel, (50, y), (50 + bar_width, y + bar_height), color, -1)

            # 数字标签
            cv2.putText(panel, str(i), (20, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

            # 百分比
            pct_text = f'{prob*100:.1f}%'
            cv2.putText(panel, pct_text, (55 + bar_max_width, y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1)

        return panel

    def run(self):
        """运行交互式识别"""
        window_name = 'LeNet-5 Digit Recognition'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.draw_callback)

        # 初始概率
        current_probs = np.zeros(10)
        predicted_digit = -1

        print(f"\n{Colors.CYAN}操作说明:{Colors.RESET}")
        print(f"  {Colors.BLUE}鼠标左键{Colors.RESET} - 书写数字 (实时识别)")
        print(f"  {Colors.BLUE}空格键{Colors.RESET}   - 清空画板")
        print(f"  {Colors.BLUE}ESC/Q{Colors.RESET}     - 退出程序\n")

        while True:
            # 如果有绘制动作，进行实时预测
            if self.need_predict:
                predicted_digit, confidence, current_probs = self.predict(self.canvas)
                self.need_predict = False

            # 创建显示图像
            canvas_bgr = cv2.cvtColor(self.canvas, cv2.COLOR_GRAY2BGR)

            # 绘制提示文字
            cv2.putText(canvas_bgr, 'Draw digit (0-9)', (90, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            cv2.putText(canvas_bgr, '[Space]Clear [Q]Quit', (100, 385),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

            # 绘制概率面板
            prob_panel = self.draw_probability_panel(current_probs, predicted_digit)

            # 拼接画板和概率面板
            display = np.hstack([canvas_bgr, prob_panel])

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                # 清空
                self.clear_canvas()
                current_probs = np.zeros(10)
                predicted_digit = -1
                self.need_predict = False

            elif key == 27 or key == ord('q'):
                # 退出
                break

        cv2.destroyAllWindows()
        print(f"\n{Colors.CYAN}程序已退出{Colors.RESET}")


def predict_image(image_path: str, model_path: str = None):
    """从图像文件预测"""
    recognizer = DigitRecognizer(model_path)

    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"{Colors.RED}无法读取图像: {image_path}{Colors.RESET}")
        return

    # preprocess 会自动处理颜色反转
    digit, confidence, probs = recognizer.predict(image)
    print(f"{Colors.GREEN}识别结果: {digit}  置信度: {confidence*100:.2f}%{Colors.RESET}")
    print(f"{Colors.BLUE}各数字概率:{Colors.RESET}")
    for i, p in enumerate(probs):
        bar = '█' * int(p * 20)
        print(f"  {i}: {bar} {p*100:.1f}%")
    return digit, confidence


def main():
    """主函数"""
    import sys

    if len(sys.argv) > 1:
        # 从文件预测
        predict_image(sys.argv[1])
    else:
        # 交互式画板
        recognizer = DigitRecognizer()
        recognizer.run()


if __name__ == '__main__':
    main()
