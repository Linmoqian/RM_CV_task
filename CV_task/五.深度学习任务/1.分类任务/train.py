"""
LeNet-5 手写数字分类训练脚本
使用 MNIST 数据集训练 LeNet-5 卷积神经网络
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime


# 颜色输出工具
class Colors:
    """终端颜色定义"""
    GREEN = '\033[92m'    # 成功
    YELLOW = '\033[93m'   # 警告
    RED = '\033[91m'      # 错误
    CYAN = '\033[96m'     # 交互提示
    BLUE = '\033[94m'     # 高亮和链接
    GRAY = '\033[90m'     # 次要信息
    RESET = '\033[0m'     # 重置


def cprint(msg: str, color: str = Colors.RESET):
    """彩色打印"""
    print(f"{color}{msg}{Colors.RESET}")


class LeNet5(nn.Module):
    """
    LeNet-5 卷积神经网络

    网络结构:
    - 输入: 32x32 灰度图像
    - C1: 卷积层 (6 filters, 5x5) -> 28x28x6
    - S2: 池化层 (2x2 avg pool) -> 14x14x6
    - C3: 卷积层 (16 filters, 5x5) -> 10x10x16
    - S4: 池化层 (2x2 avg pool) -> 5x5x16
    - C5: 卷积层 (120 filters, 5x5) -> 1x1x120
    - F6: 全连接层 (84 neurons)
    - 输出: 全连接层 (10 classes)
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        # 卷积层 (标准 LeNet-5 结构)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)    # 32x32 -> 28x28
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)   # 14x14 -> 10x10
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5) # 5x5 -> 1x1

        # 池化层
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

        # 激活函数
        self.activation = nn.Tanh()

    def forward(self, x):
        # C1 + S2
        x = self.activation(self.conv1(x))
        x = self.pool(x)

        # C3 + S4
        x = self.activation(self.conv2(x))
        x = self.pool(x)

        # C5
        x = self.activation(self.conv3(x))

        # 展平
        x = x.view(x.size(0), -1)

        # F6 + 输出
        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return x


class Trainer:
    """训练器类"""

    def __init__(self, model, device, learning_rate: float = 0.001, momentum: float = 0.9):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        self.train_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train_epoch(self, train_loader: DataLoader) -> tuple:
        """训练一个 epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # 进度显示
            if (batch_idx + 1) % 100 == 0:
                acc = 100.0 * correct / total
                cprint(
                    f"  批次 [{batch_idx + 1}/{len(train_loader)}] "
                    f"损失: {loss.item():.4f} 准确率: {acc:.2f}%",
                    Colors.GRAY
                )

        avg_loss = running_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> float:
        """验证模型"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        return 100.0 * correct / total

    def test(self, test_loader: DataLoader) -> dict:
        """测试模型"""
        self.model.eval()
        correct = 0
        total = 0
        class_correct = [0] * 10
        class_total = [0] * 10

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                # 每个类别的准确率
                for i in range(target.size(0)):
                    label = target[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1

        results = {
            'overall_accuracy': 100.0 * correct / total,
            'class_accuracies': {
                i: 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
                for i in range(10)
            }
        }
        return results


def get_data_loaders(batch_size: int = 64, data_dir: str = './data'):
    """获取数据加载器"""
    cprint("正在加载 MNIST 数据集...", Colors.CYAN)

    # 数据预处理：将 28x28 转换为 32x32 (LeNet-5 输入要求)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])

    # 加载训练集
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # 加载测试集
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows 下建议设为 0
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    cprint(f"训练集样本数: {len(train_dataset)}", Colors.BLUE)
    cprint(f"测试集样本数: {len(test_dataset)}", Colors.BLUE)

    return train_loader, test_loader


def save_model(model: nn.Module, save_dir: str, filename: str):
    """保存模型"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), save_path)
    cprint(f"模型已保存至: {save_path}", Colors.GREEN)


def main():
    """主函数"""
    # 配置参数
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    EPOCHS = 10
    SAVE_DIR = './'

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cprint(f"使用设备: {device}", Colors.CYAN)

    # 加载数据
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)

    # 创建模型
    model = LeNet5()
    cprint(f"\nLeNet-5 模型结构:", Colors.BLUE)
    print(model)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cprint(f"\n总参数量: {total_params:,}", Colors.BLUE)
    cprint(f"可训练参数量: {trainable_params:,}", Colors.BLUE)

    # 创建训练器
    trainer = Trainer(model, device, LEARNING_RATE, MOMENTUM)

    # 训练循环
    cprint(f"\n开始训练 (共 {EPOCHS} 个 epoch)...", Colors.CYAN)
    start_time = datetime.now()

    best_accuracy = 0.0

    for epoch in range(EPOCHS):
        cprint(f"\nEpoch {epoch + 1}/{EPOCHS}", Colors.YELLOW)

        # 训练
        train_loss, train_acc = trainer.train_epoch(train_loader)

        # 验证
        val_acc = trainer.validate(test_loader)

        # 记录
        trainer.train_losses.append(train_loss)
        trainer.train_accuracies.append(train_acc)
        trainer.val_accuracies.append(val_acc)

        # 输出结果
        cprint(
            f"  训练损失: {train_loss:.4f} "
            f"训练准确率: {train_acc:.2f}% "
            f"验证准确率: {val_acc:.2f}%",
            Colors.GREEN
        )

        # 保存最佳模型
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            save_model(model, SAVE_DIR, 'lenet5_best.pth')

    # 训练完成
    elapsed = datetime.now() - start_time
    cprint(f"\n训练完成! 耗时: {elapsed}", Colors.GREEN)
    cprint(f"最佳验证准确率: {best_accuracy:.2f}%", Colors.GREEN)

    # 最终测试
    cprint("\n正在进行最终测试...", Colors.CYAN)
    test_results = trainer.test(test_loader)

    cprint(f"\n测试集总体准确率: {test_results['overall_accuracy']:.2f}%", Colors.GREEN)
    cprint("\n各数字分类准确率:", Colors.BLUE)

    for digit, acc in test_results['class_accuracies'].items():
        cprint(f"  数字 {digit}: {acc:.2f}%", Colors.GRAY)

    # 保存最终模型
    save_model(model, SAVE_DIR, 'lenet5_final.pth')

    cprint("\n训练脚本执行完毕", Colors.GREEN)


if __name__ == '__main__':
    main()
