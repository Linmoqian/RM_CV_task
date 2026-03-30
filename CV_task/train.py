"""
YOLO 数字分类训练脚本
数据集: 数字0-8分类 (9类)
"""

import os
import random
import shutil
from pathlib import Path
from ultralytics import YOLO

# 设置代理
os.environ["https_proxy"] = "http://127.0.0.1:7897"


def split_dataset(source_dir: Path, target_dir: Path, train_ratio: float = 0.8, seed: int = 42):
    """正确划分多类别数据集"""
    random.seed(seed)
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    # 获取所有类别文件夹
    class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]

    print(f"\033[36m发现 {len(class_dirs)} 个类别\033[0m")

    # 创建目标目录结构
    for split in ["train", "val"]:
        for class_dir in class_dirs:
                (target_dir / split / class_dir.name).mkdir(parents=True, exist_ok=True)

    # 划分每个类别的图片
    total_train, total_val = 0, 0
    for class_dir in class_dirs:
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        random.shuffle(images)

        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # 复制训练集
        for img in train_images:
                shutil.copy2(img, target_dir / "train" / class_dir.name / img.name)
        # 复制验证集
        for img in val_images:
            shutil.copy2(img, target_dir / "val" / class_dir.name / img.name)

        total_train += len(train_images)
        total_val += len(val_images)

    print(f"\033[32m划分完成: {total_train} 训练, {total_val} 验证\033[0m")


def main():
    dataset_dir = Path(__file__).parent / "dataset"
    split_dir = Path(__file__).parent / "dataset_split"

    # 划分数据集
    if not split_dir.exists():
        split_dataset(dataset_dir, split_dir)
    else:
        print("\033[32m数据集已划分\033[0m")

    # 加载模型
    print("\033[36m加载模型: yolo11s-cls.pt\033[0m")
    model = YOLO("yolo11s-cls.pt")

    # 训练
    model.train(
        data=str(split_dir),
        epochs=100,
        imgsz=224,
        batch=32,
        device="0",
        workers=4,
        project="runs/classify",
        name="digit_classifier",
        exist_ok=True,
    )

    # 验证
    print("\033[36m验证模型...\033[0m")
    metrics = model.val()
    print(f"\033[32mTop-1: {metrics.top1:.4f}\033[0m")

    # 导出
    print("\033[36m导出ONNX...\033[0m")
    model.export(format="onnx", imgsz=224)

    print("\033[32m完成!\033[0m")


if __name__ == "__main__":
    main()
