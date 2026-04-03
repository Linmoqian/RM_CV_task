# 装甲板检测

## 功能

- 检测红/蓝装甲板灯条
- 灯条配对识别装甲板
- PnP 位姿估计（距离、角度）
- YOLO 数字分类识别

## 文件

```
armor/
├── src/
│   ├── main.py           # 主程序入口
│   ├── datect_armor.py   # 装甲板检测算法
│   ├── pnp.py            # PnP位姿求解
│   └── display.py        # 显示工具
└── images/
    ├── blue.mp4          # 蓝色装甲板测试视频
    └── red.mp4           # 红色装甲板测试视频
```

## 运行

```bash
cd CV_task/赛事题/armor/src
python main.py
```

## 算法流程

1. 蓝色通道二值化（阈值168）
2. 轮廓检测 + 面积筛选
3. 灯条配对（面积差最小）
4. PnP 求解位姿
5. YOLO 分类识别数字
