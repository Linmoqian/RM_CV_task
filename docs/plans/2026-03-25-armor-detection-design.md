# 装甲板识别系统设计文档

## 概述

基于传统视觉方法的RoboMaster装甲板识别系统，支持红蓝双色、大小装甲板检测。

## 需求

| 项目 | 描述 |
|------|------|
| 识别目标 | 红/蓝双色 + 大/小装甲板 |
| 输入源 | 实时摄像头(ROS2) + 图片/视频文件 |
| 输出信息 | 位置 + 颜色 + 尺寸 + 四角坐标 |
| 性能目标 | 精度优先，30 FPS |

## 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    ArmorDetector                         │
├─────────────────────────────────────────────────────────┤
│  输入层        │  处理层              │  输出层          │
│  ────────      │  ────────            │  ────────        │
│  - 图像帧      │  - 颜色分割          │  - 检测结果列表  │
│  (相机/文件)   │  - 灯条提取          │  - 绘制可视化    │
│                │  - 灯条匹配          │                  │
│                │  - 装甲板筛选        │                  │
└─────────────────────────────────────────────────────────┘
```

**核心模块：**

| 模块 | 职责 |
|------|------|
| `ColorSegmenter` | HSV颜色分割，提取红/蓝灯条区域 |
| `LightBarFinder` | 轮廓检测 + 几何约束，筛选有效灯条 |
| `ArmorMatcher` | 灯条配对 + 装甲板分类 |
| `ArmorDetector` | 整合以上模块，提供统一接口 |

## 处理流程

```
原图 → 预处理 → 颜色分割 → 轮廓检测 → 灯条筛选 → 灯条匹配 → 装甲板输出
        │           │           │           │           │
        ▼           ▼           ▼           ▼           ▼
      降噪/缩放   HSV阈值    findContours  几何约束    配对规则
```

| 步骤 | 操作 | 目的 |
|------|------|------|
| 1. 预处理 | 高斯模糊 + 缩放 | 降噪、统一尺寸 |
| 2. 颜色分割 | HSV阈值分割 | 提取红/蓝灯条区域 |
| 3. 轮廓检测 | `findContours` | 获取连通域 |
| 4. 灯条筛选 | 长宽比、面积、凸性 | 过滤噪声 |
| 5. 灯条匹配 | 平行度、间距、中心对齐 | 识别装甲板 |
| 6. 装甲板分类 | 灯条间距/比例 | 区分大/小装甲板 |

## 数据结构

```python
@dataclass
class LightBar:
    """灯条"""
    contour: np.ndarray       # 原始轮廓
    rect: RotatedRect          # 最小外接矩形
    center: Point              # 中心点
    size: Size                 # 宽高
    angle: float               # 倾斜角度
    color: ArmorColor          # 颜色 (RED/BLUE)

@dataclass
class Armor:
    """装甲板"""
    left_light: LightBar       # 左灯条
    right_light: LightBar      # 右灯条
    corners: List[Point]       # 四角坐标 (左上/右上/右下/左下)
    center: Point              # 中心点
    color: ArmorColor          # 颜色
    size: ArmorSize            # 尺寸 (LARGE/SMALL)
    confidence: float          # 置信度

@dataclass
class DetectResult:
    """检测结果"""
    armors: List[Armor]        # 检测到的装甲板列表
    frame_id: int              # 帧ID
    timestamp: float           # 时间戳
```

**枚举定义：**
```python
class ArmorColor(Enum):
    RED = "red"
    BLUE = "blue"

class ArmorSize(Enum):
    LARGE = "large"   # 英雄装甲板
    SMALL = "small"   # 步兵/哨兵装甲板
```

## 几何约束规则

### 灯条筛选条件

| 约束 | 条件 | 说明 |
|------|------|------|
| 面积 | `min_area < area < max_area` | 过滤噪点和过大区域 |
| 长宽比 | `1.5 < h/w < 6` | 灯条为细长条 |
| 凸性 | `isConvex = True` | 灯条轮廓应凸 |
| 填充率 | `area/rect_area > 0.6` | 轮廓接近矩形 |

### 灯条匹配条件

| 约束 | 条件 | 说明 |
|------|------|------|
| 同色 | `left.color == right.color` | 两灯条颜色一致 |
| 平行度 | `\|angle_diff\| < 15°` | 灯条近似平行 |
| 高度比 | `0.7 < h_ratio < 1.3` | 高度相近 |
| 间距比 | `1.0 < distance/h_avg < 4.5` | 间距合理 |
| 垂直对齐 | `\|y_diff\| < 0.5 * h_avg` | 中心垂直对齐 |

### 大/小装甲板区分

```python
# 根据灯条间距与灯条高度的比值判断
if 1.0 < distance/height < 2.5:
    size = SMALL   # 小装甲板
else:
    size = LARGE   # 大装甲板
```

## 文件结构

```
赛事题/
├── armor_detector/
│   ├── __init__.py
│   ├── detector.py      # ArmorDetector 主类
│   ├── light_bar.py     # LightBar 类 + 灯条提取
│   ├── armor.py         # Armor 类 + 灯条匹配
│   ├── types.py         # 枚举 + 数据类定义
│   └── config.py        # 可调参数配置
├── image_processor.py   # 已有处理模块
├── movie_publish.py     # ROS2发布节点
└── movie_reader.py      # ROS2订阅节点（集成检测器）
```

## 核心接口

```python
class ArmorDetector:
    def __init__(self, config: dict = None)
    def detect(self, frame: np.ndarray) -> DetectResult
    def draw_result(self, frame: np.ndarray, result: DetectResult) -> np.ndarray
```

## 使用示例

```python
detector = ArmorDetector()
result = detector.detect(frame)
vis_frame = detector.draw_result(frame, result)

for armor in result.armors:
    print(f"颜色: {armor.color}, 尺寸: {armor.size}")
    print(f"中心: {armor.center}, 角点: {armor.corners}")
```
