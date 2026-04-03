# ROS2 装甲板识别功能包

## 目录结构

```
armor_detection/
├── armor_detection/           # Python模块
│   ├── __init__.py
│   ├── video_publisher.py     # 视频发布节点
│   ├── camera_publisher.py    # 摄像头发布节点
│   └── armor_processor.py     # 装甲板处理节点
├── launch/                    # 启动文件
│   └── armor_detection.launch.py
├── config/                    # 配置文件
│   └── armor_config.yaml
├── resource/
├── package.xml
├── setup.py
└── setup.cfg
```

## 构建

```bash
cd ~/projects/RM_CV_task/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select armor_detection
source install/setup.bash
```

## 运行

### 方式1：使用视频文件

```bash
# 设置视频路径
export VIDEO_PATH=~/projects/RM_CV_task/CV_task/赛事题/armor/images/blue.mp4

# 运行
ros2 run armor_detection video_publisher --ros-args -p video_path:=$VIDEO_PATH &
ros2 run armor_detection armor_processor
```

### 方式2：使用摄像头

```bash
ros2 run armor_detection camera_publisher &
ros2 run armor_detection armor_processor
```

### 方式3：使用启动文件

```bash
# 视频文件
ros2 launch armor_detection armor_detection.launch.py video_path:=/path/to/video.mp4

# 摄像头
ros2 launch armor_detection armor_detection.launch.py use_camera:=true
```

## 话题

| 话题 | 类型 | 方向 | 描述 |
|------|------|------|------|
| `/camera/image_raw` | sensor_msgs/Image | 发布 | 原始图像 |
| `/armor/result` | sensor_msgs/Image | 发布 | 处理结果图像 |
| `/armor/digit` | std_msgs/String | 发布 | 识别的数字结果 |

## 参数

### armor_processor

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| input_topic | string | /camera/image_raw | 输入话题 |
| show_window | bool | true | 是否显示窗口 |
| min_area | int | 100 | 灯条最小面积 |
| top_k | int | 4 | 最大轮廓数 |

### video_publisher

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| video_path | string | "" | 视频路径 |
| loop | bool | true | 是否循环 |
| frame_rate | float | 30.0 | 帧率 |

### camera_publisher

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| camera_id | int | 0 | 摄像头ID |
| width | int | 640 | 分辨率宽度 |
| height | int | 480 | 分辨率高度 |
