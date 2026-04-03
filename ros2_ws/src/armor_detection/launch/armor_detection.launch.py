#!/usr/bin/env python3
"""
ROS2 装甲板识别启动文件
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 声明参数
    video_path_arg = DeclareLaunchArgument(
        'video_path',
        default_value='',
        description='视频文件路径'
    )

    show_window_arg = DeclareLaunchArgument(
        'show_window',
        default_value='true',
        description='是否显示处理窗口'
    )

    input_topic_arg = DeclareLaunchArgument(
        'input_topic',
        default_value='/camera/image_raw',
        description='输入图像话题'
    )

    # 视频发布节点
    video_publisher_node = Node(
        package='armor_detection',
        executable='video_publisher',
        name='video_publisher',
        parameters=[{
            'video_path': LaunchConfiguration('video_path'),
            'loop': True,
            'topic_name': LaunchConfiguration('input_topic'),
        }],
        output='screen'
    )

    # 装甲板处理节点
    armor_processor_node = Node(
        package='armor_detection',
        executable='armor_processor',
        name='armor_processor',
        parameters=[{
            'input_topic': LaunchConfiguration('input_topic'),
            'show_window': LaunchConfiguration('show_window'),
        }],
        output='screen'
    )

    return LaunchDescription([
        video_path_arg,
        show_window_arg,
        input_topic_arg,
        video_publisher_node,
        armor_processor_node,
    ])
