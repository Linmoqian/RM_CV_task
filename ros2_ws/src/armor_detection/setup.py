from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'armor_detection'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 启动文件
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # 配置文件
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lin',
    maintainer_email='lin@example.com',
    description='ROS2 装甲板识别功能包',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'video_publisher = armor_detection.video_publisher:main',
            'armor_processor = armor_detection.armor_processor:main',
            'camera_publisher = armor_detection.camera_publisher:main',
        ],
    },
)
