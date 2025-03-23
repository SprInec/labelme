"""
AI自动化模块

此模块包含用于自动标注的AI功能：
- 目标检测：使用YOLOv7检测图像中的对象
- 人体姿态估计：检测图像中的人体姿态
"""

from labelme._automation.config_loader import ConfigLoader
from labelme._automation.pose_estimation import PoseEstimator
