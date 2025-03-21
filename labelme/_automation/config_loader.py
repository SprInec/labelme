import os
import json
import logging
from typing import Dict, Any, Optional

from loguru import logger

# 配置文件路径
DEFAULT_CONFIG_DIR = os.path.join(
    os.path.expanduser("~"), ".labelme", "ai_models")
DEFAULT_CONFIG_PATH = os.path.join(DEFAULT_CONFIG_DIR, "config.json")


class ConfigLoader:
    """加载和管理AI模型配置"""

    def __init__(self, config_path: str = None):
        """
        初始化配置加载器

        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            # 确保配置目录存在
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"已加载配置文件: {self.config_path}")
                return config
            else:
                logger.warning(f"配置文件不存在: {self.config_path}，将使用默认配置")
                return self._get_default_config()
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}，将使用默认配置")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        default_config = {
            "detection": {
                "model_name": "fasterrcnn_resnet50_fpn",
                "conf_threshold": 0.5,
                "device": "cpu",
                "filter_classes": [],
                "nms_threshold": 0.45,
                "max_detections": 100,
                "use_gpu_if_available": True,
                "advanced": {
                    "pre_nms_top_n": 1000,
                    "pre_nms_threshold": 0.5,
                    "max_size": 1333,
                    "min_size": 800,
                    "score_threshold": 0.05
                }
            },
            "pose_estimation": {
                "model_name": "keypointrcnn_resnet50_fpn",
                "conf_threshold": 0.25,
                "keypoint_threshold": 0.3,
                "device": "cpu",
                "use_gpu_if_available": True,
                "advanced": {
                    "max_poses": 20,
                    "min_keypoints": 5,
                    "use_tracking": False,
                    "tracking_threshold": 0.5
                }
            },
            "yolov7": {
                "model_name": "yolov7",
                "conf_threshold": 0.5,
                "device": "cpu",
                "filter_classes": [],
                "use_gpu_if_available": True,
                "weights_path": "",
                "advanced": {
                    "img_size": 640,
                    "nms_threshold": 0.45,
                    "max_detections": 100
                }
            },
            "rtmdet": {
                "model_name": "rtmdet_s",
                "conf_threshold": 0.5,
                "device": "cpu",
                "filter_classes": [],
                "use_gpu_if_available": True,
                "advanced": {
                    "nms_threshold": 0.45,
                    "max_detections": 100
                }
            },
            "rtmpose": {
                "model_name": "rtmpose_s",
                "conf_threshold": 0.25,
                "keypoint_threshold": 0.3,
                "device": "cpu",
                "use_gpu_if_available": True,
                "advanced": {
                    "max_poses": 20,
                    "min_keypoints": 5
                }
            },
            "keypointrcnn": {
                "model_name": "keypointrcnn_resnet50_fpn",
                "conf_threshold": 0.25,
                "keypoint_threshold": 0.3,
                "device": "cpu",
                "use_gpu_if_available": True,
                "advanced": {
                    "max_poses": 20,
                    "min_keypoints": 5
                }
            }
        }
        return default_config

    def get_detection_config(self) -> Dict[str, Any]:
        """获取目标检测配置"""
        return self.config.get("detection", self._get_default_config()["detection"])

    def get_pose_estimation_config(self) -> Dict[str, Any]:
        """获取姿态估计配置"""
        return self.config.get("pose_estimation", self._get_default_config()["pose_estimation"])

    def get_yolov7_config(self) -> Dict[str, Any]:
        """获取YOLOv7配置"""
        return self.config.get("yolov7", self._get_default_config()["yolov7"])

    def get_rtmdet_config(self) -> Dict[str, Any]:
        """获取RTMDet配置"""
        return self.config.get("rtmdet", self._get_default_config()["rtmdet"])

    def get_rtmpose_config(self) -> Dict[str, Any]:
        """获取RTMPose配置"""
        return self.config.get("rtmpose", self._get_default_config()["rtmpose"])

    def save_config(self, new_config: Dict[str, Any]) -> bool:
        """保存配置到文件"""
        try:
            # 确保配置目录存在
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(new_config, f, indent=4)

            # 更新当前配置
            self.config = new_config

            logger.info(f"配置已保存到: {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            return False
