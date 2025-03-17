import os
import yaml
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """加载和管理AI模型配置的类"""

    def __init__(self, config_path=None):
        """
        初始化配置加载器

        Args:
            config_path: 配置文件路径，如果为None，则使用默认路径
        """
        if config_path is None:
            # 默认配置文件路径
            self.config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config", "ai_models.yaml"
            )
        else:
            self.config_path = config_path

        self.config = self._load_config()

    def _load_config(self):
        """加载配置文件"""
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"配置文件不存在: {self.config_path}")
                return self._get_default_config()

            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"成功加载配置文件: {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return self._get_default_config()

    def _get_default_config(self):
        """返回默认配置"""
        return {
            "detection": {
                "model_name": "fasterrcnn_resnet50_fpn",
                "conf_threshold": 0.5,
                "device": "cpu",
                "filter_classes": []
            },
            "pose_estimation": {
                "model_name": "keypointrcnn_resnet50_fpn",
                "conf_threshold": 0.5,
                "device": "cpu"
            }
        }

    def get_detection_config(self):
        """获取目标检测配置"""
        return self.config.get("detection", self._get_default_config()["detection"])

    def get_yolov7_config(self):
        """获取YOLOv7配置（已废弃，保留向后兼容）"""
        # 如果存在yolov7配置，则返回
        if "yolov7" in self.config:
            return self.config.get("yolov7")
        # 否则返回detection配置
        return self.get_detection_config()

    def get_pose_estimation_config(self):
        """获取人体姿态估计配置"""
        return self.config.get("pose_estimation", self._get_default_config()["pose_estimation"])

    def save_config(self, config=None):
        """
        保存配置到文件

        Args:
            config: 要保存的配置，如果为None，则保存当前配置
        """
        if config is None:
            config = self.config

        try:
            # 确保配置目录存在
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False,
                          allow_unicode=True)
            logger.info(f"配置已保存到: {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            return False
