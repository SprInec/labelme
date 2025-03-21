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
                "filter_classes": [],
                "nms_threshold": 0.45,  # 非极大值抑制阈值
                "max_detections": 100,  # 最大检测数
                "use_gpu_if_available": True,  # 如果可用则使用GPU
                "advanced": {
                    "pre_nms_top_n": 1000,  # 在NMS之前保留的候选框数量
                    "pre_nms_threshold": 0.5,  # 在NMS之前的分数阈值
                    "max_size": 1333,  # 输入图像的最大尺寸
                    "min_size": 800,  # 输入图像的最小尺寸
                    "score_threshold": 0.05  # 检测分数阈值
                }
            },
            "pose_estimation": {
                "model_name": "keypointrcnn_resnet50_fpn",
                "conf_threshold": 0.5,
                "device": "cpu",
                "use_detection_results": True,  # 使用已有的目标检测结果
                "keypoint_threshold": 0.3,  # 关键点置信度阈值
                "advanced": {
                    "max_poses": 20,  # 最大检测的姿态数量
                    "min_keypoints": 5,  # 最少需要检测到的关键点数
                    "keypoint_score_threshold": 0.2,  # 关键点分数阈值
                    "use_tracking": False,  # 是否使用关键点跟踪
                    "tracking_threshold": 0.5  # 关键点跟踪阈值
                }
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
