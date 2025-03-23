import json
import os
import time
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import cv2
from loguru import logger
from PyQt5 import QtCore

# 导入配置加载器
from labelme._automation.config_loader import ConfigLoader
# 导入模型下载器
from labelme._automation.model_downloader import download_rtmpose_model

# 尝试导入PyTorch依赖，如果不可用则提供错误信息
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("姿态估计依赖未安装，请安装torch")

# RTMPose模型配置文件
RTMPOSE_MODEL_CONFIGS = {
    "rtmpose-t": 'labelme/_automation/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_coco-256x192.py',
    "rtmpose-s": 'labelme/_automation/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py',
    "rtmpose-m": 'labelme/_automation/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py',
    "rtmpose-l": 'labelme/_automation/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_coco-256x192.py'
}

# RTMPose模型权重文件
RTMPOSE_MODEL_CHECKPOINTS = {
    "rtmpose-t": 'labelme/_automation/mmpose/checkpoints/rtmpose-t_simcc-aic-coco_pt-aic-coco_420e-256x192-e0c9327b_20230127.pth',
    "rtmpose-s": 'labelme/_automation/mmpose/checkpoints/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230127.pth',
    "rtmpose-m": 'labelme/_automation/mmpose/checkpoints/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth',
    "rtmpose-l": 'labelme/_automation/mmpose/checkpoints/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-1f9a0168_20230126.pth'
}

# COCO数据集的关键点定义
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# COCO数据集的骨架连接定义
COCO_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6)
]

# 姿态关键点颜色定义
KEYPOINT_COLORS = [
    (0, 255, 255),   # 0: 鼻子
    (0, 191, 255),   # 1: 左眼
    (0, 255, 102),   # 2: 右眼
    (0, 77, 255),    # 3: 左耳
    (0, 255, 0),     # 4: 右耳
    (77, 255, 255),  # 5: 左肩
    (77, 255, 204),  # 6: 右肩
    (204, 77, 255),  # 7: 左肘
    (204, 204, 77),  # 8: 右肘
    (255, 191, 77),  # 9: 左腕
    (255, 77, 36),   # 10: 右腕
    (255, 77, 255),  # 11: 左髋
    (255, 77, 204),  # 12: 右髋
    (191, 255, 77),  # 13: 左膝
    (77, 255, 77),   # 14: 右膝
    (77, 255, 255),  # 15: 左踝
    (77, 77, 255),   # 16: 右踝
]


class PoseEstimator:
    """人体姿态估计器"""

    def __init__(self, model_name: str = None, device: str = None,
                 conf_threshold: float = None, keypoint_threshold: float = None,
                 advanced_params: dict = None, draw_skeleton: bool = None):
        """
        初始化姿态估计器

        Args:
            model_name: 模型名称，可选值: 
                - rtmpose_tiny, rtmpose_s, rtmpose_m, rtmpose_l (RTMPose模型)
                - hrnet_w32, hrnet_w32_udp, hrnet_w48, hrnet_w48_udp (HRNet模型)
                - yolov7_w6_pose (YOLOv7-Pose模型)
                - keypointrcnn_resnet50_fpn (KeypointRCNN模型)
            device: 运行设备 ('cpu' 或 'cuda')
            conf_threshold: 置信度阈值
            keypoint_threshold: 关键点置信度阈值
            advanced_params: 高级参数字典
            draw_skeleton: 是否绘制骨骼连接线
        """
        if not HAS_TORCH:
            raise ImportError("姿态估计依赖未安装，请安装torch")

        # 加载配置
        config_loader = ConfigLoader()
        pose_config = config_loader.get_pose_estimation_config()

        # 使用配置值或默认值
        self.model_name = model_name or pose_config.get(
            "model_name", "keypointrcnn_resnet50_fpn")
        self.conf_threshold = conf_threshold or pose_config.get(
            "conf_threshold", 0.5)
        self.device = device or pose_config.get("device", "cpu")
        self.keypoint_threshold = keypoint_threshold or pose_config.get(
            "keypoint_threshold", 0.2)
        self.advanced_params = advanced_params or pose_config.get(
            "advanced", {})

        # 设置是否绘制骨骼的参数
        self.draw_skeleton = draw_skeleton if draw_skeleton is not None else pose_config.get(
            "draw_skeleton", True)

        # 确定模型类型
        self.model_type = None
        if self.model_name:
            if self.model_name.startswith("rtmpose"):
                self.model_type = "rtmpose"
            elif self.model_name.startswith("hrnet"):
                self.model_type = "hrnet"
            elif self.model_name.startswith("yolov7"):
                self.model_type = "yolov7_pose"
            elif self.model_name.startswith("keypointrcnn"):
                self.model_type = "keypointrcnn"
            else:
                # 默认为KeypointRCNN
                self.model_type = "keypointrcnn"

        logger.info(f"使用设备: {self.device}")

        # 初始化属性
        self.model = None
        self.rtmpose_model = None
        self.det_model = None
        self.keypointrcnn_model = None
        self.is_rtmpose = False
        self.is_hrnet = False
        self.is_keypointrcnn = False
        self.is_yolov7_pose = False
        self.imgsz = 640
        self.stride = 32
        self.visualizer = None

        # 加载模型
        self._load_model()

    def _load_model(self):
        """加载姿态检测模型"""
        try:
            # 检查CUDA可用性
            if torch.cuda.is_available() and self.device == 'cuda':
                self.device = 'cuda'
            else:
                self.device = 'cpu'

            # 根据模型类型进行加载
            if self.model_type == 'rtmpose':
                if self._init_rtmpose():
                    self.is_rtmpose = True
                else:
                    logger.warning("加载RTMPose模型失败，尝试使用KeypointRCNN")
                    self.model_type = 'keypointrcnn'
                    self._load_keypointrcnn_model()
                    self.is_keypointrcnn = True

            elif self.model_type == 'hrnet':
                self.model = self._load_hrnet_model()
                self.is_hrnet = True

            elif self.model_type == 'yolov7_pose':
                try:
                    from labelme._automation.yolov7.models.experimental import attempt_load
                    from labelme._automation.yolov7.utils.general import check_img_size, non_max_suppression_kpt
                    from labelme._automation.yolov7.utils.torch_utils import select_device
                    self.model = self._load_yolov7_pose_model()
                    self.is_yolov7_pose = True
                except ImportError:
                    logger.warning("YOLOv7依赖未安装，自动切换到KeypointRCNN模型")
                    self.model_type = 'keypointrcnn'
                    self._load_keypointrcnn_model()
                    self.is_keypointrcnn = True

            elif self.model_type == 'keypointrcnn':
                self._load_keypointrcnn_model()
                self.is_keypointrcnn = True

            else:
                # 默认使用KeypointRCNN
                logger.warning(f"未知模型类型: {self.model_type}，使用KeypointRCNN")
                self.model_type = 'keypointrcnn'
                self._load_keypointrcnn_model()
                self.is_keypointrcnn = True

        except Exception as e:
            logger.error(f"加载模型发生错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 如果加载失败，使用KeypointRCNN作为备用
            logger.warning("加载模型失败，尝试使用KeypointRCNN")
            try:
                self.model_type = 'keypointrcnn'
                self._load_keypointrcnn_model()
                self.is_keypointrcnn = True
            except Exception as e2:
                logger.error(f"加载KeypointRCNN备用模型也失败: {e2}")
                raise ValueError("无法加载任何姿态估计模型")

    def _load_hrnet_model(self):
        """加载HRNet姿态估计模型"""
        try:
            # 尝试导入MMPose相关依赖
            try:
                import torch
                import mmpose
                from mmpose.apis import inference_topdown, init_model
                from mmpose.evaluation.functional import nms
                from mmpose.structures import merge_data_samples
                from mmpose.registry import VISUALIZERS
                from mmengine.registry import init_default_scope
                # 导入检测模型所需库
                try:
                    from mmdet.apis import init_detector, inference_detector
                    HAS_MMDET = True
                except ImportError:
                    HAS_MMDET = False
                    logger.warning(
                        "MMDet未安装，无法使用RTMDet进行人体检测。请安装mmdet：pip install mmdet")

                HAS_MMPOSE = True
            except ImportError:
                HAS_MMPOSE = False
                logger.warning(
                    "MMPose未安装，无法使用HRNet模型。请安装mmpose：pip install openmim && mim install mmpose>=1.2.0")
                raise ImportError(
                    "MMPose未安装，无法使用HRNet模型。请安装mmpose：pip install openmim && mim install mmpose>=1.2.0")

            # 初始化默认作用域为mmpose
            init_default_scope('mmpose')

            # HRNet模型配置和权重映射
            hrnet_configs = {
                "hrnet_w32": {
                    "config": "td-hm_hrnet-w32_8xb64-210e_coco-256x192.py",
                    "checkpoint": "hrnet_w32_coco_256x192-c78dce93_20200708.pth"
                },
                "hrnet_w32_udp": {
                    "config": "td-hm_hrnet-w32_udp-8xb64-210e_coco-256x192.py",
                    "checkpoint": "hrnet_w32_coco_256x192_udp-aba0be42_20220624.pth"
                },
                "hrnet_w48": {
                    "config": "td-hm_hrnet-w48_8xb32-210e_coco-256x192.py",
                    "checkpoint": "hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"
                },
                "hrnet_w48_udp": {
                    "config": "td-hm_hrnet-w48_udp-8xb32-210e_coco-256x192.py",
                    "checkpoint": "hrnet_w48_coco_256x192_udp-7f9d1e8a_20220624.pth"
                }
            }

            # RTMDet人体检测配置
            rtmdet_config = {
                "config": "rtmdet_m_8xb32-300e_coco.py",
                "checkpoint": "rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
            }

            if self.model_name not in hrnet_configs:
                logger.warning(f"未知的HRNet模型: {self.model_name}，使用hrnet_w32")
                self.model_name = "hrnet_w32"

            # 获取模型配置和权重
            model_config = hrnet_configs[self.model_name]

            # 检查是否有本地配置文件，否则使用MMPose默认配置
            config_file = model_config["config"]

            # 首先检查当前目录是否有配置文件
            if not os.path.exists(config_file):
                # 检查mmpose目录下是否有配置文件
                mmpose_config_path = os.path.join(os.path.dirname(
                    __file__), "mmpose", "configs", "body_2d_keypoint", "topdown_heatmap", "coco", config_file)
                if os.path.exists(mmpose_config_path):
                    logger.info(f"使用本地mmpose目录中的配置文件: {mmpose_config_path}")
                    config_file = mmpose_config_path
                else:
                    # 使用MMPose默认配置
                    logger.info(
                        f"使用MMPose默认配置: body_2d_keypoint/topdown_heatmap/coco/{config_file}")
                    config_file = f"body_2d_keypoint/topdown_heatmap/coco/{config_file}"

            # 检查是否有本地RTMDet配置文件，否则使用MMDet默认配置
            rtmdet_config_file = rtmdet_config["config"]
            if not os.path.exists(rtmdet_config_file) and HAS_MMDET:
                # 使用MMDet默认配置
                logger.info(f"使用MMDet默认配置: {rtmdet_config_file}")
                rtmdet_config_file = f"rtmdet/{rtmdet_config_file}"

            # 初始化RTMDet检测模型(如果可用)
            self.detector_model = None
            if HAS_MMDET:
                try:
                    # 尝试从网络下载RTMDet权重文件
                    try:
                        from labelme._automation.model_downloader import download_mmdet_model
                        logger.info(f"尝试下载RTMDet-m检测模型")
                        checkpoint_path = download_mmdet_model("rtmdet_m")
                        if checkpoint_path:
                            logger.info(f"RTMDet-m模型下载成功: {checkpoint_path}")
                            rtmdet_checkpoint_file = checkpoint_path
                        else:
                            # 如果下载失败，使用MMDet默认权重
                            logger.warning(f"RTMDet-m模型下载失败, 使用MMDet默认权重")
                            rtmdet_checkpoint_file = None
                    except Exception as e:
                        logger.warning(f"下载RTMDet-m模型失败: {e}, 使用MMDet默认权重")
                        rtmdet_checkpoint_file = None

                    # 初始化RTMDet模型
                    from mmdet.apis import init_detector
                    init_default_scope('mmdet')
                    self.detector_model = init_detector(
                        rtmdet_config_file,
                        rtmdet_checkpoint_file,
                        device=self.device
                    )
                    logger.info("RTMDet-m人体检测模型加载成功")

                    # 重新设置默认作用域为mmpose
                    init_default_scope('mmpose')
                except Exception as e:
                    logger.warning(f"加载RTMDet-m模型失败: {e}，将仅使用HRNet模型")
                    self.detector_model = None

            # 检查是否有本地权重文件，否则使用MMPose默认权重
            checkpoint_file = model_config["checkpoint"]

            # 首先在_automation/mmpose/checkpoints目录查找
            try:
                from labelme._automation.model_downloader import get_model_dir
                local_checkpoint_dir = get_model_dir("mmpose")
                local_checkpoint_path = os.path.join(
                    local_checkpoint_dir, f"{self.model_name}.pth")
                if os.path.exists(local_checkpoint_path):
                    logger.info(
                        f"在本地目录找到HRNet模型权重文件: {local_checkpoint_path}")
                    checkpoint_file = local_checkpoint_path
                else:
                    # 检查原有的缓存目录
                    checkpoint_dir = os.path.join(os.path.expanduser(
                        "~"), ".cache", "torch", "hub", "checkpoints")
                    checkpoint_path = os.path.join(
                        checkpoint_dir, checkpoint_file)

                    if os.path.exists(checkpoint_path):
                        logger.info(f"在缓存目录找到HRNet模型权重文件: {checkpoint_path}")
                        checkpoint_file = checkpoint_path
                    else:
                        # 尝试从网络下载权重文件
                        logger.info(f"模型权重不存在，尝试从网络下载: {self.model_name}")
                        try:
                            # 使用已导入的download_rtmpose_model函数
                            # 即使这是HRNet模型，我们仍然使用download_rtmpose_model函数，
                            # 因为model_downloader.py中这个函数也支持下载HRNet模型
                            checkpoint_path = download_rtmpose_model(
                                self.model_name)
                            if checkpoint_path:
                                logger.info(f"模型下载成功: {checkpoint_path}")
                                checkpoint_file = checkpoint_path
                            else:
                                # 如果下载失败，使用MMPose默认权重
                                logger.warning(f"模型下载失败，使用MMPose默认权重")
                                checkpoint_file = None
                        except Exception as e:
                            logger.warning(f"下载模型失败: {e}，使用MMPose默认权重")
                            checkpoint_file = None
            except Exception as e:
                logger.warning(f"查找模型路径时出错: {e}，尝试使用默认路径")
                checkpoint_dir = os.path.join(os.path.expanduser(
                    "~"), ".cache", "torch", "hub", "checkpoints")
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

                if os.path.exists(checkpoint_path):
                    checkpoint_file = checkpoint_path
                else:
                    logger.warning(f"下载模型失败: {e}，使用MMPose默认权重")
                    checkpoint_file = None

            # 初始化模型
            model = init_model(
                config_file,
                checkpoint_file,
                device=self.device,
                cfg_options={'model': {'test_cfg': {'output_heatmaps': True}}}
            )

            # 存储可视化器
            self.visualizer = VISUALIZERS.build(model.cfg.visualizer)
            self.visualizer.set_dataset_meta(model.dataset_meta)

            logger.info(f"HRNet模型 {self.model_name} 加载成功")
            return model
        except Exception as e:
            logger.error(f"加载HRNet模型失败: {e}")
            raise

    def _load_keypointrcnn_model(self):
        """加载KeypointRCNN模型"""
        try:
            import torchvision
            logger.info("加载KeypointRCNN模型...")

            # 加载模型
            model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
                pretrained=True,
                num_keypoints=17,
                min_size=256,
                max_size=512
            )

            # 将模型移动到正确的设备并设置为评估模式
            model = model.to(self.device)
            model.eval()

            # 保存模型实例
            self.keypointrcnn_model = model
            self.model = model
            logger.info("KeypointRCNN模型加载完成")
            return model
        except Exception as e:
            logger.error(f"加载KeypointRCNN模型失败: {e}")
            raise

    def _load_yolov7_pose_model(self):
        """加载YOLOv7姿态估计模型"""
        try:
            import sys
            import os

            # 添加YOLOv7路径到系统路径
            yolov7_dir = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "yolov7")
            if yolov7_dir not in sys.path:
                sys.path.append(yolov7_dir)

            print("YOLOv7目录:", yolov7_dir)
            print("系统路径:", sys.path)

            # 导入YOLOv7依赖
            try:
                from labelme._automation.yolov7.models.experimental import attempt_load
                from labelme._automation.yolov7.utils.torch_utils import select_device
                from labelme._automation.yolov7.utils.general import check_img_size
                print("成功导入YOLOv7依赖")
            except ImportError as e:
                print(f"导入YOLOv7依赖失败: {e}")
                raise ImportError(f"无法导入YOLOv7依赖: {e}")

            # 设置设备
            self.device = select_device(
                self.device) if self.device else select_device('')
            print(f"使用设备: {self.device}")

            # 设置权重文件路径
            weights_path = self.params.get("weights_path")
            if not weights_path:
                # 尝试从_automation/yolov7/checkpoints目录获取模型
                try:
                    from labelme._automation.model_downloader import get_model_dir
                    yolo_checkpoints = get_model_dir("yolov7")
                    custom_weights_path = os.path.join(
                        yolo_checkpoints, "yolov7-w6-pose.pt")
                    if os.path.exists(custom_weights_path):
                        weights_path = custom_weights_path
                        logger.info(
                            f"在_automation/yolov7/checkpoints目录中找到YOLOv7姿态估计模型: {weights_path}")
                    else:
                        # 使用默认路径
                        weights_path = os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "weights",
                            "yolov7-w6-pose.pt"
                        )
                        logger.info(
                            f"未找到自定义路径的YOLOv7姿态估计模型，使用默认路径: {weights_path}")
                except Exception as e:
                    logger.warning(f"尝试获取自定义模型路径时出错: {e}，使用默认路径")
                    # 使用默认路径
                    weights_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "weights",
                        "yolov7-w6-pose.pt"
                    )

            print(f"权重文件路径: {weights_path}")

            # 检查权重文件是否存在
            if not os.path.exists(weights_path):
                weights_dir = os.path.dirname(weights_path)
                os.makedirs(weights_dir, exist_ok=True)
                error_msg = (
                    f"YOLOv7-pose模型权重文件不存在: {weights_path}\n"
                    f"请下载yolov7-w6-pose.pt权重文件并放置到以下位置:\n"
                    f"{weights_path}\n"
                    f"可以从https://github.com/WongKinYiu/yolov7/releases下载"
                )
                print(error_msg)
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            print(f"权重文件已找到: {weights_path}")

            # 加载模型
            print("尝试加载YOLOv7模型...")
            self.model = attempt_load(weights_path, map_location=self.device)
            print("YOLOv7模型加载成功")

            # 设置模型为评估模式
            self.model.eval()

            # 获取其他参数
            self.half = self.params.get(
                "half", False) and self.device.type != 'cpu'
            if self.half:
                self.model.half()

            # 获取步长
            self.stride = int(self.model.stride.max()) if hasattr(
                self.model, 'stride') else 32

            # 设置图像大小
            self.img_size = self.params.get("img_size", 640)
            if isinstance(self.img_size, (list, tuple)):
                self.img_size = self.img_size[0]
            self.img_size = check_img_size(self.img_size, s=self.stride)

            logger.info(f"YOLOv7姿态估计模型加载成功: {self.model_name}")
            return True

        except Exception as e:
            logger.error(f"加载YOLOv7姿态估计模型失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise ImportError(f"加载YOLOv7姿态估计模型失败，请确保已安装正确的YOLOv7依赖: {e}")

    def _load_rtmpose_model(self):
        """加载RTMPose模型"""
        try:
            # 尝试导入MMPose相关依赖
            try:
                import torch
                import mmpose
                from mmpose.apis import inference_topdown, init_model
                from mmpose.evaluation.functional import nms
                from mmpose.structures import merge_data_samples
                from mmpose.registry import VISUALIZERS
                from mmengine.registry import init_default_scope
                # 导入检测模型所需库
                try:
                    from mmdet.apis import init_detector, inference_detector
                    HAS_MMDET = True
                except ImportError:
                    HAS_MMDET = False
                    logger.warning(
                        "MMDet未安装，无法使用RTMDet进行人体检测。请安装mmdet：pip install mmdet")

                HAS_MMPOSE = True
            except ImportError:
                HAS_MMPOSE = False
                logger.warning(
                    "MMPose未安装，无法使用RTMPose模型。请安装mmpose：pip install openmim && mim install mmpose>=1.2.0")
                raise ImportError(
                    "MMPose未安装，无法使用RTMPose模型。请安装mmpose：pip install openmim && mim install mmpose>=1.2.0")

            # 初始化默认作用域为mmpose
            init_default_scope('mmpose')

            # RTMPose模型配置和权重映射
            rtmpose_configs = {
                "rtmpose_tiny": {
                    "config": "rtmpose-t_8xb256-420e_coco-256x192.py",
                    "checkpoint": "rtmpose-t_simcc-aic-coco_pt-aic-coco_420e-256x192-e0c9327b_20230127.pth"
                },
                "rtmpose_s": {
                    "config": "rtmpose-s_8xb256-420e_coco-256x192.py",
                    "checkpoint": "rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230127.pth"
                },
                "rtmpose_m": {
                    "config": "rtmpose-m_8xb256-420e_coco-256x192.py",
                    "checkpoint": "rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth"
                },
                "rtmpose_l": {
                    "config": "rtmpose-l_8xb256-420e_coco-256x192.py",
                    "checkpoint": "rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-1f9a0168_20230126.pth"
                }
            }

            # RTMDet人体检测配置
            rtmdet_config = {
                "config": "rtmdet_m_8xb32-300e_coco.py",
                "checkpoint": "rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
            }

            if self.model_name not in rtmpose_configs:
                logger.warning(f"未知的RTMPose模型: {self.model_name}，使用rtmpose_s")
                self.model_name = "rtmpose_s"

            # 获取模型配置和权重
            model_config = rtmpose_configs[self.model_name]

            # 检查是否有本地配置文件，否则使用MMPose默认配置
            config_file = model_config["config"]

            # 首先检查当前目录是否有配置文件
            if not os.path.exists(config_file):
                # 检查mmpose目录下是否有配置文件
                mmpose_config_path = os.path.join(os.path.dirname(
                    __file__), "mmpose", "configs", "body_2d_keypoint", "rtmpose", "coco", config_file)
                if os.path.exists(mmpose_config_path):
                    logger.info(f"使用本地mmpose目录中的配置文件: {mmpose_config_path}")
                    config_file = mmpose_config_path
                else:
                    # 使用MMPose默认配置
                    logger.info(
                        f"使用MMPose默认配置: body_2d_keypoint/rtmpose/{config_file}")
                    config_file = f"body_2d_keypoint/rtmpose/{config_file}"

            # 检查是否有本地RTMDet配置文件，否则使用MMDet默认配置
            rtmdet_config_file = rtmdet_config["config"]
            if not os.path.exists(rtmdet_config_file) and HAS_MMDET:
                # 使用MMDet默认配置
                logger.info(f"使用MMDet默认配置: {rtmdet_config_file}")
                rtmdet_config_file = f"rtmdet/{rtmdet_config_file}"

            # 初始化RTMDet检测模型(如果可用)
            self.detector_model = None
            if HAS_MMDET:
                try:
                    # 尝试从网络下载RTMDet权重文件
                    try:
                        from labelme._automation.model_downloader import download_mmdet_model
                        logger.info(f"尝试下载RTMDet-m检测模型")
                        checkpoint_path = download_mmdet_model("rtmdet_m")
                        if checkpoint_path:
                            logger.info(f"RTMDet-m模型下载成功: {checkpoint_path}")
                            rtmdet_checkpoint_file = checkpoint_path
                        else:
                            # 如果下载失败，使用MMDet默认权重
                            logger.warning(f"RTMDet-m模型下载失败, 使用MMDet默认权重")
                            rtmdet_checkpoint_file = None
                    except Exception as e:
                        logger.warning(f"下载RTMDet-m模型失败: {e}, 使用MMDet默认权重")
                        rtmdet_checkpoint_file = None

                    # 初始化RTMDet模型
                    from mmdet.apis import init_detector
                    init_default_scope('mmdet')
                    self.detector_model = init_detector(
                        rtmdet_config_file,
                        rtmdet_checkpoint_file,
                        device=self.device
                    )
                    logger.info("RTMDet-m人体检测模型加载成功")

                    # 重新设置默认作用域为mmpose
                    init_default_scope('mmpose')
                except Exception as e:
                    logger.warning(f"加载RTMDet-m模型失败: {e}，将仅使用RTMPose模型")
                    self.detector_model = None

            # 检查是否有本地权重文件，否则使用MMPose默认权重
            checkpoint_file = model_config["checkpoint"]

            # 首先在_automation/mmpose/checkpoints目录查找
            try:
                from labelme._automation.model_downloader import get_model_dir
                local_checkpoint_dir = get_model_dir("mmpose")
                local_checkpoint_path = os.path.join(
                    local_checkpoint_dir, f"{self.model_name}.pth")
                if os.path.exists(local_checkpoint_path):
                    logger.info(
                        f"在本地目录找到RTMPose模型权重文件: {local_checkpoint_path}")
                    checkpoint_file = local_checkpoint_path
                else:
                    # 检查原有的缓存目录
                    checkpoint_dir = os.path.join(os.path.expanduser(
                        "~"), ".cache", "torch", "hub", "checkpoints")

                    # 检查checkpoint_file是否有效，避免拼接None
                    if checkpoint_file:
                        checkpoint_path = os.path.join(
                            checkpoint_dir, os.path.basename(checkpoint_file))

                        if os.path.exists(checkpoint_path):
                            logger.info(
                                f"在缓存目录找到RTMPose模型权重文件: {checkpoint_path}")
                            checkpoint_file = checkpoint_path
                        else:
                            # 尝试从网络下载权重文件
                            try:
                                from labelme._automation.model_downloader import download_rtmpose_model
                                logger.info(
                                    f"模型权重不存在，尝试从网络下载: {self.model_name}")
                                checkpoint_path = download_rtmpose_model(
                                    self.model_name)
                                if checkpoint_path:
                                    logger.info(f"模型下载成功: {checkpoint_path}")
                                    checkpoint_file = checkpoint_path
                                else:
                                    # 如果下载失败，使用MMPose默认权重
                                    logger.warning(f"模型下载失败，使用MMPose默认权重")
                                    checkpoint_file = None
                            except Exception as e:
                                logger.warning(f"下载模型失败: {e}，使用MMPose默认权重")
                                checkpoint_file = None
                    else:
                        # 如果checkpoint_file为None，直接尝试下载
                        try:
                            from labelme._automation.model_downloader import download_rtmpose_model
                            logger.info(
                                f"没有找到预设权重文件，尝试从网络下载: {self.model_name}")
                            checkpoint_path = download_rtmpose_model(
                                self.model_name)
                            if checkpoint_path:
                                logger.info(f"模型下载成功: {checkpoint_path}")
                                checkpoint_file = checkpoint_path
                            else:
                                logger.warning(f"模型下载失败，使用MMPose默认权重")
                                checkpoint_file = None
                        except Exception as e:
                            logger.warning(f"下载模型失败: {e}，使用MMPose默认权重")
                            checkpoint_file = None
            except Exception as e:
                logger.warning(f"查找模型路径时出错: {e}，尝试使用默认路径")
                checkpoint_dir = os.path.join(os.path.expanduser(
                    "~"), ".cache", "torch", "hub", "checkpoints")
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

                if os.path.exists(checkpoint_path):
                    checkpoint_file = checkpoint_path
                else:
                    # 尝试从网络下载权重文件
                    try:
                        from labelme._automation.model_downloader import download_rtmpose_model
                        logger.info(f"模型权重不存在，尝试从网络下载: {self.model_name}")
                        checkpoint_path = download_rtmpose_model(
                            self.model_name)
                        if checkpoint_path:
                            logger.info(f"模型下载成功: {checkpoint_path}")
                            checkpoint_file = checkpoint_path
                        else:
                            # 如果下载失败，使用MMPose默认权重
                            logger.warning(f"模型下载失败，使用MMPose默认权重")
                            checkpoint_file = None
                    except Exception as e:
                        logger.warning(f"下载模型失败: {e}，使用MMPose默认权重")
                        checkpoint_file = None

            # 初始化模型
            try:
                # 确保config_file和checkpoint_file是字符串
                if not isinstance(config_file, str):
                    logger.warning(f"配置文件类型错误: {type(config_file)}，将转换为字符串")
                    config_file = str(config_file)

                if checkpoint_file is not None and not isinstance(checkpoint_file, str):
                    logger.warning(
                        f"检查点文件类型错误: {type(checkpoint_file)}，将转换为字符串")
                    checkpoint_file = str(checkpoint_file)

                # 初始化模型
                model = init_model(
                    config_file,
                    checkpoint_file,
                    device=self.device
                )

                # 验证模型对象
                if model is None:
                    logger.error("init_model返回None，初始化失败")
                    return False

                if isinstance(model, (np.ndarray, list, tuple)):
                    logger.error(f"模型类型错误: {type(model)}，无法用于姿态估计")
                    return False

                # 测试是否可以访问模型属性
                try:
                    if hasattr(model, 'cfg'):
                        logger.debug(f"模型配置: {type(model.cfg)}")
                    else:
                        logger.warning("模型没有cfg属性")
                except Exception as attr_error:
                    logger.warning(f"无法访问模型属性: {attr_error}")

                # 保存模型实例
                self.rtmpose_model = model
                self.model = model

                # 初始化可视化器
                try:
                    if hasattr(model, 'cfg') and hasattr(model, 'dataset_meta'):
                        from mmpose.visualization import VISUALIZERS
                        self.visualizer = VISUALIZERS.build(
                            model.cfg.visualizer)
                        self.visualizer.set_dataset_meta(model.dataset_meta)
                except Exception as vis_error:
                    logger.warning(f"初始化可视化器失败: {vis_error}")

                logger.info(f"RTMPose模型初始化成功: {self.model_name}")
                logger.info(f"配置文件: {config_file}")
                logger.info(f"权重文件: {checkpoint_file}")
                self.is_rtmpose = True
                return True
            except Exception as e:
                logger.error(f"初始化模型失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.rtmpose_model = None
                self.is_rtmpose = False
                return False
        except Exception as e:
            logger.error(f"加载RTMPose模型失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.is_rtmpose = False
            self.rtmpose_model = None
            return False

    def _detect_rtmpose(self, image: np.ndarray) -> Tuple[List[List[List[float]]], List[float]]:
        """
        使用RTMPose检测整张图像中的姿态

        Args:
            image: 输入图像

        Returns:
            keypoints: 关键点列表 [N, K, 3] - (x, y, conf)
            scores: 人体检测的置信度列表 [N]
        """
        try:
            # 导入必要的库
            try:
                from mmpose.apis import inference_topdown
                from mmdet.apis import inference_detector
            except ImportError as e:
                logger.error(f"导入mmpose或mmdet库失败: {e}")
                return [], []  # 不再使用备用方法

            # 确保图像是RGB格式
            if image.shape[-1] != 3:
                logger.warning(f"输入图像通道数为{image.shape[-1]}，转换为RGB")
                if len(image.shape) == 2:  # 灰度图像
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                else:
                    # 尝试转换为RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 记录原始图像尺寸
            img_height, img_width = image.shape[:2]
            logger.debug(f"原始图像尺寸: {img_width}x{img_height}")

            # 首先使用MMDet进行目标检测
            if not hasattr(self, 'det_model') or self.det_model is None:
                logger.warning("未找到检测模型，返回空结果")
                return [], []  # 不再使用备用方法

            try:
                det_results = inference_detector(self.det_model, image)
                logger.debug(f"成功调用inference_detector")

                # 提取人体检测结果
                pred_instances = det_results.pred_instances

                # 获取边界框、置信度和类别
                if hasattr(pred_instances, 'bboxes'):
                    bboxes = pred_instances.bboxes.cpu().numpy()
                    scores = pred_instances.scores.cpu().numpy()
                    labels = pred_instances.labels.cpu().numpy()

                    # 只保留人体检测结果（类别0通常是人）
                    human_indices = np.where(labels == 0)[0]
                    human_bboxes = bboxes[human_indices]
                    human_scores = scores[human_indices]

                    logger.debug(f"检测到 {len(human_bboxes)} 个人体")

                    # 过滤置信度低的检测结果
                    valid_indices = np.where(
                        human_scores >= self.conf_threshold)[0]
                    if len(valid_indices) == 0:
                        logger.warning(
                            f"未检测到置信度大于{self.conf_threshold}的人体，返回空结果")
                        return [], []  # 不再使用备用方法

                    valid_bboxes = human_bboxes[valid_indices]
                    valid_scores = human_scores[valid_indices]
                else:
                    logger.warning("检测结果中未找到边界框信息，返回空结果")
                    return [], []  # 不再使用备用方法
            except Exception as e:
                logger.error(f"目标检测失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return [], []  # 不再使用备用方法

            # 使用检测到的边界框进行姿态估计
            # 将boxes处理为list以提高性能
            valid_bboxes_list = valid_bboxes.tolist()

            # 直接调用从边界框检测的方法
            keypoints, scores = self._detect_rtmpose_from_boxes(
                image, valid_bboxes_list)

            # 返回结果
            if len(keypoints) > 0:
                logger.info(f"RTMPose检测到 {len(keypoints)} 个姿态")
                return keypoints, scores
            else:
                logger.warning("RTMPose未检测到有效姿态，返回空结果")
                return [], []  # 不再使用备用方法
        except Exception as e:
            logger.error(f"RTMPose检测失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [], []  # 不再使用备用方法

    def _detect_yolov7_pose(self, image: np.ndarray) -> Tuple[List[List[List[float]]], List[float]]:
        """使用YOLOv7姿态估计模型进行检测"""
        # 转换图像格式
        if image.shape[2] == 4:  # 如果有alpha通道
            image = image[:, :, :3]

        # 调整图像尺寸
        orig_shape = image.shape
        img = self._letterbox(image, self.imgsz, stride=self.stride)[0]

        # 转换为PyTorch张量
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        # 推理
        with torch.no_grad():
            output, _ = self.model(img)
            # NMS
            output = self.non_max_suppression_kpt(
                output, self.conf_threshold, self.keypoint_threshold, nc=1)

        # 处理输出
        keypoints_list = []
        scores_list = []

        for i, det in enumerate(output):
            if len(det):
                # 重新调整到原始图像尺寸
                scale = torch.tensor(
                    [orig_shape[1], orig_shape[0], orig_shape[1], orig_shape[0]]).to(self.device)
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6].detach().cpu().numpy())):
                    # 获取关键点
                    kpts = det[j, 6:].detach().cpu().numpy()
                    kpts = kpts.reshape(-1, 3)  # 17, 3

                    # 重新调整关键点到原始图像尺寸
                    image_width = orig_shape[1]
                    image_height = orig_shape[0]
                    r = min(self.imgsz / image_width,
                            self.imgsz / image_height)
                    pad_w = (self.imgsz - image_width * r) / 2
                    pad_h = (self.imgsz - image_height * r) / 2

                    # 调整关键点坐标
                    for k in range(len(kpts)):
                        kpts[k][0] = (kpts[k][0] - pad_w) / r
                        kpts[k][1] = (kpts[k][1] - pad_h) / r

                    keypoints_list.append(kpts.tolist())
                    scores_list.append(float(conf))

        return keypoints_list, scores_list

    def _detect_keypointrcnn(self, image: np.ndarray) -> Tuple[List[List[List[float]]], List[float]]:
        """使用KeypointRCNN进行检测"""
        try:
            # 确保模型已初始化
            if not hasattr(self, 'keypointrcnn_model') or self.keypointrcnn_model is None:
                # 不再使用备用模型初始化
                # self._init_keypointrcnn_backup()
                logger.error("KeypointRCNN模型未初始化")
                return [], []

            # 确保图像是RGB格式
            if image.shape[2] == 4:  # 如果有alpha通道
                image = image[:, :, :3]

            # 转换为PyTorch张量
            img_tensor = torch.from_numpy(image.transpose(
                2, 0, 1)).float().div(255.0).unsqueeze(0)

            # 将张量移到设备上
            img_tensor = img_tensor.to(self.device)

            # 使用模型预测
            with torch.no_grad():
                predictions = self.keypointrcnn_model(img_tensor)

            # 提取关键点和分数
            keypoints_list = []
            scores_list = []

            # 处理检测结果
            if len(predictions) > 0 and 'keypoints' in predictions[0]:
                pred = predictions[0]

                # 获取人体检测框和分数
                boxes = pred['boxes'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()
                keypoints = pred['keypoints'].cpu().numpy()

                # 过滤低置信度的检测结果
                mask = scores >= self.conf_threshold
                boxes = boxes[mask]
                scores = scores[mask]
                keypoints = keypoints[mask]

                # 调整关键点坐标到原图位置
                for i, person_kpts in enumerate(keypoints):
                    # 添加处理单个人关键点的代码
                    keypoints_list.append(person_kpts.tolist())
                    scores_list.append(float(scores[i]))

            return keypoints_list, scores_list

        except Exception as e:
            logger.error(f"KeypointRCNN检测失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [], []

    def detect_poses_from_boxes(self, image: np.ndarray, boxes: List[List[float]]) -> Tuple[List[List[List[float]]], List[float]]:
        """从已有的人体框中检测姿态关键点

        Args:
            image: 输入图像
            boxes: 边界框列表，格式为[[x1, y1, x2, y2], ...]

        Returns:
            keypoints: 关键点列表，格式为[[[x, y, v], ...], ...]
            scores: 人体检测的置信度列表
        """
        try:
            keypoints, scores = [], []

            if len(boxes) == 0:
                logger.warning("未检测到人体边界框，跳过姿态检测")
                return [], []

            if self.model_type == 'keypointrcnn':
                keypoints, scores = self._detect_keypointrcnn_from_boxes(
                    image, boxes)
            elif self.model_type == 'rtmpose':
                keypoints, scores = self._detect_rtmpose_from_boxes(
                    image, boxes, self.keypoint_threshold)
            elif self.model_type == 'hrnet':
                keypoints, scores = self._detect_hrnet_from_boxes(
                    image, boxes, self.keypoint_threshold)
            else:
                logger.error(f"不支持的姿态检测模型类型: {self.model_type}")
                return [], []

            # 仅返回原始模型的检测结果，不再使用备用模型
            if len(keypoints) > 0:
                logger.info(f"成功检测到 {len(keypoints)} 个人体姿态")
                return keypoints, scores
            else:
                logger.warning("没有检测到有效的人体姿态")
                return [], []

        except Exception as e:
            logger.error(f"姿态检测失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [], []

    def _detect_yolov7_pose_from_boxes(self, image: np.ndarray, boxes: List[List[float]]) -> Tuple[List[List[List[float]]], List[float]]:
        """使用YOLOv7-pose从给定的边界框中检测姿态"""
        try:
            # 获取图像尺寸
            height, width = image.shape[:2]

            # 准备结果列表
            keypoints_list = []
            scores_list = []

            # 对每个边界框单独处理
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]

                # 确保坐标在图像范围内
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))

                # 如果框太小，跳过
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue

                # 裁剪出框内的图像
                cropped_image = image[y1:y2, x1:x2].copy()

                # 处理裁剪图像
                # 调整图像尺寸
                img = self._letterbox(
                    cropped_image, self.imgsz, stride=self.stride)[0]

                # 转换为PyTorch张量
                img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                img = np.ascontiguousarray(img)

                img = torch.from_numpy(img).to(self.device)
                img = img.float()
                img /= 255.0
                if len(img.shape) == 3:
                    img = img.unsqueeze(0)

                # 推理
                with torch.no_grad():
                    output, _ = self.model(img)
                    # NMS
                    output = self.non_max_suppression_kpt(
                        output, self.conf_threshold, self.keypoint_threshold, nc=1)

                # 处理输出
                for i, det in enumerate(output):
                    if len(det):
                        # 重新调整到原始图像尺寸
                        scale = torch.tensor([cropped_image.shape[1], cropped_image.shape[0],
                                              cropped_image.shape[1], cropped_image.shape[0]]).to(self.device)

                        for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6].detach().cpu().numpy())):
                            # 获取关键点
                            kpts = det[j, 6:].detach().cpu().numpy()
                            kpts = kpts.reshape(-1, 3)  # 17, 3

                            # 重新调整关键点到裁剪图像尺寸
                            crop_width = cropped_image.shape[1]
                            crop_height = cropped_image.shape[0]
                            r = min(self.imgsz / crop_width,
                                    self.imgsz / crop_height)
                            pad_w = (self.imgsz - crop_width * r) / 2
                            pad_h = (self.imgsz - crop_height * r) / 2

                            # 调整关键点坐标
                            adjusted_kpts = kpts.copy()
                            for k in range(len(kpts)):
                                adjusted_kpts[k][0] = (
                                    kpts[k][0] - pad_w) / r + x1  # 调整到原图x坐标
                                adjusted_kpts[k][1] = (
                                    kpts[k][1] - pad_h) / r + y1  # 调整到原图y坐标

                            keypoints_list.append(adjusted_kpts.tolist())
                            scores_list.append(float(conf))

            return keypoints_list, scores_list

        except Exception as e:
            logger.error(f"YOLOv7从边界框检测姿态失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [], []

    def _detect_rtmpose_from_boxes(self, image, boxes, threshold=0.5):
        """使用RTMPose检测人体姿态关键点

        Args:
            image: 输入图像
            boxes: 人体边界框，格式为[[x1,y1,x2,y2], ...]
            threshold: 关键点置信度阈值

        Returns:
            keypoints_list: 关键点列表，格式为[[[x,y,v], ...], ...]
            scores_list: 关键点置信度列表，格式为[[s1, s2, ...], ...]
        """
        try:
            if self.rtmpose_model is None:
                logger.warning("RTMPose模型未初始化，将返回空结果")
                return [], []

            # RTMPose需要RGB格式
            if len(image.shape) == 3 and image.shape[2] == 3:
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
            else:
                logger.warning(f"输入图像格式不支持: {image.shape}, {image.dtype}")
                return [], []

            img_h, img_w = image.shape[:2]
            logger.debug(f"原始图像尺寸: {img_w}x{img_h}")

            # 获取模型输入尺寸
            try:
                # 检查rtmpose_model是否为数组
                if isinstance(self.rtmpose_model, np.ndarray):
                    logger.warning("RTMPose模型是NumPy数组，使用默认输入尺寸")
                    w = h = 256
                elif isinstance(self.rtmpose_model, (list, tuple)):
                    logger.warning("RTMPose模型是列表或元组，使用默认输入尺寸")
                    w = h = 256
                else:
                    # 正常的模型对象
                    if hasattr(self.rtmpose_model, 'cfg') and hasattr(self.rtmpose_model.cfg, 'model'):
                        cfg_model = self.rtmpose_model.cfg.model
                        if hasattr(cfg_model, 'backbone') and hasattr(cfg_model.backbone, 'input_size'):
                            w = h = self.rtmpose_model.cfg.model.backbone.input_size
                        else:
                            w = h = 256  # 默认输入尺寸
                    else:
                        w = h = 256  # 默认输入尺寸
            except Exception as e:
                logger.warning(f"获取模型输入尺寸失败: {e}，使用默认值")
                w = h = 256  # 默认输入尺寸

            logger.debug(f"模型输入尺寸: {w}x{h}")

            # 转换边界框格式为numpy数组
            if len(boxes) == 0:
                logger.warning("没有检测到边界框，将返回空结果")
                return [], []

            np_boxes = np.array(boxes)
            logger.debug(f"边界框数量: {len(boxes)}, 格式: {np_boxes.shape}")

            # 构建预测实例
            predictions = []

            # 对每个边界框使用topdown方法预测关键点
            try:
                from mmpose.apis import inference_topdown

                # 检查模型类型
                if isinstance(self.rtmpose_model, (np.ndarray, list, tuple)):
                    logger.warning("RTMPose模型类型不正确，无法使用inference_topdown")
                    return [], []

                # 确保图像是RGB格式
                if len(image.shape) == 2:  # 灰度图像
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:  # 带透明通道
                    image = image[:, :, :3]
                elif image.shape[2] == 3 and image.dtype == np.float32:
                    # 如果是浮点图像且值范围在[0,1]，转换为uint8
                    if np.max(image) <= 1.0:
                        image = (image * 255).astype(np.uint8)

                # 调用姿态估计推理函数
                results = inference_topdown(
                    self.rtmpose_model, image, np_boxes)
                logger.debug(f"获取到{len(results)}个姿态预测结果")
            except Exception as e:
                logger.error(f"调用inference_topdown失败: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                return [], []

            keypoints_list = []
            scores_list = []

            # 处理结果
            try:
                for i, result in enumerate(results):
                    # 检查结果对象类型
                    if not hasattr(result, 'pred_instances'):
                        logger.warning(f"结果{i}没有pred_instances属性，跳过")
                        continue

                    if not hasattr(result.pred_instances, 'keypoints'):
                        logger.warning(f"结果{i}没有keypoints属性，跳过")
                        continue

                    # 获取关键点和得分
                    keypoints = result.pred_instances.keypoints

                    # 转换PyTorch张量为NumPy数组
                    if hasattr(keypoints, 'cpu') and hasattr(keypoints, 'numpy'):
                        keypoints = keypoints.cpu().numpy()

                    # 检查得分属性
                    if hasattr(result.pred_instances, 'keypoint_scores'):
                        scores = result.pred_instances.keypoint_scores
                        # 转换得分为NumPy数组
                        if hasattr(scores, 'cpu') and hasattr(scores, 'numpy'):
                            scores = scores.cpu().numpy()
                    else:
                        # 如果没有关键点得分，创建默认得分(全1)
                        logger.warning(f"结果{i}没有keypoint_scores属性，使用默认值")
                        scores = np.ones(keypoints.shape[0], dtype=np.float32)

                    # MMPose返回的关键点格式为[K, 2]，需要转换为[K, 3]
                    if len(keypoints.shape) == 2 and keypoints.shape[1] == 2:
                        logger.debug(f"检测到关键点格式为[K, 2]，转换为[K, 3]格式")
                        # 创建可见性列为全1的[K, 3]格式关键点
                        keypoints_with_v = np.zeros(
                            (keypoints.shape[0], 3), dtype=np.float32)
                        keypoints_with_v[:, 0:2] = keypoints
                        keypoints_with_v[:, 2] = 1.0  # 设置可见性为1
                        keypoints = keypoints_with_v

                    # 应用置信度阈值筛选
                    visible = scores >= threshold
                    for k in range(len(keypoints)):
                        if not visible[k]:
                            keypoints[k, 2] = 0  # 将低于阈值的关键点可见性设为0

                    keypoints_list.append(keypoints)
                    scores_list.append(scores)
            except Exception as e:
                logger.error(f"处理推理结果失败: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                return [], []  # 返回空结果

            logger.debug(f"成功检测到{len(keypoints_list)}个姿态关键点")
            return keypoints_list, scores_list

        except Exception as e:
            logger.error(f"RTMPose姿态检测失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return [], []  # 不再使用备用模型

    def visualize_poses(self, image: np.ndarray, keypoints: List[List[List[float]]], scores: List[float] = None) -> np.ndarray:
        """
        在图像上可视化姿态估计结果

        Args:
            image: 输入图像
            keypoints: 关键点列表 [N, K, 3] - (x, y, conf)
            scores: 人体检测的置信度列表 [N]

        Returns:
            vis_image: 可视化后的图像
        """
        try:
            # 根据模型类型选择不同的可视化方法
            if self.model_name.startswith("rtmpose"):
                return self._visualize_rtmpose(image, keypoints, scores)
            elif self.model_name.startswith("hrnet"):
                # HRNet 姿态估计模型的可视化
                return self._visualize_hrnet(image, keypoints, scores)
            elif self.model_name == "yolov7_w6_pose":
                # YOLOv7 姿态估计模型的可视化
                return self._visualize_poses_generic(image, keypoints, scores)
            elif self.model_name == "keypointrcnn_resnet50_fpn":
                # KeypointRCNN 的可视化
                return self._visualize_poses_generic(image, keypoints, scores)
            else:
                # 通用的可视化方法
                return self._visualize_poses_generic(image, keypoints, scores)
        except Exception as e:
            logger.error(f"可视化姿态失败: {e}")
            return image.copy()

    def _visualize_rtmpose(self, image: np.ndarray, keypoints: List[List[List[float]]], scores: List[float] = None) -> np.ndarray:
        """
        使用RTMPose可视化器可视化姿态

        Args:
            image: 输入图像
            keypoints: 关键点列表 [N, K, 3] - (x, y, score)
            scores: 姿态分数列表 [N]

        Returns:
            vis_image: 可视化后的图像
        """
        try:
            from mmpose.structures import PoseDataSample
            from mmengine.registry import init_default_scope
            import torch
            import numpy as np

            # 确保正确的默认作用域
            init_default_scope('mmpose')

            # 确保图像是RGB格式
            if image.shape[2] == 4:  # 如果有alpha通道
                image = image[:, :, :3]

            # 从BGR转换为RGB (如果需要)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 记录图像尺寸和关键点范围（调试信息）
            img_height, img_width = image.shape[:2]
            logger.debug(f"可视化图像尺寸: {img_width}x{img_height}")

            # 记录关键点范围
            if keypoints and len(keypoints) > 0:
                all_kpts = []
                for person_kpts in keypoints:
                    for kpt in person_kpts:
                        if kpt[2] > 0:  # 只考虑可见的关键点
                            all_kpts.append(kpt[:2])

                if all_kpts:
                    all_kpts = np.array(all_kpts)
                    x_min, y_min = np.min(all_kpts, axis=0)
                    x_max, y_max = np.max(all_kpts, axis=0)
                    logger.debug(
                        f"可视化关键点范围: x({x_min:.1f}-{x_max:.1f}) y({y_min:.1f}-{y_max:.1f})")

            # 如果没有关键点，直接返回原图
            if not keypoints or len(keypoints) == 0:
                return image.copy()

            # 准备可视化数据
            pose_results = PoseDataSample()

            # 在可视化前复制一份关键点数据，确保坐标在图像范围内
            keypoints_copy = []
            for person_kpts in keypoints:
                person_kpts_copy = []
                for x, y, score in person_kpts:
                    # 确保坐标在图像范围内
                    x_valid = max(0, min(float(x), img_width - 1))
                    y_valid = max(0, min(float(y), img_height - 1))
                    person_kpts_copy.append([x_valid, y_valid, score])
                keypoints_copy.append(person_kpts_copy)

            # 转换数据格式
            # 将keypoints从list转换为tensor
            # [person_num, keypoint_num, 3]
            keypoints_array = np.array(keypoints_copy)

            # 关键点坐标和分数
            # [person_num, keypoint_num, 2]
            keypoints_xy = keypoints_array[:, :, :2]
            # [person_num, keypoint_num]
            keypoints_score = keypoints_array[:, :, 2]

            # 如果没有提供姿态分数，使用关键点分数的平均值
            if scores is None:
                # [person_num]
                scores = np.mean(keypoints_score, axis=1).tolist()

            # 转换为tensor
            keypoints_tensor = torch.from_numpy(keypoints_xy).float()
            keypoints_score_tensor = torch.from_numpy(keypoints_score).float()
            scores_tensor = torch.tensor(scores).float()

            # 设置数据
            pose_results.pred_instances = {
                'keypoints': keypoints_tensor,
                'keypoint_scores': keypoints_score_tensor,
                'scores': scores_tensor
            }

            # 添加骨架连接和关键点名称信息
            try:
                pose_results.metainfo = {
                    'skeleton_links': COCO_SKELETON,
                    'keypoint_names': COCO_KEYPOINTS,
                    'flip_indices': None
                }
            except Exception as e:
                logger.warning(f"设置骨架和关键点元信息失败: {e}")

            # 初始化可视化器
            if not hasattr(self, 'visualizer') or self.visualizer is None:
                # 模型没有初始化可视化器，使用通用可视化方法
                return self._visualize_poses_generic(image, keypoints, scores)

            # 清除旧的结果
            self.visualizer.reset()

            # 使用可视化器
            self.visualizer.set_image(image_rgb.copy())
            self.visualizer.draw_pose_results(pose_results)

            # 获取可视化结果
            vis_result = self.visualizer.get_image()

            # 添加调试信息
            # 在图像上添加坐标范围和关键点数量信息
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            font_color = (255, 255, 255)

            # 在底部添加关键点信息
            if keypoints and len(keypoints) > 0:
                info_text = f"关键点数量: {len(keypoints)}, 每人关键点: {len(keypoints[0])}"
                cv2.putText(vis_result, info_text, (10, img_height - 10),
                            font, font_scale, font_color, font_thickness)

            # 如果需要，转换回BGR
            vis_result = cv2.cvtColor(vis_result, cv2.COLOR_RGB2BGR)

            return vis_result

        except Exception as e:
            logger.error(f"RTMPose可视化失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 发生错误，返回原图
            return image.copy()

    def _visualize_keypoints_only(self, image: np.ndarray, keypoints: List[List[List[float]]], scores: List[float] = None) -> np.ndarray:
        """只绘制关键点不绘制骨骼的可视化方法"""
        vis_image = image.copy()

        # 绘制每个检测到的姿态
        for i, kpts in enumerate(keypoints):
            # 绘制关键点
            for j, kpt in enumerate(kpts):
                x, y, conf = kpt

                # 只绘制置信度高于阈值的关键点
                if conf > self.keypoint_threshold:
                    # 获取关键点颜色
                    color = KEYPOINT_COLORS[j]
                    # 绘制关键点
                    cv2.circle(vis_image, (int(x), int(y)), 5, color, -1)

            # 显示检测分数
            if scores and i < len(scores):
                score = scores[i]
                cv2.putText(vis_image, f"score: {score:.2f}",
                            (int(kpts[0][0]), int(kpts[0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return vis_image

    def _visualize_poses_generic(self, image: np.ndarray, keypoints: List[List[List[float]]], scores: List[float] = None) -> np.ndarray:
        """通用的姿态可视化方法"""
        try:
            vis_image = image.copy()

            # 获取图像尺寸
            img_height, img_width = vis_image.shape[:2]
            logger.debug(f"通用可视化图像尺寸: {img_width}x{img_height}")

            # 记录关键点范围
            if keypoints and len(keypoints) > 0:
                all_kpts = []
                for person_kpts in keypoints:
                    for kpt in person_kpts:
                        if kpt[2] > 0:  # 只考虑可见的关键点
                            all_kpts.append(kpt[:2])

                if all_kpts:
                    all_kpts = np.array(all_kpts)
                    x_min, y_min = np.min(all_kpts, axis=0)
                    x_max, y_max = np.max(all_kpts, axis=0)
                    logger.debug(
                        f"通用可视化关键点范围: x({x_min:.1f}-{x_max:.1f}) y({y_min:.1f}-{y_max:.1f})")

            # 定义线条颜色
            skeleton_color = (100, 100, 255)

            # 绘制每个检测到的姿态
            for i, kpts in enumerate(keypoints):
                # 绘制骨架
                if self.draw_skeleton:  # 根据设置决定是否绘制骨骼
                    for p1_idx, p2_idx in COCO_SKELETON:
                        if p1_idx < len(kpts) and p2_idx < len(kpts):
                            p1 = kpts[p1_idx]
                            p2 = kpts[p2_idx]

                            # 确保两个关键点都可见且在图像范围内
                            if p1[2] > self.keypoint_threshold and p2[2] > self.keypoint_threshold:
                                # 确保坐标在图像范围内
                                p1_x = max(0, min(int(p1[0]), img_width-1))
                                p1_y = max(0, min(int(p1[1]), img_height-1))
                                p2_x = max(0, min(int(p2[0]), img_width-1))
                                p2_y = max(0, min(int(p2[1]), img_height-1))

                                p1_pos = (p1_x, p1_y)
                                p2_pos = (p2_x, p2_y)
                                cv2.line(vis_image, p1_pos,
                                         p2_pos, skeleton_color, 2)

                # 绘制关键点
                for j, kpt in enumerate(kpts):
                    if j < len(COCO_KEYPOINTS):  # 确保不超出关键点列表范围
                        x, y, conf = kpt

                        # 只绘制置信度高于阈值的关键点
                        if conf > self.keypoint_threshold:
                            # 确保坐标在图像范围内
                            x_valid = max(0, min(int(x), img_width-1))
                            y_valid = max(0, min(int(y), img_height-1))

                            # 获取关键点颜色
                            color = KEYPOINT_COLORS[j % len(KEYPOINT_COLORS)]

                            # 绘制关键点
                            cv2.circle(
                                vis_image, (x_valid, y_valid), 5, color, -1)

                            # 可选：添加关键点标签
                            label = COCO_KEYPOINTS[j]
                            cv2.putText(vis_image, f"{j}", (x_valid+5, y_valid),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # 显示检测分数
                if scores and i < len(scores):
                    score = scores[i]
                    try:
                        nose_x, nose_y = int(kpts[0][0]), int(kpts[0][1])
                        nose_x = max(0, min(nose_x, img_width-1))
                        nose_y = max(0, min(nose_y, img_height-1))
                        cv2.putText(vis_image, f"score: {score:.2f}",
                                    (nose_x, nose_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    except Exception as e:
                        logger.warning(f"显示分数失败: {e}")

            # 添加调试信息
            # 在图像上添加坐标范围和关键点数量信息
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1
            font_color = (255, 255, 255)

            # 添加边界框
            cv2.rectangle(vis_image, (0, 0), (img_width -
                          1, img_height-1), (0, 255, 0), 1)

            # 在底部添加关键点信息
            if keypoints and len(keypoints) > 0:
                info_text = f"人数: {len(keypoints)}, 每人关键点: {len(keypoints[0])}"
                cv2.putText(vis_image, info_text, (10, img_height - 10),
                            font, font_scale, font_color, font_thickness)

            return vis_image
        except Exception as e:
            logger.error(f"通用姿态可视化失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 发生错误，返回原图
            return image.copy()

    def _letterbox(self, img, new_shape=(640, 640), stride=32):
        """
        调整图像大小并在边缘添加填充。
        """
        # 获取原始尺寸
        shape = img.shape[:2]  # 当前尺寸 [高, 宽]

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # 计算缩放比例和填充
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # 计算填充
        ratio = r, r  # 宽高缩放比例
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - \
            new_unpad[0], new_shape[0] - new_unpad[1]  # wh填充

        # 分配填充到左/右、上/下
        dw /= 2  # 分割填充到左和右
        dh /= 2  # 分割填充到上和下

        # 如果形状不同，则调整图像大小
        if shape[::-1] != new_unpad:  # 调整大小
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        # 添加填充
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # 添加边框

        return img, ratio, (dw, dh)

    def _detect_keypointrcnn_backup(self, image: np.ndarray) -> Tuple[List[List[List[float]]], List[float]]:
        """使用KeypointRCNN备用模型检测关键点"""
        try:
            import torch
            import torchvision.transforms.functional as F

            # 记录开始时间
            t_start = time.time()

            # 获取高级参数
            max_poses = self.advanced_params.get("max_poses", 20)
            min_keypoints = self.advanced_params.get("min_keypoints", 5)

            # 准备结果列表
            keypoints_list = []
            scores_list = []

            # 确保图像是RGB格式
            if image.shape[2] == 4:  # 如果有alpha通道
                image = image[:, :, :3]

            # 从BGR转换为RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 转换为PyTorch张量
            img_tensor = F.to_tensor(image_rgb)

            # 确保在正确的设备上
            img_tensor = img_tensor.to(self.device)

            logger.debug(
                f"KeypointRCNN输入图像形状: {img_tensor.shape}, 设备: {img_tensor.device}")

            # 使用备用模型进行推理
            with torch.no_grad():
                try:
                    # 新版本torchvision模型
                    predictions = self.keypointrcnn_model([img_tensor])
                    logger.debug(f"KeypointRCNN预测结果类型: {type(predictions)}")

                    # 检查预测结果是否为列表且有内容
                    if isinstance(predictions, list) and len(predictions) > 0:
                        # 处理新版本torchvision模型的输出
                        pred = predictions[0]

                        if 'keypoints' in pred and len(pred['keypoints']) > 0:
                            # 提取关键点和分数
                            pred_keypoints = pred['keypoints'].cpu().numpy()
                            pred_scores = pred['scores'].cpu().numpy()
                            pred_boxes = pred['boxes'].cpu(
                            ).numpy() if 'boxes' in pred else None

                            # 过滤低置信度的检测结果
                            mask = pred_scores >= self.conf_threshold
                            filtered_keypoints = pred_keypoints[mask]
                            filtered_scores = pred_scores[mask]

                            # 处理每个检测到的人
                            for i, kpts in enumerate(filtered_keypoints):
                                # 将keypoint_scores转成confidence
                                if kpts.shape[1] == 3:
                                    conf = kpts[:, 2]
                                else:
                                    # 如果没有置信度，使用默认值
                                    conf = np.ones(kpts.shape[0]) * 0.9

                                # 创建关键点数组 [K, 3] - (x, y, conf)
                                keypoints_array = np.zeros((kpts.shape[0], 3))
                                keypoints_array[:, :2] = kpts[:, :2]
                                keypoints_array[:, 2] = conf

                                # 计算可见关键点数量
                                visible_keypoints = sum(
                                    1 for _, _, c in keypoints_array if c >= self.keypoint_threshold)

                                # 过滤掉可见关键点数量少于阈值的姿态
                                if visible_keypoints >= min_keypoints:
                                    keypoints_list.append(
                                        keypoints_array.tolist())
                                    scores_list.append(
                                        float(filtered_scores[i]))

                                    # 如果达到最大姿态数量，停止添加
                                    if len(keypoints_list) >= max_poses:
                                        break
                except Exception as model_error:
                    logger.warning(f"使用标准模型推理失败: {model_error}，尝试其他格式")
                    # 记录异常信息以便调试
                    import traceback
                    logger.debug(f"异常详情: {traceback.format_exc()}")

                    # 尝试使用旧版本torchvision模型的推理方式
                    try:
                        # 检查模型是否有eval方法
                        if hasattr(self.keypointrcnn_model, 'eval'):
                            # 确保模型处于评估模式
                            self.keypointrcnn_model.eval()

                        # 转换为元组推理
                        outputs = self.keypointrcnn_model(img_tensor)

                        # 检查结果结构
                        if isinstance(outputs, tuple) and len(outputs) >= 2:
                            # 获取边界框和关键点
                            boxes = outputs[0]
                            keypoints = outputs[1]

                            # 提取关键点和分数
                            if len(keypoints) > 0:
                                keypoints_np = keypoints.cpu().numpy()

                                for i, kpts in enumerate(keypoints_np):
                                    # 创建关键点数组 [K, 3] - (x, y, conf)
                                    keypoints_array = np.zeros(
                                        (kpts.shape[0], 3))
                                    keypoints_array[:, :2] = kpts[:, :2]
                                    # 使用默认置信度
                                    keypoints_array[:, 2] = 0.9

                                    keypoints_list.append(
                                        keypoints_array.tolist())
                                    scores_list.append(0.9)  # 默认分数

                                    # 如果达到最大姿态数量，停止添加
                                    if len(keypoints_list) >= max_poses:
                                        break
                    except Exception as alt_error:
                        logger.error(f"所有推理方式都失败: {alt_error}")
                        import traceback
                        logger.error(f"异常详情: {traceback.format_exc()}")

            logger.info(
                f"KeypointRCNN备用模型检测找到{len(keypoints_list)}个姿态，耗时: {time.time() - t_start:.3f}秒")
            return keypoints_list, scores_list

        except Exception as e:
            logger.error(f"KeypointRCNN备用模型检测失败: {str(e)}")
            import traceback
            logger.error(f"异常详情: {traceback.format_exc()}")
            return [], []

    def _detect_keypointrcnn_from_boxes(self, image: np.ndarray, boxes: List[List[float]]) -> Tuple[List[List[List[float]]], List[float]]:
        """
        使用KeypointRCNN从给定边界框中检测姿态

        Args:
            image: 输入图像
            boxes: 边界框列表 [N, 4] - (x1, y1, x2, y2)

        Returns:
            keypoints: 关键点列表 [N, K, 3] - (x, y, conf)
            scores: 人体检测的置信度列表 [N]
        """
        try:
            # 获取图像尺寸
            height, width = image.shape[:2]

            # 准备结果列表
            keypoints_list = []
            scores_list = []

            # 添加日志记录模型类型
            try:
                model_type = type(self.model).__name__
                logger.debug(f"备用模型类型: {model_type}")
            except Exception as e:
                logger.warning(f"获取模型类型出错: {e}")

            # 确定使用哪种方法处理
            is_mmpose_model = not 'KeypointRCNN' in str(type(self.model))
            logger.debug(f"是否为MMPose模型: {is_mmpose_model}")

            # 对每个边界框单独处理
            for box in boxes:
                if len(box) < 4:
                    continue

                x1, y1, x2, y2 = [int(coord) for coord in box[:4]]

                # 确保坐标在图像范围内
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))

                # 如果框太小，跳过
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue

                logger.debug(
                    f"处理边界框: x1={x1}, y1={y1}, x2={x2}, y2={y2}, 尺寸={x2-x1}x{y2-y1}")

                # 裁剪出框内的图像
                cropped_image = image[y1:y2, x1:x2].copy()
                logger.debug(f"裁剪图像尺寸: {cropped_image.shape}")

                # 转换为RGB格式
                cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

                # 调整图像大小
                try:
                    # 确定适当的输入大小
                    if is_mmpose_model:
                        # 对于MMPose模型，使用256x192
                        target_size = (192, 256)
                    else:
                        # 对于torchvision模型，使用原尺寸或缩放
                        h, w = cropped_rgb.shape[:2]
                        min_size = 800
                        if hasattr(self.model, 'transform') and hasattr(self.model.transform, 'min_size'):
                            min_size = self.model.transform.min_size[0]
                        scale = min_size / max(h, w)
                        new_h, new_w = int(h * scale), int(w * scale)
                        new_h, new_w = max(1, new_h), max(1, new_w)
                        target_size = (new_w, new_h)

                    logger.debug(f"调整图像到目标尺寸: {target_size}")
                    resized_img = cv2.resize(cropped_rgb, target_size)
                    cropped_rgb = resized_img
                except Exception as e:
                    logger.warning(f"调整图像大小失败: {e}")

                # 转换为PyTorch张量
                cropped_tensor = torch.from_numpy(cropped_rgb.transpose(
                    2, 0, 1)).float().div(255.0).unsqueeze(0)
                cropped_tensor = cropped_tensor.to(self.device)
                logger.debug(f"输入张量尺寸: {cropped_tensor.shape}")

                # 根据模型类型选择推理方法
                if is_mmpose_model:
                    # MMPose模型处理
                    try:
                        # 创建数据样本
                        from mmengine.structures import InstanceData
                        from mmpose.structures import PoseDataSample

                        curr_h, curr_w = cropped_rgb.shape[:2]
                        data_sample = PoseDataSample()
                        # 确保包含flip_indices元信息
                        data_sample.set_metainfo({
                            'img_shape': (curr_h, curr_w),
                            'img_id': 0,
                            'input_size': (curr_w, curr_h),
                            'flip_indices': self.model.dataset_meta.get('flip_indices', None)
                        })

                        # 尝试使用test_step方法
                        if hasattr(self.model, 'test_step'):
                            data_batch = {'inputs': cropped_tensor,
                                          'data_samples': [data_sample]}
                            outputs = self.model.test_step(data_batch)
                        else:
                            # 使用forward方法
                            outputs = self.model(
                                cropped_tensor, mode='test', data_samples=[data_sample])

                        # 处理输出
                        if outputs and len(outputs) > 0:
                            pose_result = outputs[0]
                            if hasattr(pose_result, 'pred_instances') and len(pose_result.pred_instances) > 0:
                                pred_instances = pose_result.pred_instances

                                # 安全获取keypoints和scores
                                keypoints = pred_instances.keypoints
                                if isinstance(keypoints, torch.Tensor):
                                    keypoints = keypoints.cpu().numpy()

                                keypoint_scores = pred_instances.keypoint_scores
                                if isinstance(keypoint_scores, torch.Tensor):
                                    keypoint_scores = keypoint_scores.cpu().numpy()

                                # 获取实例分数
                                instance_score = 1.0
                                if hasattr(pred_instances, 'scores'):
                                    scores = pred_instances.scores
                                    if isinstance(scores, torch.Tensor) and len(scores) > 0:
                                        instance_score = scores[0].cpu().item()
                                    elif len(scores) > 0:
                                        instance_score = float(scores[0])

                                # 处理每个姿态
                                for i in range(len(keypoints)):
                                    # 转换坐标回原始图像
                                    scale_x = (x2 - x1) / \
                                        curr_w if curr_w > 0 else 1
                                    scale_y = (y2 - y1) / \
                                        curr_h if curr_h > 0 else 1

                                    kpts = keypoints[i].copy()
                                    kpts[:, 0] = kpts[:, 0] * scale_x + x1
                                    kpts[:, 1] = kpts[:, 1] * scale_y + y1

                                    # 创建包含置信度的关键点
                                    kpts_with_conf = np.zeros((len(kpts), 3))
                                    kpts_with_conf[:, :2] = kpts
                                    kpts_with_conf[:, 2] = keypoint_scores[i]

                                    keypoints_list.append(
                                        kpts_with_conf.tolist())
                                    scores_list.append(float(instance_score))
                    except Exception as e:
                        logger.error(f"MMPose模型推理失败: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                else:
                    # Torchvision KeypointRCNN模型处理
                    try:
                        with torch.no_grad():
                            predictions = self.model(cropped_tensor)

                        # 处理输出
                        if predictions and len(predictions) > 0 and 'keypoints' in predictions[0] and len(predictions[0]['keypoints']) > 0:
                            # 获取关键点和分数
                            keypoints = predictions[0]['keypoints'][0].cpu(
                            ).numpy()
                            score = 1.0
                            if 'scores' in predictions[0] and len(predictions[0]['scores']) > 0:
                                score = predictions[0]['scores'][0].cpu(
                                ).item()

                            # 转换坐标回原始图像
                            if keypoints.shape[0] > 0:
                                h, w = target_size[1], target_size[0]
                                scale_x = (x2 - x1) / w if w > 0 else 1
                                scale_y = (y2 - y1) / h if h > 0 else 1

                                adjusted_keypoints = keypoints.copy()
                                adjusted_keypoints[:,
                                                   0] = adjusted_keypoints[:, 0] * scale_x + x1
                                adjusted_keypoints[:,
                                                   1] = adjusted_keypoints[:, 1] * scale_y + y1

                                keypoints_list.append(
                                    adjusted_keypoints.tolist())
                                scores_list.append(float(score))
                    except Exception as e:
                        logger.error(f"Torchvision模型推理失败: {e}")
                        import traceback
                        logger.error(traceback.format_exc())

            return keypoints_list, scores_list

        except Exception as e:
            logger.error(f"KeypointRCNN从边界框检测姿态失败: {str(e)}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return [], []

    def detect_poses(self, image: np.ndarray) -> Tuple[List[List[List[float]]], List[float]]:
        """
        检测图像中的人体姿态

        Args:
            image: 输入图像数组

        Returns:
            (keypoints, scores): 关键点列表和置信度列表
        """
        start_time = time.time()

        # 检查图像有效性
        if image is None or image.size == 0:
            logger.warning("输入图像为空")
            return [], []

        # 使用不同的检测方法
        if self.is_rtmpose:
            keypoints, scores = self._detect_rtmpose(image)
        elif self.is_hrnet:
            keypoints, scores = self._detect_hrnet(image)
        elif self.is_keypointrcnn:
            keypoints, scores = self._detect_keypointrcnn(image)
        else:
            keypoints, scores = self._detect_yolov7_pose(image)

        end_time = time.time()
        logger.info(f"姿态估计耗时: {end_time - start_time:.2f}秒")

        # 返回检测结果
        return keypoints, scores

    def _detect_hrnet(self, image: np.ndarray) -> Tuple[List[List[List[float]]], List[float]]:
        """
        使用HRNet模型检测图像中的人体姿态

        Args:
            image: 输入图像

        Returns:
            keypoints: 关键点列表 [N, K, 3] - (x, y, conf)
            scores: 人体检测的置信度列表 [N]
        """
        try:
            import torch
            from mmpose.apis import inference_topdown
            from mmpose.evaluation.functional import nms
            from mmpose.structures import merge_data_samples
            from mmdet.apis import inference_detector

            # 保存原始图像尺寸
            img_h, img_w = image.shape[:2]

            # 获取检测结果
            if self.detector_model is not None:
                # 使用RTMDet模型进行人体检测
                det_result = inference_detector(self.detector_model, image)
                pred_instance = det_result.pred_instances.cpu().numpy()

                # 过滤置信度较低的检测框
                bboxes = np.concatenate(
                    (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
                bboxes = bboxes[np.logical_and(
                    pred_instance.labels == 0, pred_instance.scores > self.conf_threshold)]

                # 应用NMS
                bboxes = bboxes[nms(bboxes, 0.3)][:, :4]
            else:
                # 如果没有检测模型，则假设整个图像都是人体
                bboxes = np.array([[0, 0, img_w, img_h]])

            # 如果没有检测到人体
            if len(bboxes) == 0:
                logger.info("没有检测到人体")
                return [], []

            # 使用HRNet模型进行关键点检测
            pose_results = inference_topdown(self.model, image, bboxes)

            # 合并结果
            data_samples = merge_data_samples(pose_results)

            # 提取关键点和置信度
            keypoints_data = []
            scores_data = []

            # 检查姿态结果
            for i, pose_result in enumerate(pose_results):
                # 获取关键点
                instance_keypoints = pose_result.pred_instances.keypoints
                instance_keypoint_scores = pose_result.pred_instances.keypoint_scores

                # 转换为需要的格式 [K, 3] - (x, y, conf)
                kpts = []
                for j in range(len(instance_keypoints[0])):
                    x, y = instance_keypoints[0][j]
                    conf = float(instance_keypoint_scores[0][j])

                    # 应用置信度阈值
                    visible_conf = conf if conf >= self.keypoint_threshold else 0.0
                    kpts.append([float(x), float(y), visible_conf])

                keypoints_data.append(kpts)

                # 使用检测框的置信度作为整体置信度
                if self.detector_model is not None and i < len(bboxes):
                    score = float(pred_instance.scores[i]) if i < len(
                        pred_instance.scores) else 1.0
                else:
                    score = 1.0
                scores_data.append(score)

            return keypoints_data, scores_data

        except Exception as e:
            logger.error(f"HRNet姿态检测失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [], []

    def _visualize_hrnet(self, image: np.ndarray, keypoints: List[List[List[float]]], scores: List[float] = None) -> np.ndarray:
        """使用HRNet模型的可视化功能

        Args:
            image: 输入图像
            keypoints: 关键点列表 [N, K, 3] - (x, y, conf)
            scores: 人体置信度列表 [N]

        Returns:
            np.ndarray: 可视化后的图像
        """
        try:
            from mmpose.structures import merge_data_samples

            # 如果没有检测到姿态，则直接返回原图
            if not keypoints or len(keypoints) == 0:
                return image.copy()

            # 复制图像
            img = image.copy()

            # 准备可视化所需的数据结构
            # 创建假的检测框（与关键点匹配）
            bboxes = []
            for person_kpts in keypoints:
                valid_points = np.array(
                    [(p[0], p[1]) for p in person_kpts if p[2] > self.keypoint_threshold])
                if len(valid_points) > 0:
                    x_min, y_min = valid_points.min(axis=0)
                    x_max, y_max = valid_points.max(axis=0)
                    # 扩大bounding box
                    width = x_max - x_min
                    height = y_max - y_min
                    x_min = max(0, x_min - width * 0.1)
                    y_min = max(0, y_min - height * 0.1)
                    x_max = min(img.shape[1], x_max + width * 0.1)
                    y_max = min(img.shape[0], y_max + height * 0.1)
                    bboxes.append([x_min, y_min, x_max, y_max])
                else:
                    # 如果没有有效点，则使用整个图像
                    bboxes.append([0, 0, img.shape[1], img.shape[0]])

            # 转换为numpy数组
            bboxes = np.array(bboxes, dtype=np.float32)

            # 构造mmpose数据样本
            from mmpose.structures import PoseDataSample, InstanceData
            import torch

            pose_samples = []
            for i, person_kpts in enumerate(keypoints):
                pose_sample = PoseDataSample()
                keypoints_tensor = torch.tensor(
                    [[[kpt[0], kpt[1]] for kpt in person_kpts]], device=self.device)
                keypoint_scores_tensor = torch.tensor(
                    [[kpt[2] for kpt in person_kpts]], device=self.device)

                pred_instances = InstanceData()
                pred_instances.keypoints = keypoints_tensor
                pred_instances.keypoint_scores = keypoint_scores_tensor
                if len(bboxes) > i:
                    bbox = bboxes[i]
                    pred_instances.bboxes = torch.tensor(
                        [[bbox[0], bbox[1], bbox[2], bbox[3]]], device=self.device)
                    pred_instances.scores = torch.tensor(
                        [scores[i] if scores and i < len(scores) else 1.0], device=self.device)

                pose_sample.pred_instances = pred_instances
                pose_samples.append(pose_sample)

            # 合并数据样本
            merged_sample = merge_data_samples(pose_samples)

            # 使用mmpose的visualizer进行可视化
            # 调整可视化参数
            self.visualizer.radius = 4  # 关键点半径
            self.visualizer.line_width = 2  # 线宽
            if hasattr(self.visualizer, 'kpt_thr'):
                self.visualizer.kpt_thr = self.keypoint_threshold  # 关键点阈值

            # 进行可视化
            vis_img = self.visualizer.add_datasample(
                'result',
                img,
                data_sample=merged_sample,
                draw_gt=False,
                draw_bbox=True,
                draw_heatmap=False,
                show_kpt_idx=False,
                show=False
            )

            return vis_img
        except Exception as e:
            logger.error(f"HRNet可视化失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # 如果可视化失败，则回退到通用可视化方法
            return self._visualize_poses_generic(image, keypoints, scores)

    def _detect_hrnet_from_boxes(self, image: np.ndarray, boxes: List[List[float]], threshold: float = 0.5) -> Tuple[List[List[List[float]]], List[float]]:
        """
        使用HRNet模型从边界框中检测人体姿态

        Args:
            image: 输入图像
            boxes: 边界框列表 [N, 4] - (x1, y1, x2, y2)
            threshold: 关键点置信度阈值

        Returns:
            keypoints: 关键点列表 [N, K, 3] - (x, y, conf)
            scores: 人体检测的置信度列表 [N]
        """
        try:
            import torch
            from mmpose.apis import inference_topdown
            from mmpose.structures import merge_data_samples

            # 如果没有检测框，则返回空结果
            if not boxes or len(boxes) == 0:
                logger.info("没有检测框")
                return [], []

            # 转换边界框格式为numpy数组
            bboxes = np.array(boxes, dtype=np.float32)

            # 使用HRNet模型进行关键点检测
            pose_results = inference_topdown(self.model, image, bboxes)

            # 提取关键点和置信度
            keypoints_data = []
            scores_data = []

            # 检查姿态结果
            for i, pose_result in enumerate(pose_results):
                # 获取关键点
                instance_keypoints = pose_result.pred_instances.keypoints
                instance_keypoint_scores = pose_result.pred_instances.keypoint_scores

                # 转换为需要的格式 [K, 3] - (x, y, conf)
                kpts = []
                for j in range(len(instance_keypoints[0])):
                    x, y = instance_keypoints[0][j]
                    conf = float(instance_keypoint_scores[0][j])

                    # 应用置信度阈值
                    visible_conf = conf if conf >= threshold else 0.0
                    kpts.append([float(x), float(y), visible_conf])

                keypoints_data.append(kpts)

                # 使用检测框的置信度作为整体置信度（如无则设为1.0）
                scores_data.append(1.0)

            return keypoints_data, scores_data

        except Exception as e:
            logger.error(f"HRNet从边界框检测姿态失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [], []

    def _init_rtmpose(self):
        """初始化RTMPose模型"""
        logger.info("开始初始化RTMPose模型...")
        # 检查MMPose是否可用
        try:
            import mmcv
            import mmpose
            from mmpose.apis import init_model
            logger.info(f"成功导入MMPose库: {mmpose.__version__}")
        except ImportError as e:
            logger.error(f"导入MMPose库失败: {e}")
            return False

        try:
            # 获取模型配置文件
            config_file = None

            # 先查找内置配置文件
            for model_name, cfg_path in RTMPOSE_MODEL_CONFIGS.items():
                if model_name.lower() == self.model_name.lower():
                    # 如果找到匹配的配置，使用内置配置路径
                    config_file = cfg_path
                    logger.info(f"使用内置RTMPose配置: {config_file}")
                    break

            if not config_file:
                # 如果未找到内置配置，从文件系统查找
                from labelme._automation.model_downloader import get_model_dir
                mmpose_dir = get_model_dir("mmpose")
                config_path = os.path.join(mmpose_dir, f"{self.model_name}.py")
                if os.path.exists(config_path):
                    config_file = config_path
                    logger.info(f"使用本地RTMPose配置文件: {config_file}")

            if not config_file:
                # 如果仍未找到，根据模型名称猜测配置文件
                if 'rtmpose-s' in self.model_name.lower():
                    config_file = 'labelme/_automation/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py'
                elif 'rtmpose-m' in self.model_name.lower():
                    config_file = 'labelme/_automation/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py'
                elif 'rtmpose-l' in self.model_name.lower():
                    config_file = 'labelme/_automation/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-l_8xb256-420e_coco-256x192.py'
                elif 'rtmpose-t' in self.model_name.lower():
                    config_file = 'labelme/_automation/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_coco-256x192.py'
                else:
                    # 默认使用s模型
                    config_file = 'labelme/_automation/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py'
                logger.info(f"根据模型名称猜测RTMPose配置: {config_file}")

            # 获取checkpoint文件
            checkpoint_file = None
            for model_name, ckpt_path in RTMPOSE_MODEL_CHECKPOINTS.items():
                if model_name.lower() == self.model_name.lower():
                    checkpoint_file = ckpt_path
                    logger.info(f"使用内置RTMPose权重路径: {checkpoint_file}")
                    break

            # 首先在_automation/mmpose/checkpoints目录查找
            try:
                from labelme._automation.model_downloader import get_model_dir
                local_checkpoint_dir = get_model_dir("mmpose")
                local_checkpoint_path = os.path.join(
                    local_checkpoint_dir, f"{self.model_name}.pth")
                if os.path.exists(local_checkpoint_path):
                    logger.info(
                        f"在本地目录找到RTMPose模型权重文件: {local_checkpoint_path}")
                    checkpoint_file = local_checkpoint_path
                else:
                    # 检查原有的缓存目录
                    checkpoint_dir = os.path.join(os.path.expanduser(
                        "~"), ".cache", "torch", "hub", "checkpoints")

                    # 检查checkpoint_file是否有效，避免拼接None
                    if checkpoint_file:
                        checkpoint_path = os.path.join(
                            checkpoint_dir, os.path.basename(checkpoint_file))

                        if os.path.exists(checkpoint_path):
                            logger.info(
                                f"在缓存目录找到RTMPose模型权重文件: {checkpoint_path}")
                            checkpoint_file = checkpoint_path
                        else:
                            # 尝试从网络下载权重文件
                            try:
                                from labelme._automation.model_downloader import download_rtmpose_model
                                logger.info(
                                    f"模型权重不存在，尝试从网络下载: {self.model_name}")
                                checkpoint_path = download_rtmpose_model(
                                    self.model_name)
                                if checkpoint_path:
                                    logger.info(f"模型下载成功: {checkpoint_path}")
                                    checkpoint_file = checkpoint_path
                                else:
                                    # 如果下载失败，使用MMPose默认权重
                                    logger.warning(f"模型下载失败，使用MMPose默认权重")
                                    checkpoint_file = None
                            except Exception as e:
                                logger.warning(f"下载模型失败: {e}，使用MMPose默认权重")
                                checkpoint_file = None
                    else:
                        # 如果checkpoint_file为None，直接尝试下载
                        try:
                            from labelme._automation.model_downloader import download_rtmpose_model
                            logger.info(
                                f"没有找到预设权重文件，尝试从网络下载: {self.model_name}")
                            checkpoint_path = download_rtmpose_model(
                                self.model_name)
                            if checkpoint_path:
                                logger.info(f"模型下载成功: {checkpoint_path}")
                                checkpoint_file = checkpoint_path
                            else:
                                logger.warning(f"模型下载失败，使用MMPose默认权重")
                                checkpoint_file = None
                        except Exception as e:
                            logger.warning(f"下载模型失败: {e}，使用MMPose默认权重")
                            checkpoint_file = None
            except Exception as e:
                logger.warning(f"查找模型路径时出错: {e}，将使用MMPose默认权重")
                checkpoint_file = None

            # 初始化模型
            try:
                # 确保config_file和checkpoint_file是字符串
                if not isinstance(config_file, str):
                    logger.warning(f"配置文件类型错误: {type(config_file)}，将转换为字符串")
                    config_file = str(config_file)

                if checkpoint_file is not None and not isinstance(checkpoint_file, str):
                    logger.warning(
                        f"检查点文件类型错误: {type(checkpoint_file)}，将转换为字符串")
                    checkpoint_file = str(checkpoint_file)

                # 初始化模型
                model = init_model(
                    config_file,
                    checkpoint_file,
                    device=self.device
                )

                # 验证模型对象
                if model is None:
                    logger.error("init_model返回None，初始化失败")
                    return False

                if isinstance(model, (np.ndarray, list, tuple)):
                    logger.error(f"模型类型错误: {type(model)}，无法用于姿态估计")
                    return False

                # 测试是否可以访问模型属性
                try:
                    if hasattr(model, 'cfg'):
                        logger.debug(f"模型配置: {type(model.cfg)}")
                    else:
                        logger.warning("模型没有cfg属性")
                except Exception as attr_error:
                    logger.warning(f"无法访问模型属性: {attr_error}")

                # 保存模型实例
                self.rtmpose_model = model
                self.model = model

                # 初始化可视化器
                try:
                    if hasattr(model, 'cfg') and hasattr(model, 'dataset_meta'):
                        from mmpose.visualization import VISUALIZERS
                        self.visualizer = VISUALIZERS.build(
                            model.cfg.visualizer)
                        self.visualizer.set_dataset_meta(model.dataset_meta)
                except Exception as vis_error:
                    logger.warning(f"初始化可视化器失败: {vis_error}")

                logger.info(f"RTMPose模型初始化成功: {self.model_name}")
                logger.info(f"配置文件: {config_file}")
                logger.info(f"权重文件: {checkpoint_file}")
                self.is_rtmpose = True
                return True
            except Exception as e:
                logger.error(f"初始化模型失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                self.rtmpose_model = None
                self.is_rtmpose = False
                return False
        except Exception as e:
            logger.error(f"初始化RTMPose模型失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.is_rtmpose = False
            self.rtmpose_model = None
            return False


def get_shapes_from_poses(
    keypoints: List[List[List[float]]],
    scores: List[float] = None,
    start_group_id: int = 0,
    draw_skeleton: bool = True
) -> List[Dict]:
    """
    将姿态估计结果转换为标注形状列表

    Args:
        keypoints: 关键点列表 [N, K, 3] - (x, y, conf)
        scores: 人体检测的置信度列表 [N]
        start_group_id: 起始分组ID
        draw_skeleton: 是否创建骨架线段

    Returns:
        shapes: 形状列表
    """
    shapes = []

    # 日志输出关键点范围，方便调试
    if keypoints and len(keypoints) > 0 and len(keypoints[0]) > 0:
        all_points = np.array(
            [point[:2] for person in keypoints for point in person if point[2] > 0])
        if len(all_points) > 0:
            x_min, y_min = np.min(all_points, axis=0)
            x_max, y_max = np.max(all_points, axis=0)
            logger.debug(
                f"关键点范围: x({x_min:.1f}-{x_max:.1f}) y({y_min:.1f}-{y_max:.1f})")

    # 处理每个人体的姿态
    for i, kpts in enumerate(keypoints):
        score = scores[i] if scores and i < len(scores) else 1.0
        group_id = start_group_id + i

        # 将每个关键点转换为一个点形状
        for j, (x, y, conf) in enumerate(kpts):
            # 只添加置信度高于阈值的关键点
            if conf > 0:
                # 确保坐标是有效值 (防止NaN或无穷大)
                if not (np.isfinite(x) and np.isfinite(y)):
                    logger.warning(
                        f"跳过无效坐标: 关键点 {j} ({COCO_KEYPOINTS[j]}) - 坐标 ({x}, {y})")
                    continue

                # 获取关键点名称，不加前缀
                kpt_name = COCO_KEYPOINTS[j]

                # 根据置信度确定可见性状态
                # conf 值范围通常为 0-1
                # 将可见性转换为三种状态：0(图像中不存在)，1(存在但不可见)，2(存在且可见)
                visibility = 0  # 默认为0：图像中不存在
                if conf > 0.5:
                    visibility = 2  # 存在且可见
                elif conf > 0:
                    visibility = 1  # 存在但不可见

                # 添加描述，包含可见性信息
                description = f"{visibility}"

                # 创建形状字典 - 注意使用float类型以确保精确表示
                shape = {
                    "label": f"{kpt_name}",  # 不添加kpt_前缀
                    "points": [[float(x), float(y)]],
                    "group_id": group_id,
                    "shape_type": "point",
                    "flags": {},
                    "description": description  # 添加描述字段
                }
                shapes.append(shape)

        # 如果需要创建骨架线段
        if draw_skeleton:
            logger.debug(f"创建骨架线段，共{len(COCO_SKELETON)}条")
            # 创建骨架线段
            for p1_idx, p2_idx in COCO_SKELETON:
                p1 = kpts[p1_idx]
                p2 = kpts[p2_idx]

                # 确保两个关键点都可见且坐标有效
                if (p1[2] > 0 and p2[2] > 0 and
                    np.isfinite(p1[0]) and np.isfinite(p1[1]) and
                        np.isfinite(p2[0]) and np.isfinite(p2[1])):

                    # 获取两个关键点的名称
                    p1_name = COCO_KEYPOINTS[p1_idx]
                    p2_name = COCO_KEYPOINTS[p2_idx]

                    # 创建线段形状
                    shape = {
                        "label": f"limb_{p1_name}_{p2_name}",
                        "points": [[float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])]],
                        "group_id": group_id,
                        "shape_type": "line",
                        "flags": {}
                    }
                    shapes.append(shape)

    return shapes


def detect_poses(
    image: np.ndarray,
    model_name: str = None,
    device: str = None,
    conf_threshold: float = None,
    keypoint_threshold: float = None,
    advanced_params: dict = None,
    start_group_id: int = 0,
    draw_skeleton: bool = None
) -> List[Dict]:
    """
    检测图像中的人体姿态并返回形状列表

    Args:
        image: 输入图像
        model_name: 模型名称
        device: 运行设备
        conf_threshold: 置信度阈值
        keypoint_threshold: 关键点置信度阈值
        advanced_params: 高级参数字典
        start_group_id: 起始分组ID
        draw_skeleton: 是否绘制骨骼

    Returns:
        shapes: 形状列表
    """
    try:
        # 初始化姿态估计器
        estimator = PoseEstimator(
            model_name=model_name,
            device=device,
            conf_threshold=conf_threshold,
            keypoint_threshold=keypoint_threshold,
            advanced_params=advanced_params,
            draw_skeleton=draw_skeleton
        )

        # 检测图像中的姿态
        keypoints, scores = estimator.detect_poses(image)

        # 将姿态结果转换为形状列表
        shapes = get_shapes_from_poses(
            keypoints, scores, start_group_id, draw_skeleton)

        return shapes
    except Exception as e:
        logger.error(f"姿态估计过程中出错: {e}")
        return []


def estimate_poses(
    image: np.ndarray,
    model_name: str = None,
    conf_threshold: float = None,
    device: str = None,
    existing_person_boxes: List[List[float]] = None,
    existing_person_boxes_ids: List[int] = None,
    use_detection_results: bool = None,
    keypoint_threshold: float = None,
    advanced_params: dict = None,
    start_group_id: int = 0,
    draw_skeleton: bool = None
) -> List[Dict]:
    """
    检测图像中的人体姿态并返回形状列表（兼容旧API）

    Args:
        image: 输入图像
        model_name: 模型名称
        conf_threshold: 置信度阈值
        device: 运行设备
        existing_person_boxes: 已存在的人体框列表
        existing_person_boxes_ids: 已存在的人体框ID列表
        use_detection_results: 是否使用检测结果
        keypoint_threshold: 关键点置信度阈值
        advanced_params: 高级参数字典
        start_group_id: 起始分组ID
        draw_skeleton: 是否绘制骨骼

    Returns:
        shapes: 形状列表
    """
    try:
        # 日志记录图像信息和参数
        img_height, img_width = image.shape[:2]
        logger.info(
            f"开始进行姿态估计, 模型: {model_name}, 使用已有人体框: {use_detection_results}, 图像尺寸: {img_width}x{img_height}")

        # 检查并记录边界框信息
        if existing_person_boxes and len(existing_person_boxes) > 0:
            for i, box in enumerate(existing_person_boxes):
                if len(box) >= 4:
                    logger.debug(
                        f"边界框 {i}: x1={box[0]:.1f}, y1={box[1]:.1f}, x2={box[2]:.1f}, y2={box[3]:.1f}, 宽={box[2]-box[0]:.1f}, 高={box[3]-box[1]:.1f}")

        # 初始化姿态估计器
        estimator = PoseEstimator(
            model_name=model_name,
            device=device,
            conf_threshold=conf_threshold,
            keypoint_threshold=keypoint_threshold,
            advanced_params=advanced_params,
            draw_skeleton=draw_skeleton
        )

        keypoints = []
        scores = []

        # 如果提供了人体框并且用户选择使用已有框，从框中检测姿态
        if existing_person_boxes and len(existing_person_boxes) > 0 and (use_detection_results is True):
            logger.info(f"使用已有的 {len(existing_person_boxes)} 个人体框进行姿态估计")

            # 规范化边界框 - 确保坐标是有效值且在图像范围内
            valid_boxes = []
            for box in existing_person_boxes:
                if len(box) >= 4:
                    x1 = max(0, min(float(box[0]), img_width - 1))
                    y1 = max(0, min(float(box[1]), img_height - 1))
                    x2 = max(0, min(float(box[2]), img_width - 1))
                    y2 = max(0, min(float(box[3]), img_height - 1))

                    # 确保边界框有效
                    if x2 > x1 and y2 > y1:
                        valid_boxes.append([x1, y1, x2, y2])

            if valid_boxes:
                # 根据模型类型选择不同的处理方法
                if estimator.is_rtmpose:
                    logger.info("使用RTMPose从已有人体框进行姿态估计")
                    keypoints, scores = estimator._detect_rtmpose_from_boxes(
                        image, valid_boxes)
                else:
                    keypoints, scores = estimator.detect_poses_from_boxes(
                        image, valid_boxes)
            else:
                logger.warning("没有有效的边界框，将使用自动检测")
        else:
            # 如果没有选择使用已有框或没有提供人体框，使用自动检测
            logger.info("未启用使用已有框或未提供人体框，使用自动检测")
            if estimator.is_rtmpose:
                # 如果是RTMPose模型，优先使用RTMDet进行人体检测
                keypoints, scores = estimator._detect_rtmpose(image)
            else:
                keypoints, scores = estimator.detect_poses(image)

        # 验证并记录关键点信息
        if keypoints and len(keypoints) > 0:
            logger.info(f"检测到 {len(keypoints)} 个姿态")
            for i, person_kpts in enumerate(keypoints):
                valid_kpts = sum(1 for kpt in person_kpts if len(
                    kpt) > 2 and kpt[2] > 0)
                logger.debug(
                    f"姿态 {i}: 有效关键点数量 {valid_kpts}/{len(person_kpts)}")
        else:
            logger.warning("未检测到任何有效姿态")

        # 将姿态结果转换为形状列表
        group_id = start_group_id
        if existing_person_boxes_ids and len(existing_person_boxes_ids) > 0 and use_detection_results is True:
            group_id = existing_person_boxes_ids[0] if existing_person_boxes_ids[0] is not None else start_group_id

        shapes = get_shapes_from_poses(
            keypoints, scores, group_id, draw_skeleton)

        logger.info(f"姿态估计完成，共检测到 {len(keypoints)} 个姿态，生成 {len(shapes)} 个标注形状")
        return shapes
    except Exception as e:
        logger.error(f"姿态估计过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []
