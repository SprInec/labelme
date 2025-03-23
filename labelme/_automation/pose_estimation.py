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

# 尝试导入PyTorch依赖，如果不可用则提供错误信息
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("姿态估计依赖未安装，请安装torch")

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

        # 检查是否是RTMPose模型
        self.is_rtmpose = self.model_name.startswith("rtmpose")
        # 检查是否是KeypointRCNN模型
        self.is_keypointrcnn = self.model_name == "keypointrcnn_resnet50_fpn"

        # 如果不是RTMPose模型且不是KeypointRCNN模型，检查是否可以导入YOLOv7依赖
        if not self.is_rtmpose and not self.is_keypointrcnn:
            try:
                from labelme._automation.yolov7.models.experimental import attempt_load
                from labelme._automation.yolov7.utils.general import check_img_size, non_max_suppression_kpt
                from labelme._automation.yolov7.utils.torch_utils import select_device
                HAS_YOLOV7 = True
            except ImportError:
                HAS_YOLOV7 = False
                logger.warning("YOLOv7依赖未安装，自动切换到KeypointRCNN模型")
                self.model_name = "keypointrcnn_resnet50_fpn"
                self.is_keypointrcnn = True

        # 检查CUDA可用性
        if torch.cuda.is_available() and self.device == 'cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        logger.info(f"使用设备: {self.device}")

        # 加载模型
        self.model = self._load_model()

    def _load_model(self):
        """加载姿态估计模型"""
        try:
            # 判断是否是RTMPose模型
            if self.is_rtmpose:
                return self._load_rtmpose_model()
            elif self.is_keypointrcnn:
                return self._load_keypointrcnn_model()
            else:
                return self._load_yolov7_pose_model()
        except Exception as e:
            logger.error(f"加载姿态估计模型失败: {e}")
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
                from models.experimental import attempt_load
                from utils.torch_utils import select_device
                from utils.general import check_img_size
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
                            logger.warning(f"RTMDet-m模型下载失败，使用MMDet默认权重")
                            rtmdet_checkpoint_file = None
                    except Exception as e:
                        logger.warning(f"下载RTMDet-m模型失败: {e}，使用MMDet默认权重")
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
                    checkpoint_path = os.path.join(
                        checkpoint_dir, checkpoint_file)

                    if os.path.exists(checkpoint_path):
                        logger.info(f"在缓存目录找到RTMPose模型权重文件: {checkpoint_path}")
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
            model = init_model(
                config_file,
                checkpoint_file,
                device=self.device
            )

            # 存储可视化器
            self.visualizer = VISUALIZERS.build(model.cfg.visualizer)
            self.visualizer.set_dataset_meta(model.dataset_meta)

            # 确保有KeypointRCNN备用模型
            self._init_keypointrcnn_backup()

            logger.info(f"RTMPose模型加载成功: {self.model_name}")
            return model
        except Exception as e:
            logger.error(f"加载RTMPose模型失败: {e}")
            raise

    def _init_keypointrcnn_backup(self):
        """初始化KeypointRCNN备用模型"""
        try:
            import torchvision
            import torch
            import torchvision.models.detection as detection_models

            # 确保TORCH_HOME已设置为_automation/torch目录
            try:
                from labelme._automation.model_downloader import set_torch_home
                set_torch_home()
            except Exception as e:
                logger.warning(f"设置TORCH_HOME失败: {e}")

            # 尝试预下载模型（如果需要）
            try:
                from labelme._automation.model_downloader import download_torchvision_model
                download_torchvision_model("keypointrcnn_resnet50_fpn")
            except Exception as e:
                logger.warning(f"预下载模型失败: {e}，将在创建模型时自动下载")

            # 尝试使用新接口加载预训练的KeypointRCNN模型
            try:
                keypointrcnn_model = detection_models.keypointrcnn_resnet50_fpn(
                    weights="DEFAULT",
                    progress=True,
                    num_keypoints=17,
                    box_score_thresh=self.conf_threshold
                )
            except TypeError as e:
                logger.warning(
                    f"使用weights参数加载模型失败: {e}，尝试使用旧版接口 (pretrained=True)")
                # 尝试使用旧接口
                keypointrcnn_model = detection_models.keypointrcnn_resnet50_fpn(
                    pretrained=True,
                    progress=True,
                    num_keypoints=17,
                    box_score_thresh=self.conf_threshold
                )

            # 设置为评估模式
            keypointrcnn_model.eval()

            # 如果使用CUDA且可用
            if self.device == 'cuda' and torch.cuda.is_available():
                keypointrcnn_model = keypointrcnn_model.to('cuda')

            logger.info(f"KeypointRCNN备用模型加载成功")
            self.keypointrcnn_model = keypointrcnn_model
            return keypointrcnn_model
        except Exception as e:
            logger.error(f"加载KeypointRCNN备用模型失败: {e}")
            self.keypointrcnn_model = None
            return None

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
                return self._detect_keypointrcnn_backup(image)

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
                logger.warning("未找到检测模型，尝试使用备用方法")
                return self._detect_keypointrcnn_backup(image)

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
                            f"未检测到置信度大于{self.conf_threshold}的人体，尝试使用备用方法")
                        return self._detect_keypointrcnn_backup(image)

                    valid_bboxes = human_bboxes[valid_indices]
                    valid_scores = human_scores[valid_indices]
                else:
                    logger.warning("检测结果中未找到边界框信息，尝试使用备用方法")
                    return self._detect_keypointrcnn_backup(image)
            except Exception as e:
                logger.error(f"目标检测失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return self._detect_keypointrcnn_backup(image)

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
                logger.warning("RTMPose未检测到有效姿态，尝试使用备用方法")
                return self._detect_keypointrcnn_backup(image)
        except Exception as e:
            logger.error(f"RTMPose检测失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._detect_keypointrcnn_backup(image)

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
                adjusted_keypoints = keypoints.copy()
                adjusted_keypoints[:, 0] += boxes[:, 0][:, None]
                adjusted_keypoints[:, 1] += boxes[:, 1][:, None]

                # 如果检测分数高于阈值
                if scores >= self.conf_threshold:
                    # 过滤低置信度的关键点
                    mask = adjusted_keypoints[:, 2] < self.keypoint_threshold
                    adjusted_keypoints[mask, 2] = 0

                    keypoints_list.append(adjusted_keypoints.tolist())
                    scores_list.append(float(scores[mask][0]))

            return keypoints_list, scores_list

        except Exception as e:
            logger.error(f"KeypointRCNN检测失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [], []

    def detect_poses_from_boxes(self, image: np.ndarray, boxes: List[List[float]]) -> Tuple[List[List[List[float]]], List[float]]:
        """
        从边界框中检测人体姿态

        Args:
            image: 输入图像 (BGR格式)
            boxes: 边界框列表 [N, 4] - (x1, y1, x2, y2)

        Returns:
            keypoints: 关键点列表 [N, K, 3] - (x, y, conf)
            scores: 人体检测的置信度列表 [N]
        """
        # 判断是否使用RTMPose模型
        if self.is_rtmpose:
            return self._detect_rtmpose_from_boxes(image, boxes)
        elif self.is_keypointrcnn:
            return self._detect_keypointrcnn_from_boxes(image, boxes)
        else:
            # YOLOv7 Pose也支持从边界框中检测姿态
            return self._detect_yolov7_pose_from_boxes(image, boxes)

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

    def _detect_rtmpose_from_boxes(self, image: np.ndarray, boxes: List[List[float]]) -> Tuple[List[List[List[float]]], List[float]]:
        """
        使用RTMPose从已有的人体框中检测关键点

        Args:
            image: 输入图像
            boxes: 边界框列表 [[x1, y1, x2, y2], ...]

        Returns:
            keypoints: 关键点列表 [N, K, 3] - (x, y, conf)
            scores: 人体检测的置信度列表 [N]
        """
        try:
            # 导入必要的库
            try:
                from mmpose.apis import inference_topdown
            except ImportError as e:
                logger.error(f"导入mmpose库失败: {e}")
                return self._detect_keypointrcnn_from_boxes(image, boxes)

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

            # 获取模型配置中的输入尺寸
            model_input_size = None
            try:
                if hasattr(self.model, 'cfg') and self.model.cfg:
                    if 'data_preprocessor' in self.model.cfg and 'input_size' in self.model.cfg.data_preprocessor:
                        model_input_size = self.model.cfg.data_preprocessor.input_size
                if not model_input_size and hasattr(self.model, 'data_preprocessor'):
                    if hasattr(self.model.data_preprocessor, 'input_size'):
                        model_input_size = self.model.data_preprocessor.input_size
            except Exception as e:
                logger.warning(f"获取模型输入尺寸失败: {e}")

            if not model_input_size:
                logger.warning("未找到模型输入尺寸，使用默认值(192, 256)")
                model_input_size = (192, 256)  # 默认尺寸 (h, w)

            logger.debug(f"模型输入尺寸: {model_input_size}")

            # 将边界框转换为numpy数组
            bbox_xyxy = np.array(boxes)

            # 确保边界框格式正确
            if bbox_xyxy.shape[1] != 4:
                logger.error(
                    f"边界框格式错误，预期为[x1, y1, x2, y2]，实际为{bbox_xyxy.shape[1]}维")
                return [], []

            # 记录处理前边界框范围
            x_min, y_min = np.min(bbox_xyxy[:, :2], axis=0)
            x_max, y_max = np.max(bbox_xyxy[:, 2:4], axis=0)
            logger.debug(
                f"处理前边界框范围: x({x_min:.1f}-{x_max:.1f}) y({y_min:.1f}-{y_max:.1f})")

            # 使用inference_topdown进行预测
            results = []
            try:
                results = inference_topdown(
                    self.model, image, bbox_xyxy, bbox_format='xyxy')
                logger.debug(f"成功调用inference_topdown，获得{len(results)}个预测结果")
            except Exception as e:
                logger.error(f"inference_topdown调用失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # 尝试使用备用方法
                logger.info("尝试使用KeypointRCNN备用方法")
                return self._detect_keypointrcnn_from_boxes(image, boxes)

            # 处理预测结果
            keypoints = []
            scores = []

            # 确保结果不为空
            if not results or len(results) == 0:
                logger.warning("没有得到预测结果")
                # 尝试使用备用方法
                logger.info("尝试使用KeypointRCNN备用方法")
                return self._detect_keypointrcnn_from_boxes(image, boxes)

            logger.debug(f"预测结果类型: {type(results[0])}")

            # 处理不同类型的结果
            for i, result in enumerate(results):
                try:
                    # 获取关键点和置信度
                    kpts = None
                    kpt_scores = None
                    bbox_score = 1.0

                    # 处理PoseDataSample类型的结果
                    if hasattr(result, 'pred_instances'):
                        logger.debug(f"处理PoseDataSample类型的结果: {i}")
                        pred_instances = result.pred_instances

                        if hasattr(pred_instances, 'keypoints'):
                            kpts_tensor = pred_instances.keypoints
                            # 如果是tensor，转换为numpy
                            if not isinstance(kpts_tensor, np.ndarray):
                                kpts = kpts_tensor.cpu().numpy()
                            else:
                                kpts = kpts_tensor

                            # 获取置信度
                            if hasattr(pred_instances, 'keypoint_scores'):
                                scores_tensor = pred_instances.keypoint_scores
                                if not isinstance(scores_tensor, np.ndarray):
                                    kpt_scores = scores_tensor.cpu().numpy()
                                else:
                                    kpt_scores = scores_tensor

                            # 获取边界框置信度
                            if hasattr(pred_instances, 'bboxes_scores'):
                                bbox_score = float(pred_instances.bboxes_scores.item() if hasattr(
                                    pred_instances.bboxes_scores, 'item') else pred_instances.bboxes_scores)

                    # 如果不是PoseDataSample或获取失败，尝试直接处理结果
                    if kpts is None:
                        logger.debug(f"直接处理结果: {i}")
                        # 尝试提取关键点数据
                        if isinstance(result, np.ndarray) and result.ndim == 3:
                            # 假设结果是[k, 3]形状的ndarray，表示[x, y, score]
                            kpts = result
                        else:
                            # 尝试从字典或其他对象中提取
                            if hasattr(result, 'keypoints'):
                                kpts = result.keypoints
                            elif isinstance(result, dict) and 'keypoints' in result:
                                kpts = result['keypoints']

                    if kpts is None:
                        logger.warning(f"从结果{i}中无法提取关键点数据")
                        continue

                    # 处理关键点数据
                    if len(kpts.shape) == 3 and kpts.shape[0] == 1:
                        # [1, K, 3] -> [K, 3]
                        kpts = kpts[0]

                    # 确保关键点格式为 [K, 3]
                    if len(kpts.shape) != 2 or kpts.shape[1] != 3:
                        logger.warning(f"关键点格式错误，预期为[K, 3]，实际为{kpts.shape}")
                        continue

                    # 获取当前人体框
                    if i < len(bbox_xyxy):
                        bbox = bbox_xyxy[i]
                        # 计算框的宽度和高度
                        box_width = bbox[2] - bbox[0]
                        box_height = bbox[3] - bbox[1]

                        # 日志记录边界框信息
                        logger.debug(f"边界框 {i}: x1={bbox[0]:.1f}, y1={bbox[1]:.1f}, x2={bbox[2]:.1f}, y2={bbox[3]:.1f}, "
                                     f"宽={box_width:.1f}, 高={box_height:.1f}")

                        # 获取模型输入尺寸
                        model_height, model_width = model_input_size

                        # 记录关键点范围(处理前)
                        valid_kpts = kpts[kpts[:, 2] > 0]
                        if len(valid_kpts) > 0:
                            x_min, y_min = np.min(valid_kpts[:, :2], axis=0)
                            x_max, y_max = np.max(valid_kpts[:, :2], axis=0)
                            logger.debug(
                                f"原始关键点范围: x({x_min:.1f}-{x_max:.1f}) y({y_min:.1f}-{y_max:.1f})")

                        # 关键点坐标转换：从模型输入尺寸映射回原始图像坐标
                        # 对每个关键点进行转换
                        transformed_kpts = kpts.copy()
                        # kpts[:, 0]是x坐标(列)，范围是[0, model_width]
                        # kpts[:, 1]是y坐标(行)，范围是[0, model_height]

                        # 首先，将关键点坐标从[0-1]标准化范围转换为像素范围
                        if np.max(kpts[:, 0]) <= 1.0 and np.max(kpts[:, 1]) <= 1.0:
                            logger.debug("检测到关键点坐标为标准化值[0-1]，转换为像素坐标")
                            transformed_kpts[:, 0] *= model_width
                            transformed_kpts[:, 1] *= model_height

                        # 计算从模型输入空间到边界框空间的缩放因子
                        scale_x = box_width / model_width
                        scale_y = box_height / model_height

                        # 应用缩放和偏移
                        transformed_kpts[:, 0] = transformed_kpts[:,
                                                                  0] * scale_x + bbox[0]
                        transformed_kpts[:, 1] = transformed_kpts[:,
                                                                  1] * scale_y + bbox[1]

                        # 确保关键点在图像范围内
                        transformed_kpts[:, 0] = np.clip(
                            transformed_kpts[:, 0], 0, img_width - 1)
                        transformed_kpts[:, 1] = np.clip(
                            transformed_kpts[:, 1], 0, img_height - 1)

                        # 记录关键点范围(处理后)
                        valid_kpts = transformed_kpts[transformed_kpts[:, 2] > 0]
                        if len(valid_kpts) > 0:
                            x_min, y_min = np.min(valid_kpts[:, :2], axis=0)
                            x_max, y_max = np.max(valid_kpts[:, :2], axis=0)
                            logger.debug(
                                f"转换后关键点范围: x({x_min:.1f}-{x_max:.1f}) y({y_min:.1f}-{y_max:.1f})")

                        # 应用可见性和置信度阈值
                        final_keypoints = []
                        for j, (x, y, conf) in enumerate(transformed_kpts):
                            # 使用kpt_scores如果可用，否则使用原始置信度
                            score = kpt_scores[j] if kpt_scores is not None and j < len(
                                kpt_scores) else conf

                            # 应用阈值
                            if score >= self.keypoint_threshold:
                                final_keypoints.append(
                                    [float(x), float(y), float(score)])
                            else:
                                # 置信度不足的点设为不可见
                                final_keypoints.append(
                                    [float(x), float(y), 0.0])

                        # 添加关键点和置信度
                        keypoints.append(final_keypoints)
                        scores.append(float(bbox_score))

                        # 日志记录
                        valid_count = sum(
                            1 for kp in final_keypoints if kp[2] > 0)
                        logger.debug(
                            f"处理完成人体{i}，有效关键点：{valid_count}/{len(final_keypoints)}")
                    else:
                        logger.warning(f"结果索引{i}超出边界框数量{len(bbox_xyxy)}")
                except Exception as e:
                    logger.error(f"处理结果{i}时出错: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue

            # 返回结果
            if len(keypoints) > 0:
                logger.info(f"RTMPose从框检测到 {len(keypoints)} 个姿态")
                return keypoints, scores
            else:
                logger.warning("RTMPose未检测到有效姿态，尝试使用备用方法")
                return self._detect_keypointrcnn_from_boxes(image, boxes)
        except Exception as e:
            logger.error(f"RTMPose从边界框检测失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._detect_keypointrcnn_from_boxes(image, boxes)

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
        """使用KeypointRCNN从给定的边界框中检测姿态"""
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
