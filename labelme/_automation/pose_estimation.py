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

            logger.info(f"RTMPose模型加载成功: {self.model_name}")
            return model
        except Exception as e:
            logger.error(f"加载RTMPose模型失败: {e}")
            raise

    def _load_keypointrcnn_model(self):
        """加载KeypointRCNN模型"""
        try:
            import torchvision
            import torch
            import torchvision.models.detection as detection_models

            # 尝试预下载模型（如果需要）
            try:
                from labelme._automation.model_downloader import download_torchvision_model
                download_torchvision_model("keypointrcnn_resnet50_fpn")
            except Exception as e:
                logger.warning(f"预下载模型失败: {e}，将在创建模型时自动下载")

            # 尝试使用新接口加载预训练的KeypointRCNN模型
            try:
                model = detection_models.keypointrcnn_resnet50_fpn(
                    weights="DEFAULT",
                    progress=True,
                    num_keypoints=17,
                    box_score_thresh=self.conf_threshold
                )
            except TypeError as e:
                logger.warning(
                    f"使用weights参数加载模型失败: {e}，尝试使用旧版接口 (pretrained=True)")
                # 尝试使用旧接口
                model = detection_models.keypointrcnn_resnet50_fpn(
                    pretrained=True,
                    progress=True,
                    num_keypoints=17,
                    box_score_thresh=self.conf_threshold
                )

            # 设置为评估模式
            model.eval()

            # 如果使用CUDA且可用
            if self.device == 'cuda' and torch.cuda.is_available():
                model = model.to('cuda')

            logger.info(f"KeypointRCNN模型加载成功: {self.model_name}")
            return model
        except Exception as e:
            logger.error(f"加载KeypointRCNN模型失败: {e}")
            raise

    def detect_poses(self, image: np.ndarray) -> Tuple[List[List[List[float]]], List[float]]:
        """
        检测图像中的人体姿态关键点

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            keypoints: 关键点列表 [N, K, 3] - (x, y, conf)
            scores: 人体检测的置信度列表 [N]
        """
        logger.debug(f"使用模型 {self.model_name} 进行姿态检测")

        # 使用对应的检测方法
        if self.model_name.startswith("rtmpose"):
            return self._detect_rtmpose(image)
        elif self.model_name == "yolov7_w6_pose":
            return self._detect_yolov7_pose(image)
        elif self.model_name == "keypointrcnn_resnet50_fpn":
            return self._detect_keypointrcnn(image)
        else:
            logger.warning(f"未知的姿态估计模型: {self.model_name}，使用KeypointRCNN")
            return self._detect_keypointrcnn(image)

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

    def _detect_rtmpose(self, image: np.ndarray) -> Tuple[List[List[List[float]]], List[float]]:
        """使用RTMPose模型检测关键点"""
        import numpy as np
        from mmpose.structures import merge_data_samples
        from mmpose.apis import inference_topdown
        from mmengine.registry import init_default_scope
        import torch

        # 确保正确的默认作用域
        init_default_scope('mmpose')

        # 记录开始时间
        t_start = time.time()

        # 准备结果列表
        keypoints_list = []
        scores_list = []

        # 获取高级参数
        max_poses = self.advanced_params.get("max_poses", 20)
        min_keypoints = self.advanced_params.get("min_keypoints", 5)

        try:
            # 确保图像是RGB格式
            if image.shape[2] == 4:  # 如果有alpha通道
                image = image[:, :, :3]

            # 从BGR转换为RGB (如果输入是BGR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 构建默认的人体检测框，覆盖整个图像
            image_height, image_width = image.shape[:2]

            # 创建检测结果列表并确保bbox不为None
            try:
                # 适配mmpose API格式要求
                # 检查mmpose版本，调整格式
                import mmpose
                mmpose_version = mmpose.__version__
                logger.debug(f"当前mmpose版本: {mmpose_version}")

                # 适配不同版本的mmpose API格式
                if mmpose_version.startswith('0.'):
                    # 0.x版本格式
                    bbox = np.array(
                        [0, 0, image_width, image_height, 1.0], dtype=np.float32)
                    det_results = [{'bbox': bbox}]
                else:
                    # 根据mmpose 1.3.2版本的实际API需求定制格式
                    # 查看mmpose/apis/inference.py的inference_topdown函数

                    # 1. 创建数据样本列表
                    from mmpose.structures.bbox import bbox_xywh2xyxy, bbox_xyxy2xywh

                    # 尝试直接使用数据样本格式 (适用于1.3.x版本)
                    try:
                        # 为整张图像创建一个边界框
                        # 注意：mmpose需要的格式可能是xywh或xyxy，我们两种都尝试

                        # 方法1: 使用PoseDataSample格式
                        from mmpose.structures import PoseDataSample
                        data_sample = PoseDataSample()

                        # 设置图像信息
                        data_sample.set_metainfo({
                            'img_shape': image_rgb.shape[:2],
                            'img_id': 0
                        })

                        # 创建边界框，使用xyxy格式 (x1, y1, x2, y2, score)
                        xyxy_bbox = np.array(
                            [0, 0, image_width, image_height, 1.0], dtype=np.float32)

                        # 转换为xywh格式 (x, y, w, h, score) - mmpose内部通常使用这种格式
                        x, y, w, h = xyxy_bbox[0], xyxy_bbox[1], xyxy_bbox[2] - \
                            xyxy_bbox[0], xyxy_bbox[3]-xyxy_bbox[1]
                        xywh_bbox = np.array(
                            [x, y, w, h, 1.0], dtype=np.float32)

                        # 设置预测实例
                        data_sample.pred_instances = {
                            'bboxes': torch.tensor([xyxy_bbox[:4]]).float(),
                            'scores': torch.tensor([xyxy_bbox[4]]).float(),
                        }

                        # 创建带有数据样本的推理列表
                        logger.debug(f"使用PoseDataSample格式调用inference_topdown")
                        pose_results = inference_topdown(
                            self.model, image_rgb, [data_sample])
                        logger.debug(f"姿态检测成功，结果长度: {len(pose_results)}")

                        # 处理返回的结果
                        # 检查pose_results是否为None或空列表
                        if pose_results is None:
                            logger.error("pose_results为None，无法处理")
                            raise ValueError("pose_results为None")

                        if len(pose_results) == 0:
                            logger.warning("pose_results为空列表，未检测到姿态")
                            raise ValueError("pose_results为空列表")

                        # 如果有多个结果，合并结果
                        pose_result = merge_data_samples(pose_results)

                        # 处理预测实例
                        pred_instances = pose_result.pred_instances

                        logger.debug(
                            f"pred_instances类型: {type(pred_instances)}, 属性: {dir(pred_instances)}")

                        if len(pred_instances) > 0:
                            # 获取关键点和分数
                            keypoints = pred_instances.keypoints.cpu().numpy()
                            keypoint_scores = pred_instances.keypoint_scores.cpu().numpy()

                            # 如果有分数，则使用分数；否则，使用默认分数
                            if hasattr(pred_instances, 'scores'):
                                instance_scores = pred_instances.scores.cpu().numpy()
                            else:
                                instance_scores = np.ones(len(keypoints))

                            # 处理结果，构建关键点列表
                            for i in range(len(keypoints)):
                                kpts = np.zeros((keypoints.shape[1], 3))
                                kpts[:, :2] = keypoints[i]
                                kpts[:, 2] = keypoint_scores[i]

                                # 计算可见关键点数量
                                visible_keypoints = sum(
                                    1 for _, _, conf in kpts if conf >= self.keypoint_threshold)

                                # 过滤掉可见关键点数量少于阈值的姿态
                                if visible_keypoints >= min_keypoints and instance_scores[i] >= self.conf_threshold:
                                    keypoints_list.append(kpts.tolist())
                                    scores_list.append(
                                        float(instance_scores[i]))

                                    # 如果达到最大姿态数量，停止添加
                                    if len(keypoints_list) >= max_poses:
                                        break

                            logger.info(f"RTMPose检测找到{len(keypoints_list)}个姿态")
                            return keypoints_list, scores_list

                    except Exception as ds_error:
                        logger.warning(
                            f"使用PoseDataSample格式失败: {str(ds_error)}")

                        # 方法2: 使用字典格式 - 尝试多种边界框格式
                        try:
                            # 使用底层API
                            logger.debug("尝试使用底层API调用inference_topdown")
                            # 检查源代码中inference_topdown的实现

                            # 使用dict列表，确保每个dict包含img_path、img_id和img_shape
                            bbox_xywh = np.array(
                                [0, 0, image_width, image_height], dtype=np.float32)
                            data_info = {
                                'img': image_rgb,  # 提供图像
                                'img_shape': image_rgb.shape[:2],  # 提供形状
                                'img_id': 0,  # 提供ID
                                'bbox': bbox_xywh[None],  # 确保是[1,4]形状的数组
                                'bbox_score': np.array([1.0], dtype=np.float32),
                                'bbox_format': 'xywh'  # 指定格式
                            }

                            # 直接调用API
                            from mmpose.apis.inference import _inference_single_pose_model
                            pose_results = _inference_single_pose_model(
                                self.model, data_info)

                            # 处理API返回结果
                            if pose_results is not None and hasattr(pose_results, 'pred_instances'):
                                pred_instances = pose_results.pred_instances

                                if len(pred_instances) > 0:
                                    # 获取关键点和分数
                                    keypoints = pred_instances.keypoints.cpu().numpy()
                                    keypoint_scores = pred_instances.keypoint_scores.cpu().numpy()

                                    # 如果有分数，则使用分数；否则，使用默认分数
                                    if hasattr(pred_instances, 'scores'):
                                        instance_scores = pred_instances.scores.cpu().numpy()
                                    else:
                                        instance_scores = np.ones(
                                            len(keypoints))

                                    # 处理结果
                                    for i in range(len(keypoints)):
                                        kpts = np.zeros(
                                            (keypoints.shape[1], 3))
                                        kpts[:, :2] = keypoints[i]
                                        kpts[:, 2] = keypoint_scores[i]

                                        visible_keypoints = sum(
                                            1 for _, _, conf in kpts if conf >= self.keypoint_threshold)

                                        if visible_keypoints >= min_keypoints and instance_scores[i] >= self.conf_threshold:
                                            keypoints_list.append(
                                                kpts.tolist())
                                            scores_list.append(
                                                float(instance_scores[i]))

                                            if len(keypoints_list) >= max_poses:
                                                break

                                    logger.info(
                                        f"底层API检测找到{len(keypoints_list)}个姿态")
                                    return keypoints_list, scores_list

                        except Exception as api_error:
                            logger.warning(f"使用底层API调用失败: {str(api_error)}")

                            # 方法3: 直接调用模型
                            try:
                                # 直接使用模型前向传播
                                logger.debug("尝试直接调用模型前向传播")
                                from mmpose.structures import PoseDataSample

                                # 创建输入数据
                                input_tensor = torch.from_numpy(image_rgb.transpose(
                                    2, 0, 1)).float().div(255.0).unsqueeze(0).to(self.device)
                                data_sample = PoseDataSample()
                                data_sample.set_metainfo({
                                    'img_shape': image_rgb.shape[:2],
                                    'img_id': 0
                                })

                                # 转换为标准输入格式
                                result = self.model.forward(
                                    input_tensor, [data_sample], mode='tensor')

                                # 处理forward返回的结果
                                if result is not None and isinstance(result, list) and len(result) > 0:
                                    # 假设返回的是预测实例列表
                                    for pred_sample in result:
                                        if hasattr(pred_sample, 'pred_instances'):
                                            pred_instances = pred_sample.pred_instances

                                            if len(pred_instances) > 0:
                                                # 获取关键点和分数
                                                keypoints = pred_instances.keypoints.cpu().numpy()
                                                keypoint_scores = pred_instances.keypoint_scores.cpu().numpy()

                                                # 分数处理
                                                if hasattr(pred_instances, 'scores'):
                                                    instance_scores = pred_instances.scores.cpu().numpy()
                                                else:
                                                    instance_scores = np.ones(
                                                        len(keypoints))

                                                # 处理关键点
                                                for i in range(len(keypoints)):
                                                    kpts = np.zeros(
                                                        (keypoints.shape[1], 3))
                                                    kpts[:, :2] = keypoints[i]
                                                    kpts[:, 2] = keypoint_scores[i]

                                                    visible_keypoints = sum(
                                                        1 for _, _, conf in kpts if conf >= self.keypoint_threshold)

                                                    if visible_keypoints >= min_keypoints and instance_scores[i] >= self.conf_threshold:
                                                        keypoints_list.append(
                                                            kpts.tolist())
                                                        scores_list.append(
                                                            float(instance_scores[i]))

                                                        if len(keypoints_list) >= max_poses:
                                                            break

                                logger.info(
                                    f"直接调用方法找到{len(keypoints_list)}个姿态")
                                return keypoints_list, scores_list

                            except Exception as e:
                                logger.error(f"所有尝试都失败: {str(e)}")
                                # 使用备用模型
                                logger.info("所有RTMPose方法都失败，切换到备用模型")
                                raise

            except Exception as e:
                logger.error(f"姿态检测失败，尝试使用备用方法: {str(e)}")
                import traceback
                logger.error(f"异常详情: {traceback.format_exc()}")

                # 尝试使用备用模型方法 - 如果RTMPose失败，使用KeypointRCNN
                try:
                    logger.info("RTMPose失败，尝试使用KeypointRCNN备用方法")

                    # 创建独立的KeypointRCNN检测器实例，而不是复用self.model
                    import torchvision
                    import torch.nn as nn

                    # 检查是否已经有KeypointRCNN模型
                    if not hasattr(self, 'keypointrcnn_model') or self.keypointrcnn_model is None:
                        logger.info("创建新的KeypointRCNN模型")
                        import torchvision.models.detection as detection_models

                        # 使用兼容性更好的旧式接口
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

                        self.keypointrcnn_model = keypointrcnn_model

                    # 使用KeypointRCNN进行检测
                    import torchvision.transforms.functional as TF

                    # 确保图像是RGB格式
                    img_rgb = cv2.cvtColor(
                        image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image

                    # 转换为PyTorch张量
                    img_tensor = TF.to_tensor(img_rgb)

                    # 确保在正确的设备上
                    img_tensor = img_tensor.to(self.device)

                    # 使用模型进行推理
                    with torch.no_grad():
                        predictions = self.keypointrcnn_model([img_tensor])

                    # 处理关键点结果
                    keypoints_list = []
                    scores_list = []

                    if len(predictions) > 0:
                        pred = predictions[0]

                        if len(pred['keypoints']) > 0:
                            # 获取人体检测框和分数
                            boxes = pred['boxes'].cpu().numpy()
                            scores = pred['scores'].cpu().numpy()
                            keypoints_tensor = pred['keypoints'].cpu().numpy()

                            # 过滤低置信度的检测结果
                            mask = scores >= self.conf_threshold
                            if np.any(mask):  # 确保至少有一个有效检测
                                boxes = boxes[mask]
                                scores = scores[mask]
                                keypoints_tensor = keypoints_tensor[mask]

                                # 处理关键点
                                for i, kpts in enumerate(keypoints_tensor):
                                    # 转换keypoints格式：[x, y, visibility] -> [x, y, confidence]
                                    kpts_list = []
                                    for kpt in kpts:
                                        x, y, vis = kpt
                                        # KeypointRCNN返回的是可见性（0:不可见，1:可见但被遮挡，2:完全可见）
                                        # 我们需要将其转换为置信度（0-1之间的值）
                                        conf = vis / 2.0 if vis > 0 else 0.0
                                        kpts_list.append(
                                            [float(x), float(y), float(conf)])

                                    # 计算可见关键点数量
                                    visible_keypoints = sum(
                                        1 for _, _, conf in kpts_list if conf >= self.keypoint_threshold)

                                    # 过滤掉可见关键点数量少于阈值的姿态
                                    if visible_keypoints >= min_keypoints:
                                        keypoints_list.append(kpts_list)
                                        scores_list.append(float(scores[i]))

                                        # 如果达到最大姿态数量，停止添加
                                        if len(keypoints_list) >= max_poses:
                                            break

                    logger.info(f"KeypointRCNN检测找到{len(keypoints_list)}个姿态")
                    return keypoints_list, scores_list

                except Exception as backup_e:
                    logger.error(f"备用方法也失败: {str(backup_e)}")
                    logger.error(traceback.format_exc())
                    return [], []

        except Exception as e:
            logger.error(f"RTMPose检测失败: {str(e)}")
            import traceback
            logger.error(f"异常详情: {traceback.format_exc()}")
            return [], []

    def _detect_keypointrcnn(self, image: np.ndarray) -> Tuple[List[List[List[float]]], List[float]]:
        """使用KeypointRCNN模型检测关键点"""
        import torch
        import torch.nn.functional as F
        import torchvision.transforms.functional as TF

        # 记录开始时间
        t_start = time.time()

        # 确保图像是RGB格式
        if image.shape[2] == 4:  # 如果有alpha通道
            image = image[:, :, :3]

        # 从BGR转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 转换为PyTorch张量
        img_tensor = TF.to_tensor(image_rgb)

        # 确保在正确的设备上
        img_tensor = img_tensor.to(self.device)

        # 使用模型进行推理
        with torch.no_grad():
            predictions = self.model([img_tensor])

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

            # 获取高级参数
            max_poses = self.advanced_params.get("max_poses", 20)
            min_keypoints = self.advanced_params.get("min_keypoints", 5)

            # 处理关键点
            for i, kpts in enumerate(keypoints):
                # 转换keypoints格式：[x, y, visibility] -> [x, y, confidence]
                kpts_list = []
                for kpt in kpts:
                    x, y, vis = kpt
                    # KeypointRCNN返回的是可见性（0:不可见，1:可见但被遮挡，2:完全可见）
                    # 我们需要将其转换为置信度（0-1之间的值）
                    conf = vis / 2.0 if vis > 0 else 0.0
                    kpts_list.append([float(x), float(y), float(conf)])

                # 计算可见关键点数量
                visible_keypoints = sum(
                    1 for _, _, conf in kpts_list if conf >= self.keypoint_threshold)

                # 过滤掉可见关键点数量少于阈值的姿态
                if visible_keypoints >= min_keypoints:
                    keypoints_list.append(kpts_list)
                    scores_list.append(float(scores[i]))

                    # 如果达到最大姿态数量，停止添加
                    if len(keypoints_list) >= max_poses:
                        break

        logger.debug(
            f"KeypointRCNN检测完成: 找到 {len(keypoints_list)} 个姿态, 耗时 {time.time() - t_start:.3f} [s]")

        return keypoints_list, scores_list

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
        """使用RTMPose从给定的边界框中检测姿态"""
        try:
            from mmpose.apis import inference_topdown
            from mmpose.structures import merge_data_samples
            from mmengine.registry import init_default_scope
            import numpy as np

            # 确保正确的默认作用域
            init_default_scope('mmpose')

            # 记录开始时间
            t_start = time.time()

            # 确保图像是RGB格式
            if image.shape[2] == 4:  # 如果有alpha通道
                image = image[:, :, :3]

            # 从BGR转换为RGB (如果输入是BGR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 创建检测结果列表
            det_results = []
            for box in boxes:
                x1, y1, x2, y2 = box
                # 添加置信度为1.0
                det_results.append({'bbox': [x1, y1, x2, y2, 1.0]})

            # 如果没有检测框，返回空结果
            if not det_results:
                return [], []

            # 使用TopDown方法进行姿态估计
            pose_results = inference_topdown(
                self.model, image_rgb, det_results)

            # 提取关键点和分数
            keypoints_list = []
            scores_list = []

            if len(pose_results) > 0:
                # 合并结果
                pose_result = merge_data_samples(pose_results)

                # 处理预测实例
                pred_instances = pose_result.pred_instances

                if len(pred_instances) > 0:
                    # 获取关键点和分数
                    # [N, K, 2]
                    keypoints = pred_instances.keypoints.cpu().numpy()
                    # [N, K]
                    keypoint_scores = pred_instances.keypoint_scores.cpu().numpy()

                    # 如果有分数，则使用分数；否则，使用默认分数
                    if hasattr(pred_instances, 'scores'):
                        # [N]
                        instance_scores = pred_instances.scores.cpu().numpy()
                    else:
                        instance_scores = np.ones(len(keypoints))

                    # 获取高级参数
                    min_keypoints = self.advanced_params.get(
                        "min_keypoints", 5)

                    # 处理结果
                    for i in range(len(keypoints)):
                        kpts = np.zeros((keypoints.shape[1], 3))
                        kpts[:, :2] = keypoints[i]
                        kpts[:, 2] = keypoint_scores[i]

                        # 计算可见关键点数量
                        visible_keypoints = sum(
                            1 for _, _, conf in kpts if conf >= self.keypoint_threshold)

                        # 过滤掉可见关键点数量少于阈值的姿态
                        if visible_keypoints >= min_keypoints and instance_scores[i] >= self.conf_threshold:
                            keypoints_list.append(kpts.tolist())
                            scores_list.append(float(instance_scores[i]))

            logger.debug(
                f"RTMPose从边界框检测完成: 找到 {len(keypoints_list)} 个姿态, 耗时 {time.time() - t_start:.3f} [s]")

            return keypoints_list, scores_list

        except Exception as e:
            logger.error(f"RTMPose从边界框检测失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [], []

    def _detect_keypointrcnn_from_boxes(self, image: np.ndarray, boxes: List[List[float]]) -> Tuple[List[List[List[float]]], List[float]]:
        """使用KeypointRCNN从给定的边界框中检测姿态"""
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

                # 转换为RGB格式
                cropped_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

                # 转换为PyTorch张量
                cropped_tensor = torch.from_numpy(cropped_rgb.transpose(
                    2, 0, 1)).float().div(255.0).unsqueeze(0)

                # 将张量移到设备上
                cropped_tensor = cropped_tensor.to(self.device)

                # 使用模型预测
                with torch.no_grad():
                    predictions = self.model(cropped_tensor)

                # 如果找到关键点
                if len(predictions) > 0 and len(predictions[0]['keypoints']) > 0:
                    # 获取第一个预测结果(在裁剪图像上应该只有一个人)
                    # [K, 3] - 关键点坐标和分数
                    keypoints = predictions[0]['keypoints'][0].cpu().numpy()
                    scores = predictions[0]['scores'][0].cpu().numpy()  # 检测框分数

                    # 调整关键点坐标到原图位置
                    adjusted_keypoints = keypoints.copy()
                    adjusted_keypoints[:, 0] += x1  # 调整x坐标
                    adjusted_keypoints[:, 1] += y1  # 调整y坐标

                    # 如果检测分数高于阈值
                    if scores >= self.conf_threshold:
                        # 过滤低置信度的关键点
                        mask = adjusted_keypoints[:,
                                                  2] < self.keypoint_threshold
                        adjusted_keypoints[mask, 2] = 0

                        keypoints_list.append(adjusted_keypoints.tolist())
                        scores_list.append(float(scores))

                # 如果在裁剪图像上没有检测到关键点，直接在原始人体框上创建一个伪检测
                # 这确保至少能返回一个结果
                elif len(keypoints_list) == 0:
                    # 对整个图像进行一次姿态估计
                    logger.info("裁剪图像上没有检测到关键点，尝试对整个图像进行预测")

                    # 转换为RGB格式
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # 转换为PyTorch张量
                    image_tensor = torch.from_numpy(image_rgb.transpose(
                        2, 0, 1)).float().div(255.0).unsqueeze(0)

                    # 将张量移到设备上
                    image_tensor = image_tensor.to(self.device)

                    # 使用模型预测
                    with torch.no_grad():
                        full_predictions = self.model(image_tensor)

                    # 如果在整个图像上找到关键点
                    if len(full_predictions) > 0 and len(full_predictions[0]['keypoints']) > 0 and len(full_predictions[0]['boxes']) > 0:
                        # 找到与我们的框IoU最高的预测框
                        pred_boxes = full_predictions[0]['boxes'].cpu().numpy()
                        best_iou = 0
                        best_idx = -1

                        for i, pred_box in enumerate(pred_boxes):
                            # 计算IoU
                            px1, py1, px2, py2 = pred_box

                            # 计算交集
                            ix1 = max(x1, px1)
                            iy1 = max(y1, py1)
                            ix2 = min(x2, px2)
                            iy2 = min(y2, py2)

                            if ix2 > ix1 and iy2 > iy1:
                                # 有交集
                                intersection = (ix2 - ix1) * (iy2 - iy1)
                                box_area = (x2 - x1) * (y2 - y1)
                                pred_area = (px2 - px1) * (py2 - py1)
                                union = box_area + pred_area - intersection
                                iou = intersection / union

                                if iou > best_iou:
                                    best_iou = iou
                                    best_idx = i

                        # 如果找到了匹配的预测框
                        if best_idx >= 0 and best_iou > 0.3:
                            keypoints = full_predictions[0]['keypoints'][best_idx].cpu(
                            ).numpy()
                            scores = full_predictions[0]['scores'][best_idx].cpu(
                            ).numpy()

                            # 过滤低置信度的关键点
                            mask = keypoints[:, 2] < self.keypoint_threshold
                            keypoints[mask, 2] = 0

                            keypoints_list.append(keypoints.tolist())
                            scores_list.append(float(scores))

            return keypoints_list, scores_list

        except Exception as e:
            logger.error(f"KeypointRCNN从边界框检测姿态失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [], []

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

            # 如果没有关键点，直接返回原图
            if not keypoints or len(keypoints) == 0:
                return image.copy()

            # 准备可视化数据
            pose_results = PoseDataSample()

            # 转换数据格式
            # 将keypoints从list转换为tensor
            # [person_num, keypoint_num, 3]
            keypoints_array = np.array(keypoints)

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
        vis_image = image.copy()

        # 定义线条颜色
        skeleton_color = (100, 100, 255)

        # 绘制每个检测到的姿态
        for i, kpts in enumerate(keypoints):
            # 绘制骨架
            if self.draw_skeleton:  # 根据设置决定是否绘制骨骼
                for p1_idx, p2_idx in COCO_SKELETON:
                    p1 = kpts[p1_idx]
                    p2 = kpts[p2_idx]

                    # 确保两个关键点都可见
                    if p1[2] > self.keypoint_threshold and p2[2] > self.keypoint_threshold:
                        p1_pos = (int(p1[0]), int(p1[1]))
                        p2_pos = (int(p2[0]), int(p2[1]))
                        cv2.line(vis_image, p1_pos, p2_pos, skeleton_color, 2)

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

    # 处理每个人体的姿态
    for i, kpts in enumerate(keypoints):
        score = scores[i] if scores and i < len(scores) else 1.0
        group_id = start_group_id + i

        # 将每个关键点转换为一个点形状
        for j, (x, y, conf) in enumerate(kpts):
            # 只添加置信度高于阈值的关键点
            if conf > 0:
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

                # 创建形状字典
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

                # 确保两个关键点都可见
                if p1[2] > 0 and p2[2] > 0:
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

        # 如果提供了人体框，尝试从框中检测姿态
        if existing_person_boxes and len(existing_person_boxes) > 0 and (use_detection_results is None or use_detection_results):
            logger.info(f"使用已有的 {len(existing_person_boxes)} 个人体框进行姿态估计")
            keypoints, scores = estimator.detect_poses_from_boxes(
                image, existing_person_boxes)

        # 如果没有结果，使用通用检测
        if len(keypoints) == 0:
            logger.info("未找到已有人体框或未启用使用已有框，使用标准姿态估计")
            keypoints, scores = estimator.detect_poses(image)

        # 将姿态结果转换为形状列表
        group_id = start_group_id
        if existing_person_boxes_ids and len(existing_person_boxes_ids) > 0:
            group_id = existing_person_boxes_ids[0] if existing_person_boxes_ids[0] is not None else start_group_id

        shapes = get_shapes_from_poses(
            keypoints, scores, group_id, draw_skeleton)

        return shapes
    except Exception as e:
        logger.error(f"姿态估计过程中出错: {e}")
        return []
