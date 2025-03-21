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
                 advanced_params: dict = None):
        """
        初始化姿态估计器

        Args:
            model_name: 模型名称，可选值:
                - keypointrcnn_resnet50_fpn (Torchvision关键点检测模型)
                - yolov7_w6_pose (YOLOv7姿态估计模型)
                - rtmpose_tiny, rtmpose_s, rtmpose_m, rtmpose_l (RTMPose姿态估计模型)
            device: 运行设备 ('cpu' 或 'cuda')
            conf_threshold: 置信度阈值
            keypoint_threshold: 关键点置信度阈值
            advanced_params: 高级参数字典
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
            "conf_threshold", 0.25)
        self.keypoint_threshold = keypoint_threshold or pose_config.get(
            "keypoint_threshold", 0.3)
        self.device = device or pose_config.get("device", "cpu")

        # 加载高级参数
        self.advanced_params = advanced_params or pose_config.get(
            "advanced", {})

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
            # 导入PyTorch
            import torch
            import torchvision

            # 导入yolov7专用模型代码
            try:
                from labelme._automation.yolov7.models.experimental import attempt_load
                from labelme._automation.yolov7.utils.general import check_img_size, non_max_suppression_kpt
                from labelme._automation.yolov7.utils.torch_utils import select_device
                HAS_YOLOV7 = True
            except ImportError:
                HAS_YOLOV7 = False
                logger.warning("YOLOv7依赖未安装，请安装YOLOv7")
                raise ImportError("YOLOv7依赖未安装")

            # 检查是否已经安装了YOLOv7的依赖项
            if not HAS_YOLOV7:
                raise ImportError(
                    "YOLOv7姿态估计依赖未安装，请安装YOLOv7依赖")

            # 加载模型
            self.model_name = 'yolov7_w6_pose'  # 使用固定的模型名称
            model_path = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), 'weights', 'yolov7-w6-pose.pt')

            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                model_dir = os.path.dirname(model_path)
                os.makedirs(model_dir, exist_ok=True)
                raise FileNotFoundError(
                    f"模型文件不存在: {model_path}，请手动下载YOLOv7姿态估计模型权重文件到此位置")

            # 加载模型
            device = select_device(self.device)
            model = attempt_load(model_path, map_location=device)
            self.stride = int(model.stride.max())
            self.imgsz = check_img_size(640, s=self.stride)
            model.to(device)

            # 存储模型相关方法
            self.non_max_suppression_kpt = non_max_suppression_kpt

            # 标记模型为eval模式
            model.eval()

            logger.info(f"姿态估计模型加载成功: {self.model_name}")
            return model

        except Exception as e:
            logger.error(f"加载YOLOv7姿态估计模型失败: {e}")
            raise

    def _load_rtmpose_model(self):
        """加载RTMPose姿态估计模型"""
        try:
            # 尝试导入MMPose相关依赖
            try:
                import mmpose
                from mmpose.apis import init_model
                from mmpose.evaluation.functional import nms
                from mmpose.structures import merge_data_samples
                from mmpose.registry import VISUALIZERS

                HAS_MMPOSE = True
            except ImportError:
                HAS_MMPOSE = False
                error_msg = "MMPose未安装，无法使用RTMPose模型。请安装mmpose：pip install openmim && mim install mmpose>=1.2.0"
                logger.warning(error_msg)
                raise ImportError(error_msg)

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
            if not os.path.exists(config_file):
                # 尝试从mmpose获取配置文件
                try:
                    from mmengine.config import Config
                    # 构建完整配置路径
                    config_path = os.path.join(
                        os.path.dirname(mmpose.__file__),
                        "..", "configs", "body_2d_keypoint", "rtmpose", config_file
                    )
                    if not os.path.exists(config_path):
                        logger.warning(f"配置文件不存在: {config_path}")
                        # 使用MMPose默认配置
                        config_file = f"mmpose::body_2d_keypoint/rtmpose/{config_file}"
                    else:
                        config_file = config_path
                except Exception as e:
                    logger.warning(f"获取MMPose配置文件失败: {e}")
                    config_file = f"mmpose::body_2d_keypoint/rtmpose/{config_file}"

            # 检查是否有本地权重文件，否则使用MMPose默认权重
            checkpoint_file = model_config["checkpoint"]
            checkpoint_dir = os.path.join(os.path.expanduser(
                "~"), ".cache", "torch", "hub", "checkpoints")
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

            if not os.path.exists(checkpoint_path):
                # 使用预训练权重
                checkpoint_file = None
            else:
                checkpoint_file = checkpoint_path

            try:
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
            except Exception as init_error:
                logger.error(f"初始化RTMPose模型失败: {init_error}")
                raise

        except ImportError as ie:
            logger.error(f"加载RTMPose模型失败: {ie}")
            raise ImportError(f"RTMPose模型加载失败: {ie}")
        except Exception as e:
            logger.error(f"加载RTMPose模型失败: {e}")
            raise

    def _load_keypointrcnn_model(self):
        """加载KeypointRCNN姿态估计模型"""
        try:
            # 检查是否已经安装了torchvision依赖
            try:
                import torchvision
                HAS_TORCHVISION = True
            except ImportError:
                HAS_TORCHVISION = False
                logger.warning("torchvision未安装，无法使用KeypointRCNN模型")
                raise ImportError("torchvision未安装")

            if not HAS_TORCHVISION:
                raise ImportError("KeypointRCNN姿态估计依赖未安装，请安装torchvision")

            # 加载模型
            self.model_name = 'keypointrcnn_resnet50_fpn'  # 使用固定的模型名称

            # 直接使用预训练模型
            model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
                pretrained=True)
            model.to(self.device)

            # 标记模型为eval模式
            model.eval()

            logger.info(f"KeypointRCNN模型加载成功: {self.model_name}")
            return model

        except Exception as e:
            logger.error(f"加载KeypointRCNN姿态估计模型失败: {e}")
            raise

    def detect_poses(self, image: np.ndarray) -> Tuple[List[List[List[float]]], List[float]]:
        """
        检测图像中的人体姿态

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            keypoints: 关键点列表 [N, K, 3] - (x, y, conf)
            scores: 人体检测的置信度列表 [N]
        """
        try:
            # 判断是否使用RTMPose模型
            if self.is_rtmpose:
                return self._detect_rtmpose(image)
            elif self.is_keypointrcnn:
                return self._detect_keypointrcnn(image)
            else:
                return self._detect_yolov7_pose(image)
        except Exception as e:
            logger.error(f"姿态估计过程中出错: {e}")
            return [], []

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
        """使用RTMPose姿态估计模型进行检测"""
        # 确保图像是RGB格式，因为MMPose模型通常需要RGB输入
        if image.shape[2] == 4:  # 如果有alpha通道
            image = image[:, :, :3]

        # 从BGR转换为RGB (如果输入是BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 使用MMPose进行推理
        from mmpose.apis import inference_topdown
        from mmpose.structures import merge_data_samples

        # 首先使用人体检测器检测人体
        # 在RTMPose中，我们需要先检测人体，然后进行关键点检测
        # 这里我们可以使用简单的方法：假设整个图像都是人体区域
        # 如果有更好的人体检测器，可以在这里替换

        # 创建简单的检测框 (整个图像)
        dummy_det_results = [
            {'bbox': [0, 0, image.shape[1], image.shape[0], 1.0]}]

        # 使用TopDown方法进行姿态估计
        pose_results = inference_topdown(
            self.model, image_rgb, dummy_det_results)

        if pose_results:
            # 合并结果
            pose_result = merge_data_samples(pose_results)

            # 提取关键点和分数
            keypoints_list = []
            scores_list = []

            # 处理预测实例
            pred_instances = pose_result.pred_instances

            if len(pred_instances) > 0:
                # 获取关键点和分数
                keypoints = pred_instances.keypoints.cpu().numpy()  # [N, K, 2]
                # [N, K]
                keypoint_scores = pred_instances.keypoint_scores.cpu().numpy()

                # 如果有分数，则使用分数；否则，使用默认分数
                if hasattr(pred_instances, 'scores'):
                    # [N]
                    instance_scores = pred_instances.scores.cpu().numpy()
                else:
                    instance_scores = np.ones(len(keypoints))

                # 过滤低置信度的实例
                mask = instance_scores >= self.conf_threshold
                keypoints = keypoints[mask]
                keypoint_scores = keypoint_scores[mask]
                instance_scores = instance_scores[mask]

                # 处理结果
                for i in range(len(keypoints)):
                    kpts = np.zeros((keypoints.shape[1], 3))
                    kpts[:, :2] = keypoints[i]
                    kpts[:, 2] = keypoint_scores[i]

                    # 过滤低置信度的关键点
                    kpts[kpts[:, 2] < self.keypoint_threshold, 2] = 0

                    keypoints_list.append(kpts.tolist())
                    scores_list.append(float(instance_scores[i]))

            return keypoints_list, scores_list

        return [], []

    def _detect_keypointrcnn(self, image: np.ndarray) -> Tuple[List[List[List[float]]], List[float]]:
        """使用KeypointRCNN姿态估计模型进行检测"""
        # 检查是否已经安装了torchvision依赖
        try:
            import torchvision
            HAS_TORCHVISION = True
        except ImportError:
            HAS_TORCHVISION = False
            logger.warning("torchvision未安装，无法使用KeypointRCNN模型")
            raise ImportError("torchvision未安装")

        if not HAS_TORCHVISION:
            raise ImportError("KeypointRCNN姿态估计依赖未安装，请安装torchvision")

        # 加载模型
        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            pretrained=True)
        model.to(self.device)

        # 标记模型为eval模式
        model.eval()

        # 转换图像格式
        if image.shape[2] == 4:  # 如果有alpha通道
            image = image[:, :, :3]

        # 转换为RGB格式（PyTorch模型需要RGB）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 转换为PyTorch张量
        img = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float().div(
            255.0).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            output = model(img)

        # 处理输出
        keypoints_list = []
        scores_list = []

        for i, det in enumerate(output):
            if 'keypoints' in det and 'scores' in det:
                # 获取关键点和分数
                keypoints = det['keypoints'].cpu().numpy()  # [N, K, 3]
                scores = det['scores'].cpu().numpy()  # [N]

                # 过滤低置信度的实例
                mask = scores >= self.conf_threshold
                keypoints = keypoints[mask]
                scores = scores[mask]

                for j, (kpts, score) in enumerate(zip(keypoints, scores)):
                    # 关键点置信度处理
                    kpts_conf = np.ones((kpts.shape[0], 1)) * score
                    kpts_with_conf = np.hstack((kpts[:, :2], kpts_conf))

                    # 过滤低置信度的关键点
                    kpts_with_conf[kpts[:, 2] < self.keypoint_threshold, 2] = 0

                    keypoints_list.append(kpts_with_conf.tolist())
                    scores_list.append(float(score))

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
        else:
            # YOLOv7 Pose模型不支持从边界框中检测姿态，因此直接返回空结果
            logger.warning("YOLOv7 Pose模型不支持从边界框中检测姿态")
            return [], []

    def _detect_rtmpose_from_boxes(self, image: np.ndarray, boxes: List[List[float]]) -> Tuple[List[List[List[float]]], List[float]]:
        """使用RTMPose从给定的边界框中检测姿态"""
        # 确保图像是RGB格式
        if image.shape[2] == 4:  # 如果有alpha通道
            image = image[:, :, :3]

        # 从BGR转换为RGB (如果输入是BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 使用MMPose进行推理
        from mmpose.apis import inference_topdown
        from mmpose.structures import merge_data_samples

        # 创建检测结果列表
        det_results = []
        for box in boxes:
            x1, y1, x2, y2 = box
            # 添加置信度为1.0
            det_results.append({'bbox': [x1, y1, x2, y2, 1.0]})

        # 使用TopDown方法进行姿态估计
        pose_results = inference_topdown(self.model, image_rgb, det_results)

        if pose_results:
            # 合并结果
            pose_result = merge_data_samples(pose_results)

            # 提取关键点和分数
            keypoints_list = []
            scores_list = []

            # 处理预测实例
            pred_instances = pose_result.pred_instances

            if len(pred_instances) > 0:
                # 获取关键点和分数
                keypoints = pred_instances.keypoints.cpu().numpy()  # [N, K, 2]
                # [N, K]
                keypoint_scores = pred_instances.keypoint_scores.cpu().numpy()

                # 如果有分数，则使用分数；否则，使用默认分数
                if hasattr(pred_instances, 'scores'):
                    # [N]
                    instance_scores = pred_instances.scores.cpu().numpy()
                else:
                    instance_scores = np.ones(len(keypoints))

                # 处理结果
                for i in range(len(keypoints)):
                    kpts = np.zeros((keypoints.shape[1], 3))
                    kpts[:, :2] = keypoints[i]
                    kpts[:, 2] = keypoint_scores[i]

                    # 过滤低置信度的关键点
                    kpts[kpts[:, 2] < self.keypoint_threshold, 2] = 0

                    keypoints_list.append(kpts.tolist())
                    scores_list.append(float(instance_scores[i]))

            return keypoints_list, scores_list

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
        if self.is_rtmpose and hasattr(self, 'visualizer'):
            return self._visualize_rtmpose(image, keypoints, scores)
        else:
            return self._visualize_poses_generic(image, keypoints, scores)

    def _visualize_rtmpose(self, image: np.ndarray, keypoints: List[List[List[float]]], scores: List[float] = None) -> np.ndarray:
        """使用RTMPose可视化器进行可视化"""
        # 拷贝图像，避免修改原图
        vis_image = image.copy()

        # 如果没有检测到关键点，直接返回原图
        if not keypoints:
            return vis_image

        # 转换为RGB
        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

        try:
            # 将关键点和分数转换为模型需要的格式
            from mmpose.structures import PoseDataSample
            import torch

            pose_data_sample = PoseDataSample()
            instance_data = {}

            # 将关键点转换为张量
            kpts_tensor = []
            kpt_scores_tensor = []
            for kpts in keypoints:
                kpts_array = np.array(kpts)
                kpts_tensor.append(kpts_array[:, :2])
                kpt_scores_tensor.append(kpts_array[:, 2])

            if kpts_tensor:
                instance_data['keypoints'] = torch.tensor(
                    np.array(kpts_tensor))
                instance_data['keypoint_scores'] = torch.tensor(
                    np.array(kpt_scores_tensor))

                if scores:
                    instance_data['scores'] = torch.tensor(np.array(scores))
                else:
                    instance_data['scores'] = torch.ones(len(kpts_tensor))

                pose_data_sample.pred_instances = instance_data

                # 使用可视化器绘制关键点
                vis_image_rgb = self.visualizer.visualize_pose(
                    image=vis_image_rgb,
                    data_sample=pose_data_sample,
                    draw_bbox=True,
                    kpt_thr=self.keypoint_threshold
                )

                # 转换回BGR
                vis_image = cv2.cvtColor(vis_image_rgb, cv2.COLOR_RGB2BGR)

                return vis_image
            else:
                return vis_image
        except Exception as e:
            logger.error(f"RTMPose可视化失败: {e}")
            # 如果RTMPose可视化失败，使用通用可视化方法
            return self._visualize_poses_generic(image, keypoints, scores)

    def _visualize_poses_generic(self, image: np.ndarray, keypoints: List[List[List[float]]], scores: List[float] = None) -> np.ndarray:
        """通用的姿态可视化方法"""
        vis_image = image.copy()

        # 定义线条颜色
        skeleton_color = (100, 100, 255)

        # 绘制每个检测到的姿态
        for i, kpts in enumerate(keypoints):
            # 绘制骨架
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

    def _letterbox(self, img, new_shape=640, color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        """
        调整图像大小并填充以保持宽高比
        """
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        # only scale down, do not scale up (for better test mAP)
        if not scaleup:
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]
                  ), (new_shape[0] - new_unpad[1])  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            # width, height ratios
            ratio = (new_shape[1] / shape[1]), (new_shape[0] / shape[0])

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, ratio, (dw, dh)


def get_shapes_from_poses(
    keypoints: List[List[List[float]]],
    scores: List[float] = None,
    start_group_id: int = 0
) -> List[Dict]:
    """
    将姿态估计结果转换为标注形状列表

    Args:
        keypoints: 关键点列表 [N, K, 3] - (x, y, conf)
        scores: 人体检测的置信度列表 [N]
        start_group_id: 起始分组ID

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
                # 获取关键点名称
                kpt_name = COCO_KEYPOINTS[j]

                # 创建形状字典
                shape = {
                    "label": f"kpt_{kpt_name}",
                    "points": [[float(x), float(y)]],
                    "group_id": group_id,
                    "shape_type": "point",
                    "flags": {}
                }
                shapes.append(shape)

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
    start_group_id: int = 0
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
            advanced_params=advanced_params
        )

        # 检测图像中的姿态
        keypoints, scores = estimator.detect_poses(image)

        # 将姿态结果转换为形状列表
        shapes = get_shapes_from_poses(keypoints, scores, start_group_id)

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
    start_group_id: int = 0
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
            advanced_params=advanced_params
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

        shapes = get_shapes_from_poses(keypoints, scores, group_id)

        return shapes
    except Exception as e:
        logger.error(f"姿态估计过程中出错: {e}")
        return []
