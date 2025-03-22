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
            model_path = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), 'weights', 'yolov7-w6-pose.pt')

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
                HAS_MMPOSE = True
            except ImportError:
                HAS_MMPOSE = False
                logger.warning(
                    "MMPose未安装，无法使用RTMPose模型。请安装mmpose：pip install openmim && mim install mmpose>=1.2.0")
                raise ImportError(
                    "MMPose未安装，无法使用RTMPose模型。请安装mmpose：pip install openmim && mim install mmpose>=1.2.0")

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
                    from mmpose.utils import register_all_modules
                    register_all_modules()

                    # 构建完整配置路径
                    config_path = os.path.join(
                        os.path.dirname(mmpose.__file__),
                        "..", "configs", "body_2d_keypoint", "rtmpose", config_file
                    )
                    if not os.path.exists(config_path):
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
                # 尝试从网络下载权重文件
                try:
                    from labelme._automation.model_downloader import download_rtmpose_model
                    logger.info(f"模型权重不存在，尝试从网络下载: {self.model_name}")
                    checkpoint_path = download_rtmpose_model(self.model_name)
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
                checkpoint_file = checkpoint_path

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
        """使用RTMPose模型检测"""
        try:
            from mmpose.apis import inference_topdown
            from mmpose.structures import merge_data_samples
            import torch

            # 记录开始时间
            t_start = time.time()

            # 检测运行设备
            if torch.cuda.is_available() and self.device == 'cuda':
                device = 'cuda'
            else:
                device = 'cpu'

            # 获取高级参数
            max_poses = self.advanced_params.get("max_poses", 20)
            min_keypoints = self.advanced_params.get("min_keypoints", 5)
            use_tracking = self.advanced_params.get("use_tracking", False)
            tracking_threshold = self.advanced_params.get(
                "tracking_threshold", 0.5)

            # 构建默认的人体框，覆盖整个图像
            image_height, image_width = image.shape[:2]
            default_bbox = [0, 0, image_width, image_height]

            # 使用默认边界框进行预测
            person_bboxes = torch.tensor(
                [[default_bbox[0], default_bbox[1],
                    default_bbox[2], default_bbox[3], 1.0]],
                device=device)

            # 进行姿态估计
            pose_results = inference_topdown(
                self.model, image, person_bboxes)

            # 如果只有一个结果，直接使用
            if len(pose_results) == 1:
                pose_result = pose_results[0]
            else:
                # 合并多个结果
                pose_result = merge_data_samples(pose_results)

            # 获取预测的关键点
            keypoints_tensor = pose_result.pred_instances.keypoints
            if self.model.dataset_meta.get('keypoint_weights', None) is not None:
                # 找到COCO数据集的17个关键点
                if 'keypoint_weights' in self.model.dataset_meta:
                    keypoint_weights = self.model.dataset_meta['keypoint_weights']
                    if len(keypoint_weights) == 17 and keypoints_tensor.shape[1] != 17:
                        # 找到COCO数据集对应的索引
                        coco_indices = [
                            i for i, w in enumerate(keypoint_weights) if w > 0
                        ]
                        if len(coco_indices) == 17:
                            keypoints_tensor = keypoints_tensor[:,
                                                                coco_indices]

            keypoints = keypoints_tensor.cpu().numpy()
            scores = pose_result.pred_instances.scores.cpu().numpy()

            # 过滤低置信度的姿态
            valid_poses = []
            valid_scores = []
            for i, (kpts, score) in enumerate(zip(keypoints, scores)):
                # 计算可见关键点数量
                visible_keypoints = sum(
                    1 for _, _, conf in kpts if conf >= self.keypoint_threshold)

                # 过滤掉可见关键点数量少于阈值的姿态
                if visible_keypoints >= min_keypoints and score >= self.conf_threshold:
                    valid_poses.append(kpts)
                    valid_scores.append(score)

                    # 如果达到最大姿态数量，停止添加
                    if len(valid_poses) >= max_poses:
                        break

            logger.debug(
                f"RTMPose检测完成: 找到 {len(valid_poses)} 个姿态, 耗时 {time.time() - t_start:.3f} [s]")

            return valid_poses, valid_scores
        except Exception as e:
            logger.error(f"RTMPose检测失败: {e}")
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
        """使用RTMPose可视化器进行可视化"""
        # 拷贝图像，避免修改原图
        vis_image = image.copy()

        # 如果没有检测到关键点，直接返回原图
        if not keypoints:
            return vis_image

        # 如果不绘制骨骼，使用只绘制关键点的方法
        if not self.draw_skeleton:
            return self._visualize_keypoints_only(image, keypoints, scores)

        # 如果没有RTMPose可视化器，使用通用可视化函数
        if not hasattr(self, 'visualizer'):
            return self._visualize_poses_generic(image, keypoints, scores)

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
                    kpt_thr=self.keypoint_threshold,
                    skeleton=True  # 强制绘制骨骼
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

                # 创建形状字典
                shape = {
                    "label": f"{kpt_name}",  # 不添加kpt_前缀
                    "points": [[float(x), float(y)]],
                    "group_id": group_id,
                    "shape_type": "point",
                    "flags": {}
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
