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

# 尝试导入人体姿态估计依赖，如果不可用则提供错误信息
try:
    import torch
    import torchvision
    HAS_POSE = True
except ImportError:
    HAS_POSE = False
    logger.warning("人体姿态估计依赖未安装，请安装torch和torchvision")

# COCO数据集关键点定义
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# 关键点连接定义，用于可视化
COCO_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
    [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4], [3, 5], [4, 6]
]

# 人体姿态估计模型类


class PoseEstimator:
    def __init__(self, model_name: str = None, device: str = None, conf_threshold: float = None):
        """
        初始化人体姿态估计器

        Args:
            model_name: 模型名称，可选值：
                - 'keypointrcnn_resnet50_fpn'：使用torchvision预训练模型
                - 自定义模型路径
            device: 运行设备 ('cpu' 或 'cuda')
            conf_threshold: 置信度阈值
        """
        if not HAS_POSE:
            raise ImportError("人体姿态估计依赖未安装，请安装torch和torchvision")

        # 加载配置
        config_loader = ConfigLoader()
        pose_config = config_loader.get_pose_estimation_config()

        # 使用配置值或默认值
        self.model_name = model_name or pose_config.get(
            "model_name", "keypointrcnn_resnet50_fpn")
        self.device = device or pose_config.get("device", "cpu")
        self.conf_threshold = conf_threshold or pose_config.get(
            "conf_threshold", 0.5)

        # 检查CUDA可用性
        self.device = self.device if torch.cuda.is_available(
        ) and self.device == 'cuda' else 'cpu'
        logger.info(f"使用设备: {self.device}")

        # 加载模型
        self.model = self._load_model(self.model_name)
        self.keypoints = COCO_KEYPOINTS
        self.skeleton = COCO_SKELETON

    def _load_model(self, model_name: str):
        """加载人体姿态估计模型"""
        try:
            if model_name == 'keypointrcnn_resnet50_fpn':
                # 使用torchvision预训练模型
                model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
                    pretrained=True,
                    progress=True,
                    num_keypoints=17,  # COCO数据集有17个关键点
                    num_classes=2  # 背景和人
                )
            else:
                # 加载自定义模型
                if not os.path.exists(model_name):
                    raise FileNotFoundError(f"模型文件不存在: {model_name}")
                model = torch.load(model_name, map_location=self.device)

            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise

    def detect_poses(self, image: np.ndarray, conf_threshold: float = None) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """
        检测图像中的人体姿态

        Args:
            image: 输入图像 (BGR格式)
            conf_threshold: 关键点置信度阈值，如果为None则使用初始化时的阈值

        Returns:
            keypoints_list: 每个人的关键点坐标列表 [N, 17, 2]
            keypoints_scores: 每个人的关键点置信度列表 [N, 17]
            person_scores: 每个人的检测置信度列表 [N]
        """
        t_start = time.time()

        # 使用传入的阈值或默认阈值
        conf_threshold = conf_threshold if conf_threshold is not None else self.conf_threshold

        # 转换图像格式
        if image.shape[2] == 4:  # 如果有alpha通道
            image = image[:, :, :3]

        # 转换为RGB格式（PyTorch模型需要RGB）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 转换为PyTorch张量
        image_tensor = torch.from_numpy(
            image_rgb.transpose(2, 0, 1)).float().div(255.0)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # 使用模型进行推理
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # 处理结果
        keypoints_list = []
        keypoints_scores = []
        person_scores = []

        for i, pred in enumerate(predictions):
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            # [N, 17, 3] - (x, y, score)
            keypoints = pred['keypoints'].cpu().numpy()

            for j, (box, score, kps) in enumerate(zip(boxes, scores, keypoints)):
                # 只处理人的检测结果
                if score < conf_threshold:
                    continue

                # 提取关键点坐标和置信度
                kp_coords = kps[:, :2]  # [17, 2] - (x, y)
                kp_scores = kps[:, 2]   # [17] - score

                keypoints_list.append(kp_coords)
                keypoints_scores.append(kp_scores)
                person_scores.append(score)

        logger.debug(
            f"姿态估计完成: 找到 {len(keypoints_list)} 个人, 耗时 {time.time() - t_start:.3f} [s]")

        return keypoints_list, keypoints_scores, person_scores

    def visualize_poses(self, image: np.ndarray, keypoints_list: List[np.ndarray],
                        keypoints_scores: List[np.ndarray], conf_threshold: float = None) -> np.ndarray:
        """
        在图像上可视化人体姿态

        Args:
            image: 输入图像
            keypoints_list: 关键点坐标列表 [N, 17, 2]
            keypoints_scores: 关键点置信度列表 [N, 17]
            conf_threshold: 关键点置信度阈值，如果为None则使用初始化时的阈值

        Returns:
            vis_image: 可视化后的图像
        """
        # 使用传入的阈值或默认阈值
        conf_threshold = conf_threshold if conf_threshold is not None else self.conf_threshold

        vis_image = image.copy()

        # 定义颜色
        colors = [
            (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
            (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
            (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
            (255, 0, 255), (255, 0, 170), (255, 0, 85)
        ]

        # 绘制每个人的姿态
        for person_kps, person_scores in zip(keypoints_list, keypoints_scores):
            # 绘制关键点
            for i, ((x, y), score) in enumerate(zip(person_kps, person_scores)):
                if score < conf_threshold:
                    continue

                cv2.circle(vis_image, (int(x), int(y)), 5, colors[i], -1)

            # 绘制骨架
            for i, (j1, j2) in enumerate(self.skeleton):
                if (person_scores[j1] < conf_threshold or
                        person_scores[j2] < conf_threshold):
                    continue

                pt1 = (int(person_kps[j1][0]), int(person_kps[j1][1]))
                pt2 = (int(person_kps[j2][0]), int(person_kps[j2][1]))
                cv2.line(vis_image, pt1, pt2, colors[i % len(colors)], 2)

        return vis_image


def get_shapes_from_poses(
    keypoints_list: List[np.ndarray],
    keypoints_scores: List[np.ndarray],
    person_scores: List[float],
    conf_threshold: float = 0.5
) -> List[Dict]:
    """
    将姿态估计结果转换为labelme形状格式

    Args:
        keypoints_list: 关键点坐标列表 [N, 17, 2]
        keypoints_scores: 关键点置信度列表 [N, 17]
        person_scores: 人体检测置信度列表 [N]
        conf_threshold: 关键点置信度阈值

    Returns:
        shapes: labelme形状列表
    """
    shapes = []

    for person_idx, (keypoints, scores, person_score) in enumerate(zip(keypoints_list, keypoints_scores, person_scores)):
        # 为每个人创建一个组ID
        group_id = f"person_{person_idx}"

        # 为每个关键点创建一个点形状
        for i, ((x, y), score) in enumerate(zip(keypoints, scores)):
            if score < conf_threshold:
                continue

            keypoint_name = COCO_KEYPOINTS[i]
            shape = {
                "label": keypoint_name,
                "points": [[float(x), float(y)]],
                "group_id": group_id,
                "shape_type": "point",
                "flags": {},
                "description": json.dumps({
                    "score": float(score),
                    "person_score": float(person_score),
                    "keypoint_id": i
                }),
                "other_data": {},
                "mask": None,
            }
            shapes.append(shape)

    return shapes


def estimate_poses(
    image: np.ndarray,
    model_name: str = None,
    conf_threshold: float = None,
    device: str = None
) -> List[Dict]:
    """
    在图像中估计人体姿态并返回labelme形状格式的结果

    Args:
        image: 输入图像
        model_name: 模型名称或路径，如果为None则使用配置文件中的值
        conf_threshold: 置信度阈值，如果为None则使用配置文件中的值
        device: 运行设备，如果为None则使用配置文件中的值

    Returns:
        shapes: labelme形状列表
    """
    try:
        # 初始化姿态估计器（会自动从配置文件加载默认值）
        pose_estimator = PoseEstimator(
            model_name=model_name,
            device=device,
            conf_threshold=conf_threshold
        )

        # 检测姿态
        keypoints_list, keypoints_scores, person_scores = pose_estimator.detect_poses(
            image, conf_threshold=conf_threshold)

        # 转换为labelme形状格式
        shapes = get_shapes_from_poses(
            keypoints_list, keypoints_scores, person_scores,
            conf_threshold=conf_threshold if conf_threshold is not None else pose_estimator.conf_threshold
        )

        return shapes
    except Exception as e:
        logger.error(f"姿态估计失败: {e}")
        return []
