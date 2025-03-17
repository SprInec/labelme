import json
import os
import time
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import cv2
from loguru import logger
from PyQt5 import QtCore

# 尝试导入YOLOv7，如果不可用则提供错误信息
try:
    import torch
    import torchvision
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    logger.warning("YOLOv7依赖未安装，请安装torch和torchvision")

# YOLOv7模型类


class YOLOv7Detector:
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        初始化YOLOv7检测器

        Args:
            model_path: YOLOv7模型路径
            device: 运行设备 ('cpu' 或 'cuda')
        """
        if not HAS_YOLO:
            raise ImportError("YOLOv7依赖未安装，请安装torch和torchvision")

        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        logger.info(f"使用设备: {self.device}")

        # 加载模型
        self.model = self._load_model(model_path)

        # COCO数据集类别
        self.classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

    def _load_model(self, model_path: str):
        """加载YOLOv7模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        try:
            # 加载YOLOv7模型
            model = torch.hub.load('WongKinYiu/yolov7', 'custom', model_path)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise

    def detect(self, image: np.ndarray, conf_thres: float = 0.25,
               iou_thres: float = 0.45, classes: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        使用YOLOv7检测图像中的对象

        Args:
            image: 输入图像 (BGR格式)
            conf_thres: 置信度阈值
            iou_thres: IoU阈值用于NMS
            classes: 要检测的类别ID列表，None表示检测所有类别

        Returns:
            boxes: 边界框坐标 [x1, y1, x2, y2]
            scores: 置信度分数
            labels: 类别ID
        """
        t_start = time.time()

        # 转换图像格式
        if image.shape[2] == 4:  # 如果有alpha通道
            image = image[:, :, :3]

        # 使用模型进行推理
        results = self.model(image, size=640)
        # [x1, y1, x2, y2, conf, cls]
        detections = results.xyxy[0].cpu().numpy()

        # 过滤结果
        if classes is not None:
            mask = np.isin(detections[:, 5], classes)
            detections = detections[mask]

        boxes = detections[:, :4]  # [x1, y1, x2, y2]
        scores = detections[:, 4]  # confidence
        labels = detections[:, 5].astype(int)  # class id

        logger.debug(
            f"检测完成: 找到 {len(boxes)} 个对象, 耗时 {time.time() - t_start:.3f} [s]")

        return boxes, scores, labels

    def get_class_name(self, class_id: int) -> str:
        """获取类别名称"""
        if 0 <= class_id < len(self.classes):
            return self.classes[class_id]
        return f"unknown_{class_id}"


def get_shapes_from_detections(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray,
    class_names: List[str], filter_classes: Optional[List[str]] = None
) -> List[Dict]:
    """
    将检测结果转换为labelme形状格式

    Args:
        boxes: 边界框坐标 [x1, y1, x2, y2]
        scores: 置信度分数
        labels: 类别ID
        class_names: 类别名称列表
        filter_classes: 要包含的类别名称列表，None表示包含所有类别

    Returns:
        shapes: labelme形状列表
    """
    shapes = []

    for box, score, label_id in zip(boxes, scores, labels):
        if label_id >= len(class_names):
            label = f"unknown_{label_id}"
        else:
            label = class_names[label_id]

        # 如果指定了过滤类别，则跳过不在列表中的类别
        if filter_classes and label not in filter_classes:
            continue

        xmin, ymin, xmax, ymax = box

        shape = {
            "label": label,
            "points": [[xmin, ymin], [xmax, ymax]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
            "description": json.dumps({"score": float(score), "class_id": int(label_id)}),
        }
        shapes.append(shape)

    return shapes


def detect_objects(
    image: np.ndarray,
    model_path: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    filter_classes: Optional[List[str]] = None,
    device: str = 'cpu'
) -> List[Dict]:
    """
    在图像中检测对象并返回labelme形状格式的结果

    Args:
        image: 输入图像
        model_path: YOLOv7模型路径
        conf_threshold: 置信度阈值
        iou_threshold: IoU阈值
        filter_classes: 要检测的类别名称列表，None表示检测所有类别
        device: 运行设备 ('cpu' 或 'cuda')

    Returns:
        shapes: labelme形状列表
    """
    try:
        # 初始化检测器
        detector = YOLOv7Detector(model_path, device)

        # 如果指定了过滤类别，转换为类别ID
        class_ids = None
        if filter_classes:
            class_ids = [i for i, name in enumerate(
                detector.classes) if name in filter_classes]

        # 执行检测
        boxes, scores, labels = detector.detect(
            image,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=class_ids
        )

        # 转换为labelme形状
        shapes = get_shapes_from_detections(
            boxes, scores, labels, detector.classes, filter_classes
        )

        return shapes

    except Exception as e:
        logger.error(f"检测对象时出错: {e}")
        return []
