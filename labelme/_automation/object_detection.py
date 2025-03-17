import os
import time
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import cv2
from loguru import logger

# 导入配置加载器
from labelme._automation.config_loader import ConfigLoader

# 尝试导入YOLOv7依赖，如果不可用则提供错误信息
try:
    import torch
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    logger.warning("目标检测依赖未安装，请安装torch")


class ObjectDetector:
    """YOLOv7目标检测器"""

    def __init__(self, model_name: str = None, conf_threshold: float = None,
                 device: str = None, filter_classes: List[str] = None):
        """
        初始化目标检测器

        Args:
            model_name: 模型名称，可选值: fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn, retinanet_resnet50_fpn
            conf_threshold: 置信度阈值
            device: 运行设备 ('cpu' 或 'cuda')
            filter_classes: 过滤类别列表
        """
        if not HAS_YOLO:
            raise ImportError("目标检测依赖未安装，请安装torch")

        # 加载配置
        config_loader = ConfigLoader()
        detection_config = config_loader.get_detection_config()

        # 使用配置值或默认值
        self.model_name = model_name or detection_config.get(
            "model_name", "fasterrcnn_resnet50_fpn")
        self.conf_threshold = conf_threshold or detection_config.get(
            "conf_threshold", 0.5)
        self.device = device or detection_config.get("device", "cpu")
        self.filter_classes = filter_classes or detection_config.get(
            "filter_classes", [])

        # 检查CUDA可用性
        self.device = self.device if torch.cuda.is_available(
        ) and self.device == 'cuda' else 'cpu'
        logger.info(f"使用设备: {self.device}")

        # 加载模型
        self.model = self._load_model()

    def _load_model(self):
        """加载目标检测模型"""
        try:
            # 导入torchvision检测模型
            import torchvision.models.detection as detection_models

            # 根据模型名称选择不同的模型
            if self.model_name == "fasterrcnn_resnet50_fpn":
                model = detection_models.fasterrcnn_resnet50_fpn(
                    pretrained=True)
            elif self.model_name == "maskrcnn_resnet50_fpn":
                model = detection_models.maskrcnn_resnet50_fpn(pretrained=True)
            elif self.model_name == "retinanet_resnet50_fpn":
                model = detection_models.retinanet_resnet50_fpn(
                    pretrained=True)
            else:
                logger.warning(
                    f"未知的模型名称: {self.model_name}，使用默认的Faster R-CNN模型")
                model = detection_models.fasterrcnn_resnet50_fpn(
                    pretrained=True)

            model.to(self.device)
            model.eval()

            # COCO数据集的类别名称
            self.class_names = [
                'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
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

            logger.info(f"模型加载成功，使用torchvision预训练的{self.model_name}模型")

            return model
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise

    def detect(self, image: np.ndarray) -> Tuple[List[List[float]], List[int], List[float]]:
        """
        检测图像中的对象

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            boxes: 边界框列表 [N, 4] - (x1, y1, x2, y2)
            class_ids: 类别ID列表 [N]
            scores: 置信度列表 [N]
        """
        t_start = time.time()

        try:
            # 转换图像格式
            if image.shape[2] == 4:  # 如果有alpha通道
                image = image[:, :, :3]

            # 转换为RGB格式（PyTorch模型需要RGB）
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 转换为PyTorch张量
            img = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float().div(
                255.0).unsqueeze(0).to(self.device)

            # 使用模型进行推理
            with torch.no_grad():
                predictions = self.model(img)

                # Faster R-CNN返回的是字典列表
                pred = predictions[0]

                # 获取边界框、类别ID和置信度
                pred_boxes = pred['boxes'].cpu().numpy()
                pred_scores = pred['scores'].cpu().numpy()
                pred_labels = pred['labels'].cpu().numpy()

                # 过滤低置信度的检测结果
                mask = pred_scores >= self.conf_threshold
                boxes = pred_boxes[mask]
                scores = pred_scores[mask]
                labels = pred_labels[mask]

            # 提取边界框、类别ID和置信度
            boxes_list = []
            class_ids = []
            scores_list = []

            for box, label, score in zip(boxes, labels, scores):
                # 如果指定了过滤类别，且当前类别不在过滤列表中，跳过
                if self.filter_classes and self.class_names[label] not in self.filter_classes:
                    continue

                boxes_list.append(box.tolist())
                class_ids.append(int(label))
                scores_list.append(float(score))

            logger.debug(
                f"目标检测完成: 找到 {len(boxes_list)} 个对象, 耗时 {time.time() - t_start:.3f} [s]")

            return boxes_list, class_ids, scores_list
        except Exception as e:
            logger.error(f"检测过程中出错: {e}")
            return [], [], []

    def visualize(self, image: np.ndarray, boxes: List[List[float]],
                  class_ids: List[int], scores: List[float]) -> np.ndarray:
        """
        在图像上可视化检测结果

        Args:
            image: 输入图像
            boxes: 边界框列表 [N, 4] - (x1, y1, x2, y2)
            class_ids: 类别ID列表 [N]
            scores: 置信度列表 [N]

        Returns:
            vis_image: 可视化后的图像
        """
        vis_image = image.copy()

        # 定义颜色
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (0, 255, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128)
        ]

        # 绘制每个检测结果
        for box, cls_id, score in zip(boxes, class_ids, scores):
            x1, y1, x2, y2 = map(int, box)
            color = colors[cls_id % len(colors)]

            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            cls_name = self.class_names[cls_id]
            label = f"{cls_name}: {score:.2f}"

            # 计算标签位置
            text_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_w, text_h = text_size

            # 绘制标签背景
            cv2.rectangle(vis_image, (x1, y1 - text_h - 4),
                          (x1 + text_w, y1), color, -1)

            # 绘制标签文本
            cv2.putText(vis_image, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return vis_image


def get_shapes_from_detections(
    boxes: List[List[float]],
    class_ids: List[int],
    scores: List[float],
    class_names: List[str]
) -> List[Dict]:
    """
    将检测结果转换为labelme形状格式

    Args:
        boxes: 边界框列表 [N, 4] - (x1, y1, x2, y2)
        class_ids: 类别ID列表 [N]
        scores: 置信度列表 [N]
        class_names: 类别名称列表

    Returns:
        shapes: labelme形状列表
    """
    shapes = []

    for box, cls_id, score in zip(boxes, class_ids, scores):
        x1, y1, x2, y2 = map(float, box)
        cls_name = class_names[cls_id]

        shape = {
            "label": cls_name,
            "points": [[x1, y1], [x2, y2]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
            "description": f"score: {score:.2f}",
            "other_data": {},
            "mask": None,
        }
        shapes.append(shape)

    return shapes


def detect_objects(
    image: np.ndarray,
    model_name: str = None,
    conf_threshold: float = None,
    device: str = None,
    filter_classes: List[str] = None
) -> List[Dict]:
    """
    在图像中检测对象并返回labelme形状格式的结果

    Args:
        image: 输入图像
        model_name: 模型名称，如果为None则使用配置文件中的值
        conf_threshold: 置信度阈值，如果为None则使用配置文件中的值
        device: 运行设备，如果为None则使用配置文件中的值
        filter_classes: 过滤类别列表，如果为None则使用配置文件中的值

    Returns:
        shapes: labelme形状列表
    """
    try:
        # 初始化目标检测器
        detector = ObjectDetector(
            model_name=model_name,
            conf_threshold=conf_threshold,
            device=device,
            filter_classes=filter_classes
        )

        # 检测对象
        boxes, class_ids, scores = detector.detect(image)

        # 转换为labelme形状格式
        shapes = get_shapes_from_detections(
            boxes, class_ids, scores, detector.class_names
        )

        return shapes
    except Exception as e:
        logger.error(f"目标检测失败: {e}")
        return []
