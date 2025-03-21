import os
import time
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import cv2
from loguru import logger

# 导入配置加载器
from labelme._automation.config_loader import ConfigLoader

# 尝试导入PyTorch依赖，如果不可用则提供错误信息
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("目标检测依赖未安装，请安装torch")

# COCO数据集的类别名称
COCO_CLASSES = [
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


class ObjectDetector:
    """目标检测器"""

    def __init__(self, model_name: str = None, conf_threshold: float = None,
                 device: str = None, filter_classes: List[str] = None,
                 nms_threshold: float = None, max_detections: int = None,
                 use_gpu_if_available: bool = None, advanced_params: dict = None):
        """
        初始化目标检测器

        Args:
            model_name: 模型名称，可选值: 
                - fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn, retinanet_resnet50_fpn (torchvision模型)
                - rtmdet_tiny, rtmdet_s, rtmdet_m, rtmdet_l (RTMDet模型)
            conf_threshold: 置信度阈值
            device: 运行设备 ('cpu' 或 'cuda')
            filter_classes: 过滤类别列表
            nms_threshold: 非极大值抑制阈值
            max_detections: 最大检测数量
            use_gpu_if_available: 如果可用则使用GPU
            advanced_params: 高级参数字典
        """
        if not HAS_TORCH:
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
        self.nms_threshold = nms_threshold or detection_config.get(
            "nms_threshold", 0.45)
        self.max_detections = max_detections or detection_config.get(
            "max_detections", 100)
        self.use_gpu_if_available = use_gpu_if_available if use_gpu_if_available is not None else detection_config.get(
            "use_gpu_if_available", True)

        # 加载高级参数
        self.advanced_params = advanced_params or detection_config.get(
            "advanced", {})
        self.pre_nms_top_n = self.advanced_params.get("pre_nms_top_n", 1000)
        self.pre_nms_threshold = self.advanced_params.get(
            "pre_nms_threshold", 0.5)
        self.max_size = self.advanced_params.get("max_size", 1333)
        self.min_size = self.advanced_params.get("min_size", 800)
        self.score_threshold = self.advanced_params.get(
            "score_threshold", 0.05)

        # 检查是否是RTMDet模型
        self.is_rtmdet = self.model_name.startswith("rtmdet")

        # 检查CUDA可用性
        if self.use_gpu_if_available and torch.cuda.is_available() and self.device == 'cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        logger.info(f"使用设备: {self.device}")

        # 加载模型
        self.model = self._load_model()

    def _load_model(self):
        """加载目标检测模型"""
        try:
            # 判断是否为RTMDet模型
            if self.is_rtmdet:
                return self._load_rtmdet_model()
            else:
                return self._load_torchvision_model()
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise

    def _load_torchvision_model(self):
        """加载torchvision中的目标检测模型"""
        # 导入torchvision检测模型
        import torchvision.models.detection as detection_models
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.transform import GeneralizedRCNNTransform

        # 尝试预下载模型（如果需要）
        try:
            from labelme._automation.model_downloader import download_torchvision_model
            download_torchvision_model(self.model_name)
        except Exception as e:
            logger.warning(f"预下载模型失败: {e}，将在创建模型时自动下载")

        # 根据模型名称选择不同的模型
        try:
            # 首先尝试使用weights参数（新版本torchvision）
            if self.model_name == "fasterrcnn_resnet50_fpn":
                model = detection_models.fasterrcnn_resnet50_fpn(
                    weights="DEFAULT",
                    min_size=self.min_size,
                    max_size=self.max_size,
                    box_score_thresh=self.score_threshold,
                    box_nms_thresh=self.nms_threshold,
                    box_detections_per_img=self.max_detections,
                    box_fg_iou_thresh=0.5,
                    box_bg_iou_thresh=0.5,
                    rpn_pre_nms_top_n_train=self.pre_nms_top_n,
                    rpn_pre_nms_top_n_test=self.pre_nms_top_n,
                    rpn_post_nms_top_n_train=self.pre_nms_top_n // 2,
                    rpn_post_nms_top_n_test=self.pre_nms_top_n // 2,
                    rpn_nms_thresh=self.pre_nms_threshold,
                    rpn_fg_iou_thresh=0.7,
                    rpn_bg_iou_thresh=0.3
                )
            elif self.model_name == "maskrcnn_resnet50_fpn":
                model = detection_models.maskrcnn_resnet50_fpn(
                    weights="DEFAULT",
                    min_size=self.min_size,
                    max_size=self.max_size,
                    box_score_thresh=self.score_threshold,
                    box_nms_thresh=self.nms_threshold,
                    box_detections_per_img=self.max_detections
                )
            elif self.model_name == "retinanet_resnet50_fpn":
                model = detection_models.retinanet_resnet50_fpn(
                    weights="DEFAULT",
                    min_size=self.min_size,
                    max_size=self.max_size,
                    score_thresh=self.score_threshold,
                    nms_thresh=self.nms_threshold,
                    detections_per_img=self.max_detections
                )
            else:
                logger.warning(
                    f"未知的模型名称: {self.model_name}，使用默认的Faster R-CNN模型")
                model = detection_models.fasterrcnn_resnet50_fpn(
                    weights="DEFAULT",
                    min_size=self.min_size,
                    max_size=self.max_size
                )
        except TypeError as e:
            logger.warning(
                f"使用weights参数加载模型失败: {e}，尝试使用旧版接口 (pretrained=True)")

            # 如果新接口失败，尝试使用旧接口（兼容旧版本torchvision）
            if self.model_name == "fasterrcnn_resnet50_fpn":
                model = detection_models.fasterrcnn_resnet50_fpn(
                    pretrained=True,
                    min_size=self.min_size,
                    max_size=self.max_size,
                    box_score_thresh=self.score_threshold,
                    box_nms_thresh=self.nms_threshold,
                    box_detections_per_img=self.max_detections,
                    box_fg_iou_thresh=0.5,
                    box_bg_iou_thresh=0.5,
                    rpn_pre_nms_top_n_train=self.pre_nms_top_n,
                    rpn_pre_nms_top_n_test=self.pre_nms_top_n,
                    rpn_post_nms_top_n_train=self.pre_nms_top_n // 2,
                    rpn_post_nms_top_n_test=self.pre_nms_top_n // 2,
                    rpn_nms_thresh=self.pre_nms_threshold,
                    rpn_fg_iou_thresh=0.7,
                    rpn_bg_iou_thresh=0.3
                )
            elif self.model_name == "maskrcnn_resnet50_fpn":
                model = detection_models.maskrcnn_resnet50_fpn(
                    pretrained=True,
                    min_size=self.min_size,
                    max_size=self.max_size,
                    box_score_thresh=self.score_threshold,
                    box_nms_thresh=self.nms_threshold,
                    box_detections_per_img=self.max_detections
                )
            elif self.model_name == "retinanet_resnet50_fpn":
                model = detection_models.retinanet_resnet50_fpn(
                    pretrained=True,
                    min_size=self.min_size,
                    max_size=self.max_size,
                    score_thresh=self.score_threshold,
                    nms_thresh=self.nms_threshold,
                    detections_per_img=self.max_detections
                )
            else:
                logger.warning(
                    f"未知的模型名称: {self.model_name}，使用默认的Faster R-CNN模型")
                model = detection_models.fasterrcnn_resnet50_fpn(
                    pretrained=True,
                    min_size=self.min_size,
                    max_size=self.max_size
                )

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

        logger.info(
            f"模型加载成功，使用torchvision预训练的{self.model_name if self.model_name in ['fasterrcnn_resnet50_fpn', 'maskrcnn_resnet50_fpn', 'retinanet_resnet50_fpn', 'keypointrcnn_resnet50_fpn'] else 'fasterrcnn_resnet50_fpn'}模型")

        return model

    def _load_rtmdet_model(self):
        """加载RTMDet模型"""
        try:
            # 尝试导入MMDetection相关依赖
            try:
                import mmdet
                from mmdet.apis import init_detector, inference_detector
                HAS_MMDET = True
            except ImportError:
                HAS_MMDET = False
                logger.warning(
                    "MMDetection未安装，无法使用RTMDet模型。请安装mmdet：pip install openmim && mim install mmdet>=3.0.0")
                raise ImportError("MMDetection未安装，无法使用RTMDet模型")

            # RTMDet模型配置和权重映射
            rtmdet_configs = {
                "rtmdet_tiny": {
                    "config": "rtmdet_tiny_8xb32-300e_coco.py",
                    "checkpoint": "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
                },
                "rtmdet_s": {
                    "config": "rtmdet_s_8xb32-300e_coco.py",
                    "checkpoint": "rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth"
                },
                "rtmdet_m": {
                    "config": "rtmdet_m_8xb32-300e_coco.py",
                    "checkpoint": "rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
                },
                "rtmdet_l": {
                    "config": "rtmdet_l_8xb32-300e_coco.py",
                    "checkpoint": "rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"
                }
            }

            if self.model_name not in rtmdet_configs:
                logger.warning(f"未知的RTMDet模型: {self.model_name}，使用rtmdet_tiny")
                self.model_name = "rtmdet_tiny"

            # 获取模型配置和权重
            model_config = rtmdet_configs[self.model_name]

            # 检查是否有本地配置文件，否则使用MMDetection默认配置
            config_file = model_config["config"]
            if not os.path.exists(config_file):
                # 尝试从mmdet获取配置文件
                try:
                    from mmengine.config import Config
                    from mmdet.utils import register_all_modules
                    register_all_modules()

                    # 构建完整配置路径
                    config_path = os.path.join(
                        os.path.dirname(mmdet.__file__),
                        "..", "configs", "rtmdet", config_file
                    )
                    if not os.path.exists(config_path):
                        logger.warning(f"配置文件不存在: {config_path}")
                        # 使用MMDetection默认配置
                        config_file = f"mmdet::rtmdet/{config_file}"
                    else:
                        config_file = config_path
                except Exception as e:
                    logger.warning(f"获取MMDetection配置文件失败: {e}")
                    config_file = f"mmdet::rtmdet/{config_file}"

            # 检查是否有本地权重文件，否则使用MMDetection默认权重
            checkpoint_file = model_config["checkpoint"]
            checkpoint_dir = os.path.join(os.path.expanduser(
                "~"), ".cache", "torch", "hub", "checkpoints")
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

            if not os.path.exists(checkpoint_path):
                # 尝试从网络下载权重文件
                try:
                    from labelme._automation.model_downloader import download_rtmdet_model
                    logger.info(f"模型权重不存在，尝试从网络下载: {self.model_name}")
                    checkpoint_path = download_rtmdet_model(self.model_name)
                    if checkpoint_path:
                        logger.info(f"模型下载成功: {checkpoint_path}")
                        checkpoint_file = checkpoint_path
                    else:
                        # 如果下载失败，使用MMDetection默认权重
                        logger.warning(f"模型下载失败，使用MMDetection默认权重")
                        checkpoint_file = None
                except Exception as e:
                    logger.warning(f"下载模型失败: {e}，使用MMDetection默认权重")
                    checkpoint_file = None
            else:
                checkpoint_file = checkpoint_path

            # 初始化模型
            model = init_detector(
                config_file,
                checkpoint_file,
                device=self.device
            )

            # 设置推理阈值
            if hasattr(model, 'test_cfg'):
                if hasattr(model.test_cfg, 'score_thr'):
                    model.test_cfg.score_thr = self.conf_threshold

            # COCO数据集的类别名称
            self.class_names = model.dataset_meta['classes']

            logger.info(f"RTMDet模型加载成功: {self.model_name}")
            return model

        except Exception as e:
            logger.error(f"加载RTMDet模型失败: {e}")
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
            # 判断是否使用RTMDet模型
            if self.is_rtmdet:
                return self._detect_rtmdet(image)
            else:
                return self._detect_torchvision(image)
        except Exception as e:
            logger.error(f"检测过程中出错: {e}")
            return [], [], []

    def _detect_torchvision(self, image: np.ndarray) -> Tuple[List[List[float]], List[int], List[float]]:
        """使用torchvision模型进行检测"""
        import torch
        import numpy as np
        from torchvision.transforms import functional as F

        # 记录开始时间
        t_start = time.time()

        # 将图像转换为RGB（如果是BGR格式）
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 转换为张量
        input_tensor = F.to_tensor(rgb_image)
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            # 进行推理
            output = self.model([input_tensor])

        # 提取预测结果
        boxes = []
        class_ids = []
        scores = []

        # 获取预测结果
        pred_boxes = output[0]['boxes'].cpu().numpy()
        pred_scores = output[0]['scores'].cpu().numpy()
        pred_labels = output[0]['labels'].cpu().numpy()

        # 遍历所有预测框
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            # 过滤低置信度的检测结果
            if score < self.conf_threshold:
                continue

            # 如果指定了过滤类别，且当前类别不在过滤列表中，跳过
            if self.filter_classes:
                # COCO类别ID映射
                coco_class_name = COCO_CLASSES[label - 1] if 0 < label <= len(
                    COCO_CLASSES) else f"unknown_{label}"
                if coco_class_name not in self.filter_classes:
                    continue

            boxes.append(box.tolist())
            class_ids.append(int(label))
            scores.append(float(score))

        logger.debug(
            f"目标检测完成: 找到 {len(boxes)} 个对象, 耗时 {time.time() - t_start:.3f} [s]")

        return boxes, class_ids, scores

    def _detect_rtmdet(self, image: np.ndarray) -> Tuple[List[List[float]], List[int], List[float]]:
        """使用RTMDet模型进行检测"""
        from mmdet.apis import inference_detector

        # 记录开始时间
        t_start = time.time()

        # 运行推理
        result = inference_detector(self.model, image)

        # 提取预测结果
        boxes_list = []
        class_ids = []
        scores_list = []

        # 处理检测结果
        pred_instances = result.pred_instances

        # 获取边界框、类别ID和置信度
        if len(pred_instances) > 0:
            pred_boxes = pred_instances.bboxes.cpu().numpy()
            pred_scores = pred_instances.scores.cpu().numpy()
            pred_labels = pred_instances.labels.cpu().numpy()

            # 过滤低置信度的检测结果
            mask = pred_scores >= self.conf_threshold
            boxes = pred_boxes[mask]
            scores = pred_scores[mask]
            labels = pred_labels[mask]

            # 处理过滤后的结果
            for box, label, score in zip(boxes, labels, scores):
                # 如果指定了过滤类别，且当前类别不在过滤列表中，跳过
                class_name = self.class_names[int(label)]
                if self.filter_classes and class_name not in self.filter_classes:
                    continue

                boxes_list.append(box.tolist())
                class_ids.append(int(label))
                scores_list.append(float(score))

        logger.debug(
            f"RTMDet检测完成: 找到 {len(boxes_list)} 个对象, 耗时 {time.time() - t_start:.3f} [s]")

        return boxes_list, class_ids, scores_list

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
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
            (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (0, 128, 128), (128, 0, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64),
            (64, 64, 0), (0, 64, 64), (64, 0, 64)
        ]

        for box, class_id, score in zip(boxes, class_ids, scores):
            x1, y1, x2, y2 = [int(i) for i in box]

            # 确保类别ID在有效范围内
            if class_id < 0 or class_id >= len(self.class_names):
                class_name = f"unknown({class_id})"
            else:
                class_name = self.class_names[class_id]

            color = colors[class_id % len(colors)]
            label = f"{class_name}: {score:.2f}"

            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # 绘制标签背景和文本
            text_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis_image, (x1, y1 - text_size[1] - 5),
                          (x1 + text_size[0], y1), color, -1)
            cv2.putText(vis_image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return vis_image


def get_shapes_from_detections(
    boxes: List[List[float]],
    class_ids: List[int],
    scores: List[float],
    class_names: List[str],
    start_group_id: int = 0
) -> List[Dict]:
    """
    将检测结果转换为标注形状列表

    Args:
        boxes: 边界框列表 [N, 4] - (x1, y1, x2, y2)
        class_ids: 类别ID列表 [N]
        scores: 置信度列表 [N]
        class_names: 类别名称列表
        start_group_id: 起始分组ID

    Returns:
        shapes: 形状列表
    """
    shapes = []
    for i, (box, class_id, score) in enumerate(zip(boxes, class_ids, scores)):
        x1, y1, x2, y2 = box

        # 确保类别ID在有效范围内
        if class_id < 0 or class_id >= len(class_names):
            label = f"unknown({class_id})"
        else:
            label = class_names[class_id]

        # 创建形状字典
        shape = {
            "label": label,
            "points": [[x1, y1], [x2, y2]],
            "group_id": start_group_id + i,
            "shape_type": "rectangle",
            "flags": {}
        }
        shapes.append(shape)

    return shapes


def detect_objects(
    image: np.ndarray,
    model_name: str = None,
    conf_threshold: float = None,
    device: str = None,
    filter_classes: List[str] = None,
    nms_threshold: float = None,
    max_detections: int = None,
    use_gpu_if_available: bool = None,
    advanced_params: dict = None,
    start_group_id: int = 0
) -> List[Dict]:
    """
    检测图像中的对象并返回形状列表

    Args:
        image: 输入图像
        model_name: 模型名称
        conf_threshold: 置信度阈值
        device: 运行设备
        filter_classes: 过滤类别列表
        nms_threshold: 非极大值抑制阈值
        max_detections: 最大检测数量
        use_gpu_if_available: 如果可用则使用GPU
        advanced_params: 高级参数字典
        start_group_id: 起始分组ID

    Returns:
        shapes: 形状列表
    """
    try:
        # 初始化目标检测器
        detector = ObjectDetector(
            model_name=model_name,
            conf_threshold=conf_threshold,
            device=device,
            filter_classes=filter_classes,
            nms_threshold=nms_threshold,
            max_detections=max_detections,
            use_gpu_if_available=use_gpu_if_available,
            advanced_params=advanced_params
        )

        # 检测图像中的对象
        boxes, class_ids, scores = detector.detect(image)

        # 将检测结果转换为形状列表
        shapes = get_shapes_from_detections(
            boxes, class_ids, scores, detector.class_names, start_group_id)

        return shapes
    except Exception as e:
        logger.error(f"目标检测过程中出错: {e}")
        return []
