import os
import yaml
from PyQt5 import QtCore, QtGui, QtWidgets
import labelme.styles

from labelme._automation.config_loader import ConfigLoader
from labelme._automation.model_downloader import set_torch_home

# 定义可用的目标检测模型
DETECTION_MODELS = {
    "fasterrcnn_resnet50_fpn": "Faster R-CNN ResNet-50 FPN",
    "maskrcnn_resnet50_fpn": "Mask R-CNN ResNet-50 FPN",
    "retinanet_resnet50_fpn": "RetinaNet ResNet-50 FPN",
    "rtmdet_tiny": "RTMDet-Tiny",
    "rtmdet_s": "RTMDet-Small",
    "rtmdet_m": "RTMDet-Medium",
    "rtmdet_l": "RTMDet-Large",
    "custom": "自定义模型",
}

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


class AISettingsDialog(QtWidgets.QDialog):
    """AI设置对话框"""

    def __init__(self, parent=None):
        """初始化AI设置对话框"""
        super(AISettingsDialog, self).__init__(parent)
        self.parent = parent

        # 设置PyTorch警告过滤
        try:
            import warnings
            warnings.filterwarnings("ignore", category=UserWarning)
            import os
            os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

            # 设置PyTorch模型下载镜像源
            set_torch_home()
        except:
            pass

        self.config_loader = ConfigLoader()
        self.config = self.config_loader.config

        self.setWindowTitle(self.tr("半自动标注配置"))
        self.setWindowFlags(
            self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint
        )
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)

        self.initUI()
        self.loadSettings()

    def initUI(self):
        """初始化UI"""
        layout = QtWidgets.QVBoxLayout()

        # 创建选项卡
        self.tabs = QtWidgets.QTabWidget()
        self.detection_tab = QtWidgets.QWidget()
        self.pose_tab = QtWidgets.QWidget()
        self.mask_tab = QtWidgets.QWidget()  # 新增Mask选项卡
        self.tabs.addTab(self.detection_tab, self.tr("目标检测"))
        self.tabs.addTab(self.pose_tab, self.tr("人体姿态估计"))
        self.tabs.addTab(self.mask_tab, self.tr("分割蒙版"))  # 添加AI蒙版选项卡

        # 设置目标检测选项卡
        self.setupDetectionTab()

        # 设置人体姿态估计选项卡
        self.setupPoseTab()

        # 设置AI蒙版选项卡
        self.setupMaskTab()

        # 添加按钮
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout.addWidget(self.tabs)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def setupDetectionTab(self):
        """设置目标检测选项卡"""
        layout = QtWidgets.QFormLayout()

        # 模型选择
        self.detection_model_combo = QtWidgets.QComboBox()
        # 添加Torchvision模型
        self.detection_model_combo.addItem(
            "Faster R-CNN ResNet-50 FPN", "fasterrcnn_resnet50_fpn")
        self.detection_model_combo.addItem(
            "Mask R-CNN ResNet-50 FPN", "maskrcnn_resnet50_fpn")
        self.detection_model_combo.addItem(
            "RetinaNet ResNet-50 FPN", "retinanet_resnet50_fpn")

        # 添加RTMDet模型
        self.detection_model_combo.addItem("RTMDet-Tiny", "rtmdet_tiny")
        self.detection_model_combo.addItem("RTMDet-Small", "rtmdet_s")
        self.detection_model_combo.addItem("RTMDet-Medium", "rtmdet_m")
        self.detection_model_combo.addItem("RTMDet-Large", "rtmdet_l")

        # 添加YOLO模型
        self.detection_model_combo.addItem("YOLOv7", "yolov7")
        self.detection_model_combo.addItem("YOLOv7-Tiny", "yolov7-tiny")
        self.detection_model_combo.addItem("YOLOv7x", "yolov7x")
        self.detection_model_combo.addItem("YOLOv7-W6", "yolov7-w6")
        self.detection_model_combo.addItem("YOLOv7-E6", "yolov7-e6")
        self.detection_model_combo.addItem("YOLOv7-D6", "yolov7-d6")
        self.detection_model_combo.addItem("YOLOv7-E6E", "yolov7-e6e")

        # 添加自定义模型
        self.detection_model_combo.addItem("自定义模型", "custom")

        self.detection_model_combo.currentIndexChanged.connect(
            self.onDetectionModelChanged)
        layout.addRow(self.tr("检测模型:"), self.detection_model_combo)

        # 自定义模型路径
        self.detection_custom_model_widget = QtWidgets.QWidget()
        custom_model_layout = QtWidgets.QHBoxLayout()
        custom_model_layout.setContentsMargins(0, 0, 0, 0)
        self.detection_custom_model_path = QtWidgets.QLineEdit()
        browse_button = QtWidgets.QPushButton(self.tr("浏览..."))
        browse_button.clicked.connect(self.browseDetectionModel)
        custom_model_layout.addWidget(self.detection_custom_model_path)
        custom_model_layout.addWidget(browse_button)
        self.detection_custom_model_widget.setLayout(custom_model_layout)
        layout.addRow(self.tr("自定义模型路径:"), self.detection_custom_model_widget)

        # 置信度阈值
        self.detection_conf_threshold = QtWidgets.QDoubleSpinBox()
        self.detection_conf_threshold.setRange(0.01, 1.0)
        self.detection_conf_threshold.setSingleStep(0.05)
        self.detection_conf_threshold.setDecimals(2)
        layout.addRow(self.tr("置信度阈值:"), self.detection_conf_threshold)

        # NMS阈值
        self.detection_nms_threshold = QtWidgets.QDoubleSpinBox()
        self.detection_nms_threshold.setRange(0.01, 1.0)
        self.detection_nms_threshold.setSingleStep(0.05)
        self.detection_nms_threshold.setDecimals(2)
        layout.addRow(self.tr("NMS阈值:"), self.detection_nms_threshold)

        # 最大检测数
        self.detection_max_detections = QtWidgets.QSpinBox()
        self.detection_max_detections.setRange(1, 1000)
        self.detection_max_detections.setSingleStep(10)
        layout.addRow(self.tr("最大检测数:"), self.detection_max_detections)

        # 设备选择
        self.detection_device = QtWidgets.QComboBox()
        self.detection_device.addItems(["cpu", "cuda"])
        layout.addRow(self.tr("运行设备:"), self.detection_device)

        # 自动使用GPU选项
        self.detection_use_gpu = QtWidgets.QCheckBox(self.tr("如果可用则使用GPU"))
        layout.addRow("", self.detection_use_gpu)

        # 高级设置分组 - Torchvision专用
        self.detection_torchvision_advanced_group = QtWidgets.QGroupBox(
            self.tr("Torchvision高级设置"))
        torchvision_advanced_layout = QtWidgets.QFormLayout()

        # Torchvision高级设置参数
        self.detection_pre_nms_top_n = QtWidgets.QSpinBox()
        self.detection_pre_nms_top_n.setRange(100, 10000)
        self.detection_pre_nms_top_n.setSingleStep(100)
        torchvision_advanced_layout.addRow(self.tr("NMS前候选框数量:"),
                                           self.detection_pre_nms_top_n)

        self.detection_pre_nms_threshold = QtWidgets.QDoubleSpinBox()
        self.detection_pre_nms_threshold.setRange(0.01, 1.0)
        self.detection_pre_nms_threshold.setSingleStep(0.05)
        self.detection_pre_nms_threshold.setDecimals(2)
        torchvision_advanced_layout.addRow(
            self.tr("NMS前阈值:"), self.detection_pre_nms_threshold)

        self.detection_max_size = QtWidgets.QSpinBox()
        self.detection_max_size.setRange(300, 5000)
        self.detection_max_size.setSingleStep(100)
        torchvision_advanced_layout.addRow(
            self.tr("最大图像尺寸:"), self.detection_max_size)

        self.detection_min_size = QtWidgets.QSpinBox()
        self.detection_min_size.setRange(100, 2000)
        self.detection_min_size.setSingleStep(100)
        torchvision_advanced_layout.addRow(
            self.tr("最小图像尺寸:"), self.detection_min_size)

        self.detection_score_threshold = QtWidgets.QDoubleSpinBox()
        self.detection_score_threshold.setRange(0.01, 1.0)
        self.detection_score_threshold.setSingleStep(0.01)
        self.detection_score_threshold.setDecimals(2)
        torchvision_advanced_layout.addRow(
            self.tr("检测分数阈值:"), self.detection_score_threshold)

        self.detection_torchvision_advanced_group.setLayout(
            torchvision_advanced_layout)
        layout.addRow(self.detection_torchvision_advanced_group)

        # 高级设置分组 - MMDetection专用
        self.detection_mmdet_advanced_group = QtWidgets.QGroupBox(
            self.tr("MMDetection高级设置"))
        mmdet_advanced_layout = QtWidgets.QFormLayout()

        # MMDetection高级设置参数
        self.detection_mmdet_max_detections = QtWidgets.QSpinBox()
        self.detection_mmdet_max_detections.setRange(1, 1000)
        self.detection_mmdet_max_detections.setSingleStep(10)
        mmdet_advanced_layout.addRow(
            self.tr("最大检测数:"), self.detection_mmdet_max_detections)

        self.detection_mmdet_nms_threshold = QtWidgets.QDoubleSpinBox()
        self.detection_mmdet_nms_threshold.setRange(0.01, 1.0)
        self.detection_mmdet_nms_threshold.setSingleStep(0.05)
        self.detection_mmdet_nms_threshold.setDecimals(2)
        mmdet_advanced_layout.addRow(
            self.tr("NMS阈值:"), self.detection_mmdet_nms_threshold)

        self.detection_mmdet_advanced_group.setLayout(mmdet_advanced_layout)
        layout.addRow(self.detection_mmdet_advanced_group)

        # 检测类别（多选框）
        self.detection_classes_widget = QtWidgets.QWidget()
        classes_main_layout = QtWidgets.QVBoxLayout()
        classes_main_layout.setContentsMargins(0, 0, 0, 0)

        self.detection_classes_list = QtWidgets.QListWidget()
        self.detection_classes_list.setSelectionMode(
            QtWidgets.QAbstractItemView.MultiSelection)
        for class_name in COCO_CLASSES:
            item = QtWidgets.QListWidgetItem(class_name)
            self.detection_classes_list.addItem(item)

        # 添加全选/取消全选按钮
        select_buttons_layout = QtWidgets.QHBoxLayout()
        select_all_button = QtWidgets.QPushButton(self.tr("全选"))
        deselect_all_button = QtWidgets.QPushButton(self.tr("取消全选"))
        select_all_button.clicked.connect(self.selectAllClasses)
        deselect_all_button.clicked.connect(self.deselectAllClasses)
        select_buttons_layout.addWidget(select_all_button)
        select_buttons_layout.addWidget(deselect_all_button)

        # 创建一个垂直布局来包含列表和按钮
        classes_layout = QtWidgets.QVBoxLayout()
        classes_layout.addWidget(self.detection_classes_list)
        classes_layout.addLayout(select_buttons_layout)

        # 将垂直布局添加到表单布局
        classes_main_layout.addLayout(classes_layout)
        self.detection_classes_widget.setLayout(classes_main_layout)
        layout.addRow(self.tr("检测类别:"), self.detection_classes_widget)

        self.detection_tab.setLayout(layout)

        # 初始状态下隐藏自定义模型路径控件
        self.detection_custom_model_widget.setVisible(False)

    def onDetectionModelChanged(self, index):
        """检测模型改变时的回调函数"""
        model_id = self.detection_model_combo.currentData()

        # 自定义模型路径显示逻辑
        is_custom = model_id == "custom"
        self.detection_custom_model_widget.setVisible(is_custom)

        # 根据模型类型显示/隐藏特定设置
        is_torchvision = model_id in [
            "fasterrcnn_resnet50_fpn", "maskrcnn_resnet50_fpn", "retinanet_resnet50_fpn"]
        is_mmdet = model_id.startswith("rtmdet")
        is_yolo = model_id.startswith("yolo")

        # 显示/隐藏相应的高级设置
        self.detection_torchvision_advanced_group.setVisible(is_torchvision)
        self.detection_mmdet_advanced_group.setVisible(is_mmdet)

        # 更新提示信息
        if is_yolo:
            # 检查是否安装了YOLOv7依赖
            try:
                import torch
                has_torch = True
            except ImportError:
                has_torch = False

            if not has_torch:
                # 确保应用正确的主题
                app = QtWidgets.QApplication.instance()
                parent_window = self.parent() if self.parent() else None
                if parent_window and hasattr(parent_window, 'currentTheme') and app:
                    current_theme = parent_window.currentTheme
                    if current_theme == "dark":
                        app.setPalette(labelme.styles.get_dark_palette())
                        app.setStyleSheet(labelme.styles.DARK_STYLE)
                    elif current_theme == "light":
                        app.setPalette(labelme.styles.get_light_palette())
                        app.setStyleSheet(labelme.styles.LIGHT_STYLE)
                    else:  # default theme
                        app.setPalette(
                            QtWidgets.QApplication.style().standardPalette())
                        app.setStyleSheet("")

                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("依赖缺失"),
                    self.tr("YOLO模型依赖未安装，请安装torch和torchvision后再使用YOLO模型。"
                            "系统将自动切换到Faster R-CNN模型。")
                )
                # 切换到Faster R-CNN
                for i in range(self.detection_model_combo.count()):
                    if self.detection_model_combo.itemData(i) == "fasterrcnn_resnet50_fpn":
                        self.detection_model_combo.setCurrentIndex(i)
                        break

        # 类别过滤仅适用于Torchvision和MMDet
        self.detection_classes_widget.setVisible(
            is_torchvision or is_mmdet or is_yolo)

    def browsePoseModel(self):
        """浏览姿态估计模型文件"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, self.tr("选择姿态估计模型文件"), "", "模型文件 (*.pt *.pth);;所有文件 (*)"
        )
        if path:
            self.pose_custom_model_path.setText(path)

    def browseDetectionModel(self):
        """浏览自定义检测模型文件"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, self.tr("选择模型文件"), "",
            self.tr("模型文件 (*.pt *.pth *.onnx *.bin);;所有文件 (*)"))
        if path:
            self.detection_custom_model_path.setText(path)

    def loadSettings(self):
        """加载设置"""
        # 目标检测设置
        detection_config = self.config.get("detection", {})
        if detection_config:
            # 模型名称
            model_name = detection_config.get(
                "model_name", "fasterrcnn_resnet50_fpn")
            # 找到对应的索引
            for i in range(self.detection_model_combo.count()):
                if self.detection_model_combo.itemData(i) == model_name:
                    self.detection_model_combo.setCurrentIndex(i)
                    break

            # 自定义模型路径
            if model_name == "custom":
                custom_path = detection_config.get("custom_model_path", "")
                self.detection_custom_model_path.setText(custom_path)
                self.detection_custom_model_widget.setVisible(True)
            else:
                self.detection_custom_model_widget.setVisible(False)

            # 其他参数
            self.detection_conf_threshold.setValue(
                detection_config.get("conf_threshold", 0.5))
            self.detection_nms_threshold.setValue(
                detection_config.get("nms_threshold", 0.45))
            self.detection_max_detections.setValue(
                detection_config.get("max_detections", 100))

            # 设备
            device = detection_config.get("device", "cpu")
            self.detection_device.setCurrentText(device)

            # 自动使用GPU
            self.detection_use_gpu.setChecked(
                detection_config.get("use_gpu_if_available", True))

            # 过滤类别
            filter_classes = detection_config.get("filter_classes", [])
            if filter_classes:
                # 取消全选并选中指定类别
                for i in range(self.detection_classes_list.count()):
                    item = self.detection_classes_list.item(i)
                    item.setSelected(item.text() in filter_classes)
            else:
                # 如果没有过滤类别，则全不选
                for i in range(self.detection_classes_list.count()):
                    self.detection_classes_list.item(i).setSelected(False)

            # 高级参数
            advanced_params = detection_config.get("advanced", {})
            if advanced_params:
                self.detection_pre_nms_top_n.setValue(
                    advanced_params.get("pre_nms_top_n", 1000))
                self.detection_pre_nms_threshold.setValue(
                    advanced_params.get("pre_nms_threshold", 0.5))
                self.detection_max_size.setValue(
                    advanced_params.get("max_size", 1333))
                self.detection_min_size.setValue(
                    advanced_params.get("min_size", 800))
                self.detection_score_threshold.setValue(
                    advanced_params.get("score_threshold", 0.05))

        # 姿态估计设置
        pose_config = self.config.get("pose_estimation", {})
        if pose_config:
            # 模型名称
            model_name = pose_config.get(
                "model_name", "keypointrcnn_resnet50_fpn")
            # 找到对应的索引
            for i in range(self.pose_model_name.count()):
                if self.pose_model_name.itemData(i) == model_name:
                    self.pose_model_name.setCurrentIndex(i)
                    break

            # 如果是自定义模型，设置路径
            if model_name == "custom":
                custom_path = pose_config.get("custom_model_path", "")
                self.pose_custom_model_path.setText(custom_path)
                self.pose_custom_model_widget.setVisible(True)

            # 置信度阈值
            self.pose_conf_threshold.setValue(
                pose_config.get("conf_threshold", 0.25))

            # 关键点阈值
            self.pose_keypoint_threshold.setValue(
                pose_config.get("keypoint_threshold", 0.3))

            # 绘制骨骼选项
            self.pose_draw_skeleton.setChecked(
                pose_config.get("draw_skeleton", True))

            # 设备
            device = pose_config.get("device", "cpu")
            self.pose_device.setCurrentText(device)

            # 使用检测结果
            self.pose_use_detection_results.setChecked(
                pose_config.get("use_detection_results", True))

            # 高级参数
            advanced_params = pose_config.get("advanced", {})
            if advanced_params:
                self.pose_max_poses.setValue(
                    advanced_params.get("max_poses", 20))
                self.pose_min_keypoints.setValue(
                    advanced_params.get("min_keypoints", 5))
                self.pose_use_tracking.setChecked(
                    advanced_params.get("use_tracking", False))
                self.pose_tracking_threshold.setValue(
                    advanced_params.get("tracking_threshold", 0.5))

        # 分割蒙版设置
        mask_config = self.config.get("mask", {})
        if mask_config:
            # 模型名称
            model_name = mask_config.get("model_name", "sam:latest")
            for i in range(self.mask_model_combo.count()):
                if self.mask_model_combo.itemData(i) == model_name:
                    self.mask_model_combo.setCurrentIndex(i)
                    break

            # 置信度阈值
            if hasattr(self, 'mask_conf_threshold') and self.mask_conf_threshold is not None:
                self.mask_conf_threshold.setValue(
                    mask_config.get("conf_threshold", 0.5))

            # 多边形简化容差
            if hasattr(self, 'mask_simplify_tolerance') and self.mask_simplify_tolerance is not None:
                self.mask_simplify_tolerance.setValue(
                    mask_config.get("simplify_tolerance", 1.0))

            # 多点预测总数
            self.mask_points_per_side.setValue(
                mask_config.get("points_per_side", 32))

            # 设备
            device = mask_config.get("device", "cpu")
            self.mask_device.setCurrentText(device)

            # 自动使用GPU
            self.mask_use_gpu.setChecked(
                mask_config.get("use_gpu_if_available", True))

            # 交互模式
            interactive_mode = mask_config.get("interactive_mode", "points")
            for i in range(self.mask_interactive_mode.count()):
                if self.mask_interactive_mode.itemData(i) == interactive_mode:
                    self.mask_interactive_mode.setCurrentIndex(i)
                    break

            # 高级参数
            if hasattr(self, 'mask_max_segments') and self.mask_max_segments is not None:
                self.mask_max_segments.setValue(
                    mask_config.get("max_segments", 5))

            if hasattr(self, 'mask_min_area') and self.mask_min_area is not None:
                self.mask_min_area.setValue(
                    mask_config.get("min_area", 100))

            if hasattr(self, 'mask_preprocess') and self.mask_preprocess is not None:
                self.mask_preprocess.setChecked(
                    mask_config.get("preprocess", True))

            if hasattr(self, 'mask_postprocess') and self.mask_postprocess is not None:
                self.mask_postprocess.setChecked(
                    mask_config.get("postprocess", True))

        # 更新AI Prompt
        prompt_config = self.config.get("prompt", {})
        self.text_prompt_input.setText(prompt_config.get("text", ""))
        self.score_threshold.setValue(
            prompt_config.get("score_threshold", 0.1))
        self.iou_threshold.setValue(prompt_config.get("iou_threshold", 0.5))

    def accept(self):
        """保存设置"""
        # 更新配置
        new_config = self.config.copy() if self.config else {}

        # 更新目标检测配置
        detection_config = {}

        # 获取模型名称
        model_idx = self.detection_model_combo.currentIndex()
        detection_config["model_name"] = self.detection_model_combo.itemData(
            model_idx)

        # 如果是自定义模型，保存路径
        if detection_config["model_name"] == "custom":
            detection_config["custom_model_path"] = self.detection_custom_model_path.text(
            )

        # 更新其他参数
        detection_config["conf_threshold"] = self.detection_conf_threshold.value()
        detection_config["nms_threshold"] = self.detection_nms_threshold.value()
        detection_config["max_detections"] = self.detection_max_detections.value()
        detection_config["device"] = self.detection_device.currentText()
        detection_config["use_gpu_if_available"] = self.detection_use_gpu.isChecked()

        # 更新过滤类别
        filter_classes = []
        for i in range(self.detection_classes_list.count()):
            item = self.detection_classes_list.item(i)
            if item.isSelected():
                filter_classes.append(item.text())
        detection_config["filter_classes"] = filter_classes

        # 更新高级参数
        advanced_params = {}
        advanced_params["pre_nms_top_n"] = self.detection_pre_nms_top_n.value()
        advanced_params["pre_nms_threshold"] = self.detection_pre_nms_threshold.value()
        advanced_params["max_size"] = self.detection_max_size.value()
        advanced_params["min_size"] = self.detection_min_size.value()
        advanced_params["score_threshold"] = self.detection_score_threshold.value()
        detection_config["advanced"] = advanced_params

        new_config["detection"] = detection_config

        # 更新姿态估计配置
        pose_config = {}

        # 获取模型名称
        model_idx = self.pose_model_name.currentIndex()
        model_name = self.pose_model_name.itemData(model_idx)

        # 如果是自定义模型，则使用自定义路径
        if model_name == "custom":
            custom_path = self.pose_custom_model_path.text()
            if custom_path:
                pose_config["model_name"] = "custom"
                pose_config["custom_model_path"] = custom_path
            else:
                # 如果未指定路径，则使用默认模型
                pose_config["model_name"] = "keypointrcnn_resnet50_fpn"
        else:
            pose_config["model_name"] = model_name

        # 更新其他参数
        pose_config["conf_threshold"] = self.pose_conf_threshold.value()
        pose_config["keypoint_threshold"] = self.pose_keypoint_threshold.value()
        pose_config["device"] = self.pose_device.currentText()
        pose_config["use_detection_results"] = self.pose_use_detection_results.isChecked()

        # 保存绘制骨骼设置
        pose_config["draw_skeleton"] = self.pose_draw_skeleton.isChecked()

        # 更新高级参数
        advanced_params = {}
        advanced_params["max_poses"] = self.pose_max_poses.value()
        advanced_params["min_keypoints"] = self.pose_min_keypoints.value()
        advanced_params["use_tracking"] = self.pose_use_tracking.isChecked()
        advanced_params["tracking_threshold"] = self.pose_tracking_threshold.value()
        pose_config["advanced"] = advanced_params

        new_config["pose_estimation"] = pose_config

        # 更新分割蒙版配置
        mask_config = {}

        # 获取模型名称
        model_idx = self.mask_model_combo.currentIndex()
        model_name = self.mask_model_combo.itemData(model_idx)
        mask_config["model_name"] = model_name

        # 设置模型类型（根据模型名称前缀）
        if ":" in model_name:
            mask_config["model_type"] = model_name.split(":")[0]
        else:
            mask_config["model_type"] = "sam"  # 默认类型

        # 更新其他参数
        mask_config["conf_threshold"] = self.mask_conf_threshold.value()
        mask_config["simplify_tolerance"] = self.mask_simplify_tolerance.value()
        mask_config["points_per_side"] = self.mask_points_per_side.value()
        mask_config["device"] = self.mask_device.currentText()
        mask_config["use_gpu_if_available"] = self.mask_use_gpu.isChecked()

        # 获取交互模式
        interactive_mode_idx = self.mask_interactive_mode.currentIndex()
        mask_config["interactive_mode"] = self.mask_interactive_mode.itemData(
            interactive_mode_idx)

        # 更新高级参数
        mask_config["max_segments"] = self.mask_max_segments.value()
        mask_config["min_area"] = self.mask_min_area.value()
        mask_config["preprocess"] = self.mask_preprocess.isChecked()
        mask_config["postprocess"] = self.mask_postprocess.isChecked()

        new_config["mask"] = mask_config

        # 更新默认AI模型
        if "ai" not in new_config:
            new_config["ai"] = {}

        # 确保使用的是模型标识符而不是UI显示名称
        new_config["ai"]["default"] = model_name

        # 更新AI Prompt配置
        prompt_config = {}
        prompt_config["text"] = self.text_prompt_input.toPlainText()
        prompt_config["score_threshold"] = self.score_threshold.value()
        prompt_config["iou_threshold"] = self.iou_threshold.value()
        new_config["prompt"] = prompt_config

        # 保存配置
        self.config_loader.save_config(new_config)

        super(AISettingsDialog, self).accept()

    def setupPoseTab(self):
        """设置人体姿态估计选项卡"""
        layout = QtWidgets.QFormLayout()

        # 模型选择
        self.pose_model_name = QtWidgets.QComboBox()
        self.pose_model_name.addItem(
            "KeypointRCNN (Torchvision)", "keypointrcnn_resnet50_fpn")
        # RTMO模型选项
        self.pose_model_name.addItem("RTMO-Tiny", "rtmo-t")
        self.pose_model_name.addItem("RTMO-Small", "rtmo-s")
        self.pose_model_name.addItem("RTMO-Medium", "rtmo-m")
        self.pose_model_name.addItem("RTMO-Large", "rtmo-l")
        # RTMPose模型选项
        self.pose_model_name.addItem("RTMPose-Tiny", "rtmpose-t")
        self.pose_model_name.addItem("RTMPose-Small", "rtmpose-s")
        self.pose_model_name.addItem("RTMPose-Medium", "rtmpose-m")
        self.pose_model_name.addItem("RTMPose-Large", "rtmpose-l")
        # 添加HRNet模型选项
        self.pose_model_name.addItem("HRNet-W32", "hrnet-w32")
        self.pose_model_name.addItem("HRNet-W48", "hrnet-w48")
        self.pose_model_name.addItem("YOLOv7-Pose", "yolov7_w6_pose")
        self.pose_model_name.addItem("自定义模型", "custom")
        self.pose_model_name.currentIndexChanged.connect(
            self.onPoseModelChanged)
        layout.addRow(self.tr("模型:"), self.pose_model_name)

        # 自定义模型路径
        self.pose_custom_model_widget = QtWidgets.QWidget()
        custom_model_layout = QtWidgets.QHBoxLayout()
        custom_model_layout.setContentsMargins(0, 0, 0, 0)
        self.pose_custom_model_path = QtWidgets.QLineEdit()
        browse_button = QtWidgets.QPushButton(self.tr("浏览..."))
        browse_button.clicked.connect(self.browsePoseModel)
        custom_model_layout.addWidget(self.pose_custom_model_path)
        custom_model_layout.addWidget(browse_button)
        self.pose_custom_model_widget.setLayout(custom_model_layout)
        layout.addRow(self.tr("自定义模型路径:"), self.pose_custom_model_widget)

        # 置信度阈值
        self.pose_conf_threshold = QtWidgets.QDoubleSpinBox()
        self.pose_conf_threshold.setRange(0.01, 1.0)
        self.pose_conf_threshold.setSingleStep(0.05)
        self.pose_conf_threshold.setDecimals(2)
        layout.addRow(self.tr("置信度阈值:"), self.pose_conf_threshold)

        # 关键点置信度阈值
        self.pose_keypoint_threshold = QtWidgets.QDoubleSpinBox()
        self.pose_keypoint_threshold.setRange(0.01, 1.0)
        self.pose_keypoint_threshold.setSingleStep(0.05)
        self.pose_keypoint_threshold.setDecimals(2)
        layout.addRow(self.tr("关键点置信度阈值:"), self.pose_keypoint_threshold)

        # 设备选择
        self.pose_device = QtWidgets.QComboBox()
        self.pose_device.addItems(["cpu", "cuda"])
        layout.addRow(self.tr("运行设备:"), self.pose_device)

        # 绘制骨骼选项
        self.pose_draw_skeleton = QtWidgets.QCheckBox(self.tr("绘制骨骼"))
        self.pose_draw_skeleton.setToolTip(
            self.tr("开启时检测结果会绘制骨骼连接线，关闭时只显示关键点"))
        self.pose_draw_skeleton.setChecked(True)  # 默认开启
        layout.addRow("", self.pose_draw_skeleton)

        # 使用已有的目标检测结果
        self.pose_use_detection_results = QtWidgets.QCheckBox(
            self.tr("使用已有的目标检测结果"))
        self.pose_use_detection_results.setToolTip(
            self.tr("当视图中存在person矩形框时，使用该框作为输入预测关键点"))
        self.pose_use_detection_results_widget = QtWidgets.QWidget()
        use_detection_layout = QtWidgets.QHBoxLayout()
        use_detection_layout.setContentsMargins(0, 0, 0, 0)
        use_detection_layout.addWidget(self.pose_use_detection_results)
        self.pose_use_detection_results_widget.setLayout(use_detection_layout)
        layout.addRow("", self.pose_use_detection_results_widget)

        # 高级设置分组
        self.pose_advanced_group = QtWidgets.QGroupBox(self.tr("高级设置"))
        advanced_layout = QtWidgets.QFormLayout()

        # 高级设置参数
        self.pose_max_poses = QtWidgets.QSpinBox()
        self.pose_max_poses.setRange(1, 100)
        self.pose_max_poses.setSingleStep(1)
        advanced_layout.addRow(self.tr("最大姿态数:"), self.pose_max_poses)

        self.pose_min_keypoints = QtWidgets.QSpinBox()
        self.pose_min_keypoints.setRange(1, 17)
        self.pose_min_keypoints.setSingleStep(1)
        advanced_layout.addRow(self.tr("最少关键点数:"), self.pose_min_keypoints)

        # 跟踪相关设置 (仅适用于RTMPose模型)
        self.pose_tracking_widget = QtWidgets.QWidget()
        tracking_layout = QtWidgets.QFormLayout()
        tracking_layout.setContentsMargins(0, 0, 0, 0)

        self.pose_use_tracking = QtWidgets.QCheckBox(self.tr("使用关键点跟踪"))
        tracking_layout.addRow("", self.pose_use_tracking)

        self.pose_tracking_threshold = QtWidgets.QDoubleSpinBox()
        self.pose_tracking_threshold.setRange(0.01, 1.0)
        self.pose_tracking_threshold.setSingleStep(0.05)
        self.pose_tracking_threshold.setDecimals(2)
        tracking_layout.addRow(self.tr("跟踪阈值:"), self.pose_tracking_threshold)

        self.pose_tracking_widget.setLayout(tracking_layout)
        advanced_layout.addRow(self.pose_tracking_widget)

        self.pose_advanced_group.setLayout(advanced_layout)
        layout.addRow(self.pose_advanced_group)

        self.pose_tab.setLayout(layout)

        # 初始状态下隐藏自定义模型路径控件
        self.pose_custom_model_widget.setVisible(False)

    def setupMaskTab(self):
        """设置AI蒙版选项卡"""
        layout = QtWidgets.QFormLayout()

        # AI Mask模型选择
        self.mask_model_combo = QtWidgets.QComboBox()
        MODEL_NAMES = [
            ("sam:latest", "SegmentAnything (accuracy)"),
            ("sam:300m", "SegmentAnything (balanced)"),
            ("sam:100m", "SegmentAnything (speed)"),
            ("efficientsam:latest", "EfficientSam (accuracy)"),
            ("efficientsam:10m", "EfficientSam (speed)"),
            ("sam2:large", "Sam2 (accuracy)"),
            ("sam2:latest", "Sam2 (balanced)"),
            ("sam2:small", "Sam2 (speed)"),
            ("sam2:tiny", "Sam2 (tiny)")
        ]
        for model_name, model_ui_name in MODEL_NAMES:
            self.mask_model_combo.addItem(model_ui_name, userData=model_name)
        layout.addRow(self.tr("AI蒙版模型:"), self.mask_model_combo)

        # 模型状态标签
        self.model_status_label = QtWidgets.QLabel(self.tr("模型状态: 未加载"))
        layout.addRow(self.tr(""), self.model_status_label)

        # AI Prompt设置
        prompt_group = QtWidgets.QGroupBox(self.tr("AI提示设置"))
        prompt_layout = QtWidgets.QVBoxLayout()

        # 文本提示输入
        prompt_input_layout = QtWidgets.QHBoxLayout()
        prompt_label = QtWidgets.QLabel(self.tr("AI Prompt:"))
        self.text_prompt_input = QtWidgets.QTextEdit()
        self.text_prompt_input.setPlaceholderText(
            self.tr("e.g., dog,cat,bird"))
        self.text_prompt_input.setMaximumHeight(50)
        prompt_input_layout.addWidget(prompt_label)
        prompt_input_layout.addWidget(self.text_prompt_input)
        prompt_layout.addLayout(prompt_input_layout)

        # Score和IoU阈值设置
        threshold_layout = QtWidgets.QHBoxLayout()

        # Score阈值
        score_layout = QtWidgets.QHBoxLayout()
        score_label = QtWidgets.QLabel(self.tr("Score阈值:"))
        self.score_threshold = QtWidgets.QDoubleSpinBox()
        self.score_threshold.setRange(0, 1)
        self.score_threshold.setSingleStep(0.05)
        self.score_threshold.setValue(0.1)
        score_layout.addWidget(score_label)
        score_layout.addWidget(self.score_threshold)
        threshold_layout.addLayout(score_layout)

        # IoU阈值
        iou_layout = QtWidgets.QHBoxLayout()
        iou_label = QtWidgets.QLabel(self.tr("IoU阈值:"))
        self.iou_threshold = QtWidgets.QDoubleSpinBox()
        self.iou_threshold.setRange(0, 1)
        self.iou_threshold.setSingleStep(0.05)
        self.iou_threshold.setValue(0.5)
        iou_layout.addWidget(iou_label)
        iou_layout.addWidget(self.iou_threshold)
        threshold_layout.addLayout(iou_layout)

        prompt_layout.addLayout(threshold_layout)
        prompt_group.setLayout(prompt_layout)
        layout.addRow(prompt_group)

        # 置信度阈值
        self.mask_conf_threshold = QtWidgets.QDoubleSpinBox()
        self.mask_conf_threshold.setRange(0.01, 1.0)
        self.mask_conf_threshold.setSingleStep(0.05)
        self.mask_conf_threshold.setDecimals(2)
        self.mask_conf_threshold.setValue(0.5)
        layout.addRow(self.tr("置信度阈值:"), self.mask_conf_threshold)

        # 设备选择
        self.mask_device = QtWidgets.QComboBox()
        self.mask_device.addItems(["cpu", "cuda"])
        layout.addRow(self.tr("运行设备:"), self.mask_device)

        # GPU选项
        self.mask_use_gpu = QtWidgets.QCheckBox(self.tr("如果可用则使用GPU"))
        self.mask_use_gpu.setChecked(True)
        layout.addRow("", self.mask_use_gpu)

        # 多边形简化参数
        self.mask_simplify_tolerance = QtWidgets.QDoubleSpinBox()
        self.mask_simplify_tolerance.setRange(0.0, 10.0)
        self.mask_simplify_tolerance.setSingleStep(0.1)
        self.mask_simplify_tolerance.setDecimals(1)
        self.mask_simplify_tolerance.setValue(1.0)
        layout.addRow(self.tr("多边形简化容差:"), self.mask_simplify_tolerance)

        # 互动模式
        self.mask_interactive_mode = QtWidgets.QComboBox()
        self.mask_interactive_mode.addItem("点击", "points")
        self.mask_interactive_mode.addItem("框选", "boxes")
        layout.addRow(self.tr("互动模式:"), self.mask_interactive_mode)

        # 点采样数量
        self.mask_points_per_side = QtWidgets.QSpinBox()
        self.mask_points_per_side.setRange(1, 64)
        self.mask_points_per_side.setSingleStep(1)
        self.mask_points_per_side.setValue(32)
        layout.addRow(self.tr("点采样数量:"), self.mask_points_per_side)

        # 高级参数组
        advanced_group = QtWidgets.QGroupBox(self.tr("高级参数"))
        advanced_layout = QtWidgets.QFormLayout()

        # 最大分割数量
        self.mask_max_segments = QtWidgets.QSpinBox()
        self.mask_max_segments.setRange(1, 20)
        self.mask_max_segments.setValue(5)
        advanced_layout.addRow(self.tr("最大分割数量:"), self.mask_max_segments)

        # 最小区域大小
        self.mask_min_area = QtWidgets.QSpinBox()
        self.mask_min_area.setRange(0, 10000)
        self.mask_min_area.setSingleStep(10)
        self.mask_min_area.setValue(100)
        advanced_layout.addRow(self.tr("最小区域大小(像素):"), self.mask_min_area)

        # 预处理选项
        self.mask_preprocess = QtWidgets.QCheckBox(self.tr("启用图像预处理"))
        self.mask_preprocess.setChecked(True)
        advanced_layout.addRow("", self.mask_preprocess)

        # 后处理选项
        self.mask_postprocess = QtWidgets.QCheckBox(self.tr("启用结果后处理"))
        self.mask_postprocess.setChecked(True)
        advanced_layout.addRow("", self.mask_postprocess)

        advanced_group.setLayout(advanced_layout)
        layout.addRow(advanced_group)

        self.mask_tab.setLayout(layout)

    def selectAllClasses(self):
        """全选所有类别"""
        for i in range(self.detection_classes_list.count()):
            self.detection_classes_list.item(i).setSelected(True)

    def deselectAllClasses(self):
        """取消全选所有类别"""
        for i in range(self.detection_classes_list.count()):
            self.detection_classes_list.item(i).setSelected(False)

    def onPoseModelChanged(self, index):
        """姿态估计模型改变时的处理函数"""
        model_key = self.pose_model_name.currentData()

        # 对于所有模型，都显示绘制骨骼选项
        self.pose_draw_skeleton.setVisible(True)

        # 根据模型类型显示或隐藏"使用已有检测结果"选项
        # YOLOv7模型不支持从边界框检测姿态
        is_yolov7 = model_key == "yolov7_w6_pose"
        self.pose_use_detection_results_widget.setVisible(not is_yolov7)

        # 如果当前模型是YOLOv7，取消选中"使用已有检测结果"选项
        if is_yolov7:
            self.pose_use_detection_results.setChecked(False)

        # 显示或隐藏自定义模型路径控件
        is_custom = model_key == "custom"
        self.pose_custom_model_widget.setVisible(is_custom)

        # 根据模型类型显示/隐藏特定设置
        is_rtmpose = model_key.startswith("rtmpose")
        is_hrnet = model_key.startswith("hrnet")
        # RTMPose和HRNet模型特有的设置
        self.pose_tracking_widget.setVisible(is_rtmpose or is_hrnet)
