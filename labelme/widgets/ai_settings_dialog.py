import os
import yaml
from PyQt5 import QtCore, QtGui, QtWidgets

from labelme._automation.config_loader import ConfigLoader

# 定义可用的目标检测模型
DETECTION_MODELS = {
    "fasterrcnn_resnet50_fpn": "Faster R-CNN ResNet-50 FPN",
    "maskrcnn_resnet50_fpn": "Mask R-CNN ResNet-50 FPN",
    "retinanet_resnet50_fpn": "RetinaNet ResNet-50 FPN",
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
        super(AISettingsDialog, self).__init__(parent)
        self.parent = parent
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.config

        self.setWindowTitle(self.tr("AI设置"))
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
        self.tabs.addTab(self.mask_tab, self.tr("AI蒙版"))  # 添加AI蒙版选项卡

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
        for model_id, model_name in DETECTION_MODELS.items():
            self.detection_model_combo.addItem(model_name, model_id)
        layout.addRow(self.tr("检测模型:"), self.detection_model_combo)

        # 置信度阈值
        self.detection_conf_threshold = QtWidgets.QDoubleSpinBox()
        self.detection_conf_threshold.setRange(0.01, 1.0)
        self.detection_conf_threshold.setSingleStep(0.05)
        self.detection_conf_threshold.setDecimals(2)
        layout.addRow(self.tr("置信度阈值:"), self.detection_conf_threshold)

        # 设备选择
        self.detection_device = QtWidgets.QComboBox()
        self.detection_device.addItems(["cpu", "cuda"])
        layout.addRow(self.tr("运行设备:"), self.detection_device)

        # 检测类别（多选框）
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
        classes_widget = QtWidgets.QWidget()
        classes_widget.setLayout(classes_layout)
        layout.addRow(self.tr("检测类别:"), classes_widget)

        self.detection_tab.setLayout(layout)

    def setupPoseTab(self):
        """设置人体姿态估计选项卡"""
        layout = QtWidgets.QFormLayout()

        # 模型选择
        self.pose_model_name = QtWidgets.QComboBox()
        self.pose_model_name.addItem(
            "keypointrcnn_resnet50_fpn", "keypointrcnn_resnet50_fpn")
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

        # 设备选择
        self.pose_device = QtWidgets.QComboBox()
        self.pose_device.addItems(["cpu", "cuda"])
        layout.addRow(self.tr("运行设备:"), self.pose_device)

        self.pose_tab.setLayout(layout)

    def setupMaskTab(self):
        """设置AI蒙版选项卡"""
        layout = QtWidgets.QFormLayout()

        # AI Mask模型选择
        self.mask_model_combo = QtWidgets.QComboBox()
        MODEL_NAMES = [
            ("efficientsam:10m", "EfficientSam (speed)"),
            ("efficientsam:latest", "EfficientSam (accuracy)"),
            ("sam:100m", "SegmentAnything (speed)"),
            ("sam:300m", "SegmentAnything (balanced)"),
            ("sam:latest", "SegmentAnything (accuracy)"),
            ("sam2:small", "Sam2 (speed)"),
            ("sam2:latest", "Sam2 (balanced)"),
            ("sam2:large", "Sam2 (accuracy)"),
        ]
        for model_name, model_ui_name in MODEL_NAMES:
            self.mask_model_combo.addItem(model_ui_name, userData=model_name)
        layout.addRow(self.tr("AI蒙版模型:"), self.mask_model_combo)

        # 当前选择的模型
        self.current_model_label = QtWidgets.QLabel(self.tr("当前使用的模型:"))
        layout.addRow(self.current_model_label)

        # AI Prompt设置
        prompt_group = QtWidgets.QGroupBox(self.tr("AI提示设置"))
        prompt_layout = QtWidgets.QVBoxLayout()

        # 文本提示输入
        prompt_input_layout = QtWidgets.QHBoxLayout()
        prompt_label = QtWidgets.QLabel(self.tr("AI Prompt:"))
        self.text_prompt_input = QtWidgets.QLineEdit()
        self.text_prompt_input.setPlaceholderText(
            self.tr("e.g., dog,cat,bird"))
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

        # 多边形简化参数
        self.mask_simplify_tolerance = QtWidgets.QDoubleSpinBox()
        self.mask_simplify_tolerance.setRange(0.0, 10.0)
        self.mask_simplify_tolerance.setSingleStep(0.1)
        self.mask_simplify_tolerance.setDecimals(1)
        self.mask_simplify_tolerance.setValue(1.0)
        layout.addRow(self.tr("多边形简化容差:"), self.mask_simplify_tolerance)

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
        """当姿态估计模型选择改变时"""
        is_custom = self.pose_model_name.currentData() == "custom"
        self.pose_custom_model_widget.setVisible(is_custom)

    def browsePoseModel(self):
        """浏览姿态估计模型文件"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, self.tr("选择姿态估计模型文件"), "", "模型文件 (*.pt *.pth);;所有文件 (*)"
        )
        if path:
            self.pose_custom_model_path.setText(path)

    def loadSettings(self):
        """加载设置"""
        # 目标检测设置
        detection_config = self.config.get("detection", {})

        # 设置模型
        model_name = detection_config.get(
            "model_name", "fasterrcnn_resnet50_fpn")
        for i in range(self.detection_model_combo.count()):
            if self.detection_model_combo.itemData(i) == model_name:
                self.detection_model_combo.setCurrentIndex(i)
                break

        # 设置置信度阈值
        self.detection_conf_threshold.setValue(
            detection_config.get("conf_threshold", 0.5))

        # 设置设备
        device = detection_config.get("device", "cpu")
        self.detection_device.setCurrentText(device)

        # 设置过滤类别
        filter_classes = detection_config.get("filter_classes", [])
        for i in range(self.detection_classes_list.count()):
            item = self.detection_classes_list.item(i)
            if item.text() in filter_classes:
                item.setSelected(True)

        # 姿态估计设置
        pose_config = self.config.get("pose_estimation", {})

        # 设置模型
        pose_model_name = pose_config.get(
            "model_name", "keypointrcnn_resnet50_fpn")
        if pose_model_name == "keypointrcnn_resnet50_fpn":
            self.pose_model_name.setCurrentIndex(0)
            self.pose_custom_model_widget.setVisible(False)
        else:
            self.pose_model_name.setCurrentIndex(1)
            self.pose_custom_model_path.setText(pose_model_name)
            self.pose_custom_model_widget.setVisible(True)

        # 设置置信度阈值
        self.pose_conf_threshold.setValue(
            pose_config.get("conf_threshold", 0.5))

        # 设置设备
        pose_device = pose_config.get("device", "cpu")
        self.pose_device.setCurrentText(pose_device)

        # AI蒙版设置
        ai_config = self.config.get("ai", {})
        mask_config = ai_config.get("mask", {})
        default_model = ai_config.get("default", "EfficientSam (speed)")

        # 设置AI蒙版模型
        for i in range(self.mask_model_combo.count()):
            if self.mask_model_combo.itemText(i) == default_model:
                self.mask_model_combo.setCurrentIndex(i)
                break

        # 设置AI蒙版置信度阈值
        self.mask_conf_threshold.setValue(
            mask_config.get("conf_threshold", 0.5))

        # 设置AI蒙版设备
        mask_device = mask_config.get("device", "cpu")
        self.mask_device.setCurrentText(mask_device)

        # 设置多边形简化容差
        self.mask_simplify_tolerance.setValue(
            mask_config.get("simplify_tolerance", 1.0))

        # 设置高级参数
        self.mask_max_segments.setValue(
            mask_config.get("max_segments", 5))
        self.mask_min_area.setValue(
            mask_config.get("min_area", 100))
        self.mask_preprocess.setChecked(
            mask_config.get("preprocess", True))
        self.mask_postprocess.setChecked(
            mask_config.get("postprocess", True))

        # 设置AI Prompt
        prompt_config = ai_config.get("prompt", {})
        self.text_prompt_input.setText(prompt_config.get("text", ""))
        self.score_threshold.setValue(
            prompt_config.get("score_threshold", 0.1))
        self.iou_threshold.setValue(prompt_config.get("iou_threshold", 0.5))

    def accept(self):
        """保存设置并关闭对话框"""
        # 更新目标检测设置
        detection_config = self.config.get("detection", {})
        detection_config["model_name"] = self.detection_model_combo.currentData()
        detection_config["conf_threshold"] = self.detection_conf_threshold.value()
        detection_config["device"] = self.detection_device.currentText()

        # 处理检测类别
        filter_classes = []
        for i in range(self.detection_classes_list.count()):
            item = self.detection_classes_list.item(i)
            if item.isSelected():
                filter_classes.append(item.text())
        detection_config["filter_classes"] = filter_classes

        self.config["detection"] = detection_config

        # 更新姿态估计设置
        pose_config = self.config.get("pose_estimation", {})

        if self.pose_model_name.currentData() == "custom":
            pose_config["model_name"] = self.pose_custom_model_path.text()
        else:
            pose_config["model_name"] = "keypointrcnn_resnet50_fpn"

        pose_config["conf_threshold"] = self.pose_conf_threshold.value()
        pose_config["device"] = self.pose_device.currentText()
        self.config["pose_estimation"] = pose_config

        # 更新AI蒙版设置
        ai_config = self.config.get("ai", {})
        ai_config["default"] = self.mask_model_combo.currentText()

        # 更新AI蒙版参数
        mask_config = ai_config.get("mask", {})
        mask_config["conf_threshold"] = self.mask_conf_threshold.value()
        mask_config["device"] = self.mask_device.currentText()
        mask_config["simplify_tolerance"] = self.mask_simplify_tolerance.value()

        # 更新高级参数
        mask_config["max_segments"] = self.mask_max_segments.value()
        mask_config["min_area"] = self.mask_min_area.value()
        mask_config["preprocess"] = self.mask_preprocess.isChecked()
        mask_config["postprocess"] = self.mask_postprocess.isChecked()

        ai_config["mask"] = mask_config

        # 更新AI Prompt设置
        prompt_config = ai_config.get("prompt", {})
        prompt_config["text"] = self.text_prompt_input.text()
        prompt_config["score_threshold"] = self.score_threshold.value()
        prompt_config["iou_threshold"] = self.iou_threshold.value()
        ai_config["prompt"] = prompt_config

        self.config["ai"] = ai_config

        # 保存配置
        self.config_loader.save_config(self.config)

        super(AISettingsDialog, self).accept()
