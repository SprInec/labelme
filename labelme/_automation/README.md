# labelme AI自动化模块

此模块为labelme添加了AI辅助标注功能，包括目标检测和人体姿态估计。

## 功能

1. **目标检测**：使用YOLOv7模型自动检测图像中的对象，并创建矩形标注。
2. **人体姿态估计**：检测图像中的人体姿态，并创建关键点标注。

## 依赖

要使用AI功能，需要安装以下依赖：

```bash
pip install torch torchvision opencv-python pyyaml
```

## 使用方法

1. 在labelme界面中，点击"AI"菜单，选择相应的功能：
   - "运行目标检测"：检测当前图像中的对象
   - "运行人体姿态估计"：检测当前图像中的人体姿态
   - "AI设置"：配置AI模型参数

2. 在"AI设置"对话框中，可以配置：
   - 模型路径
   - 置信度阈值
   - 运行设备（CPU或CUDA）
   - 其他特定参数

## 模型

### YOLOv7

默认情况下，目标检测功能使用YOLOv7模型。您需要下载预训练的YOLOv7模型文件，并在设置中指定其路径。

YOLOv7模型可以从以下地址下载：
- [YOLOv7官方仓库](https://github.com/WongKinYiu/yolov7)

### 人体姿态估计

人体姿态估计功能默认使用torchvision中的KeypointRCNN-ResNet50-FPN模型，无需额外下载。

您也可以使用自定义的姿态估计模型，在设置中指定其路径。

## 配置文件

AI模型的配置保存在`labelme/config/ai_models.yaml`文件中，包含以下内容：

```yaml
# YOLOv7模型配置
yolov7:
  model_path: "models/yolov7.pt"  # 模型路径
  conf_threshold: 0.25            # 置信度阈值
  iou_threshold: 0.45             # IoU阈值
  device: "cpu"                   # 运行设备
  filter_classes: []              # 过滤类别

# 人体姿态估计模型配置
pose_estimation:
  model_name: "keypointrcnn_resnet50_fpn"  # 模型名称或路径
  conf_threshold: 0.5                      # 置信度阈值
  device: "cpu"                            # 运行设备
```

## 开发者信息

如需扩展AI功能，可以修改以下文件：

- `labelme/_automation/object_detection.py`：目标检测模块
- `labelme/_automation/pose_estimation.py`：人体姿态估计模块
- `labelme/_automation/config_loader.py`：配置加载器
- `labelme/widgets/ai_settings_dialog.py`：AI设置对话框 