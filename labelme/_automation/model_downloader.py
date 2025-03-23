import os
import sys
import requests
import logging
from tqdm import tqdm
from pathlib import Path
import urllib.request
import shutil

logger = logging.getLogger(__name__)

# YOLOv7模型下载链接
YOLOV7_MODELS = {
    "yolov7.pt": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt",
    "yolov7-tiny.pt": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt",
    "yolov7x.pt": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt",
    "yolov7-w6.pt": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt",
    "yolov7-e6.pt": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt",
    "yolov7-d6.pt": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt",
    "yolov7-e6e.pt": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt",
}

# 国内镜像站YOLOv7模型下载链接
YOLOV7_MODELS_MIRROR = {
    "yolov7.pt": "https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_d1_8xb16-300e_coco/yolov7_d1_8xb16-300e_coco_20221123_023601-40376bae.pth",  # OpenMMLab镜像
    "yolov7-tiny.pt": "https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_tiny_syncbn_fast_8xb16-300e_coco/yolov7_tiny_syncbn_fast_8xb16-300e_coco_20221126_102719-0ee5bbdf.pth",  # OpenMMLab镜像
    # 其他模型暂无直接镜像，使用原链接
    "yolov7x.pt": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt",
    "yolov7-w6.pt": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt",
    "yolov7-e6.pt": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt",
    "yolov7-d6.pt": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt",
    "yolov7-e6e.pt": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt",
}

# RTMPose模型下载链接 - 更新为备用链接
RTMPOSE_MODELS = {
    "rtmpose_tiny": "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth",
    "rtmpose_s": "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth",
    "rtmpose_m": "https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth",
    "rtmpose_l": "https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_256x192-6e6babf0_20200708.pth"
}

# 国内镜像站RTMPose模型下载链接 - 更新为多个备用镜像
RTMPOSE_MODELS_MIRROR = {
    # 主备用路径
    "rtmpose_tiny": "https://mirror.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth",
    "rtmpose_s": "https://mirror.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth",
    "rtmpose_m": "https://mirror.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth",
    "rtmpose_l": "https://mirror.openmmlab.com/mmpose/top_down/resnet/res101_coco_256x192-6e6babf0_20200708.pth"
}

# RTMDet模型下载链接
RTMDET_MODELS = {
    "rtmdet_tiny": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
    "rtmdet_s": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth",
    "rtmdet_m": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth",
    "rtmdet_l": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"
}

# 国内镜像站RTMDet模型下载链接 (OpenMMLab模型已经可以在国内正常访问，这里提供备用镜像)
RTMDET_MODELS_MIRROR = {
    "rtmdet_tiny": "https://mirror.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
    "rtmdet_s": "https://mirror.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth",
    "rtmdet_m": "https://mirror.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth",
    "rtmdet_l": "https://mirror.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"
}

# 设置PyTorch镜像源环境变量


def get_automation_dir():
    """获取_automation目录的路径"""
    return os.path.dirname(os.path.abspath(__file__))


def get_model_dir(model_type):
    """
    获取模型存储目录

    Args:
        model_type: 模型类型，如'yolov7', 'mmpose', 'mmdetection', 'torch'

    Returns:
        str: 模型存储目录的路径
    """
    base_dir = get_automation_dir()
    model_dir = os.path.join(base_dir, model_type)
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def get_torch_home():
    """获取自定义的torch模型目录"""
    torch_dir = get_model_dir("torch")
    return torch_dir


def set_torch_home():
    """设置PyTorch模型下载镜像源和模型存储目录"""
    # 获取自定义的torch模型目录
    torch_home = get_torch_home()

    # 设置PyTorch模型下载目录（使用自定义目录而不是默认的~/.cache/torch）
    os.environ['TORCH_HOME'] = torch_home

    # 设置PyTorch Hub镜像源
    os.environ['TORCH_MODEL_ZOO'] = 'https://download.pytorch.org/models'

    # 尝试设置镜像源，如果环境变量已存在则不覆盖
    if 'TORCH_MIRROR' not in os.environ:
        mirrors = [
            'https://mirror.sjtu.edu.cn/pytorch-models',  # 上海交大镜像
            'https://mirrors.tuna.tsinghua.edu.cn/pytorch-models',  # 清华镜像
            'https://mirrors.aliyun.com/pytorch-models',  # 阿里云镜像
            'https://mirrors.huaweicloud.com/pytorch-models'  # 华为云镜像
        ]
        # 选择一个镜像源
        os.environ['TORCH_MIRROR'] = mirrors[0]

    logger.info(f"设置PyTorch模型下载目录: {os.environ.get('TORCH_HOME')}")
    logger.info(f"设置PyTorch模型下载镜像源: {os.environ.get('TORCH_MIRROR', '使用默认源')}")


set_torch_home()


def download_file(url, dest_path, chunk_size=8192, use_mirror=True):
    """
    下载文件并显示进度条，支持镜像源

    Args:
        url: 下载链接
        dest_path: 目标路径
        chunk_size: 块大小
        use_mirror: 是否使用镜像源

    Returns:
        bool: 下载是否成功
    """
    # 尝试使用镜像源处理
    if use_mirror and "download.openmmlab.com" in url:
        mirror_url = url.replace(
            "download.openmmlab.com", "mirror.openmmlab.com")
        logger.info(f"尝试使用镜像源下载: {mirror_url}")
        try:
            if download_file(mirror_url, dest_path, chunk_size, use_mirror=False):
                return True
            else:
                logger.warning(f"镜像源下载失败，尝试原始源")
        except Exception as e:
            logger.warning(f"镜像源下载失败: {e}，尝试原始源")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        # 确保目标目录存在
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)

        with open(dest_path, 'wb') as f, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    size = f.write(chunk)
                    bar.update(size)

        return True
    except Exception as e:
        logger.error(f"下载文件失败: {e}")
        return False


def download_yolov7_model(model_name="yolov7.pt", dest_dir=None):
    """
    下载YOLOv7模型，优先使用镜像源

    Args:
        model_name: 模型名称，可选值: yolov7.pt, yolov7-tiny.pt, yolov7x.pt, yolov7-w6.pt, yolov7-e6.pt, yolov7-d6.pt, yolov7-e6e.pt
        dest_dir: 目标目录，如果为None则使用_automation/yolov7/checkpoints

    Returns:
        str: 模型路径，如果下载失败则返回None
    """
    if model_name not in YOLOV7_MODELS:
        logger.error(
            f"未知的模型名称: {model_name}，可用模型: {', '.join(YOLOV7_MODELS.keys())}")
        return None

    # 优先使用指定的存储位置，若不指定则存放到_automation/yolov7/checkpoints
    if dest_dir is None:
        dest_dir = get_model_dir("yolov7")

    dest_path = os.path.join(dest_dir, model_name)

    # 如果模型已存在，则直接返回路径
    if os.path.exists(dest_path):
        logger.info(f"模型已存在: {dest_path}")
        return dest_path

    # 优先使用镜像链接
    mirror_url = YOLOV7_MODELS_MIRROR.get(model_name)
    original_url = YOLOV7_MODELS.get(model_name)

    logger.info(f"开始下载模型: {model_name}")

    # 先尝试使用镜像
    if mirror_url and mirror_url != original_url:
        logger.info(f"尝试使用镜像源下载: {mirror_url}")
        if download_file(mirror_url, dest_path):
            logger.info(f"模型从镜像源下载成功: {dest_path}")
            return dest_path
        else:
            logger.warning(f"从镜像源下载失败，尝试使用原始源")

    # 如果镜像下载失败或没有镜像，使用原始链接
    if download_file(original_url, dest_path):
        logger.info(f"模型从原始源下载成功: {dest_path}")
        return dest_path
    else:
        logger.error(f"模型下载失败: {model_name}")
        return None


# mmpose权重链接
RTMPOSE_MODEL_LINKS = {
    "rtmpose_tiny": {
        # 原始下载链接
        "original": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230228.pth",
        # 新增备用链接
        "backup": [
            "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth",
            "https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth"
        ]
    },
    "rtmpose_s": {
        "original": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230228.pth",
        "backup": [
            "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth",
            "https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_256x192-6e6babf0_20200708.pth"
        ]
    },
    "rtmpose_m": {
        "original": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230228.pth",
        "backup": [
            "https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_256x192-f6e307c2_20200709.pth",
            "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288-314c8528_20200708.pth"
        ]
    },
    "rtmpose_l": {
        "original": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-f016ffe0_20230228.pth",
        "backup": [
            "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288-314c8528_20200708.pth",
            "https://download.openmmlab.com/mmpose/top_down/resnet/res152_coco_384x288-3860d4c9_20200709.pth"
        ]
    },
}


def download_rtmpose_model(model_name: str, target_dir: str = None) -> str:
    """下载RTMPose模型权重并返回路径

    Args:
        model_name: 模型名称，例如：rtmpose_tiny, rtmpose_s, rtmpose_m, rtmpose_l
        target_dir: 保存目录，默认为_automation/mmpose/checkpoints

    Returns:
        str: 模型文件路径
    """
    if model_name not in RTMPOSE_MODEL_LINKS:
        logger.error(
            f"RTMPose模型{model_name}不在支持列表中。支持: {list(RTMPOSE_MODEL_LINKS.keys())}")
        return None

    # 默认保存目录
    if target_dir is None:
        target_dir = get_model_dir("mmpose")

    # 确保目录存在
    os.makedirs(target_dir, exist_ok=True)

    # 模型文件路径
    dest_path = os.path.join(target_dir, f"{model_name}.pth")

    # 如果模型已存在，则直接返回路径
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024 * 1024:  # 确保文件大小合理
        logger.info(f"RTMPose模型已存在: {dest_path}")
        return dest_path

    # 开始下载模型
    logger.info(f"开始下载RTMPose模型: {model_name}")
    model_links = RTMPOSE_MODEL_LINKS[model_name]

    # 首先尝试原始链接
    original_url = model_links["original"]
    logger.info(f"尝试从原始链接下载: {original_url}")
    if download_file(original_url, dest_path):
        logger.info(f"模型下载成功: {dest_path}")
        return dest_path

    # 原始链接失败，尝试备用链接
    backup_links = model_links.get("backup", [])
    for i, backup_url in enumerate(backup_links):
        logger.info(f"尝试从备用链接{i+1}下载: {backup_url}")

        # 创建临时文件路径
        temp_path = f"{dest_path}.downloading"

        try:
            if download_file(backup_url, temp_path):
                # 下载成功，重命名文件
                shutil.move(temp_path, dest_path)
                logger.info(f"模型从备用链接下载成功: {dest_path}")
                return dest_path

            # 删除下载失败的临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

        except Exception as e:
            logger.warning(f"从 {backup_url} 下载失败: {str(e)}")
            # 如果临时文件存在，删除它
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # 如果所有下载都失败，检查是否可以使用其他模型
    logger.error(f"无法下载{model_name}模型权重，检查网络连接")

    # 尝试查找本地其他版本的模型
    available_models = []
    for name in RTMPOSE_MODEL_LINKS.keys():
        local_path = os.path.join(target_dir, f"{name}.pth")
        if os.path.exists(local_path) and os.path.getsize(local_path) > 1024 * 1024:
            available_models.append((name, local_path))

    # 如果存在其他可用模型，使用它
    if available_models:
        backup_name, backup_path = available_models[0]
        logger.info(f"将使用备用模型 {backup_name}: {backup_path}")
        return backup_path

    # 所有尝试都失败
    return None


def download_rtmdet_model(model_name="rtmdet_s", dest_dir=None):
    """
    下载RTMDet模型，优先使用镜像源

    Args:
        model_name: 模型名称，可选值: rtmdet_tiny, rtmdet_s, rtmdet_m, rtmdet_l
        dest_dir: 目标目录，如果为None则使用_automation/mmdetection/checkpoints

    Returns:
        str: 模型路径，如果下载失败则返回None
    """
    if model_name not in RTMDET_MODELS:
        logger.error(
            f"未知的模型名称: {model_name}，可用模型: {', '.join(RTMDET_MODELS.keys())}")
        return None

    # 如果未指定目标目录，使用默认的目录
    if dest_dir is None:
        dest_dir = get_model_dir("mmdetection")

    # 确保目标目录存在
    os.makedirs(dest_dir, exist_ok=True)

    # 获取模型文件名
    if model_name == "rtmdet_tiny":
        file_name = "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
    elif model_name == "rtmdet_s":
        file_name = "rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth"
    elif model_name == "rtmdet_m":
        file_name = "rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
    elif model_name == "rtmdet_l":
        file_name = "rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth"
    else:
        file_name = "rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth"

    dest_path = os.path.join(dest_dir, file_name)

    # 如果模型已存在，则直接返回路径
    if os.path.exists(dest_path):
        logger.info(f"模型已存在: {dest_path}")
        return dest_path

    # 优先使用镜像链接
    mirror_url = RTMDET_MODELS_MIRROR.get(model_name)
    original_url = RTMDET_MODELS.get(model_name)

    logger.info(f"开始下载RTMDet模型: {model_name}")

    # 先尝试使用镜像
    if mirror_url:
        logger.info(f"尝试使用镜像源下载: {mirror_url}")
        if download_file(mirror_url, dest_path):
            logger.info(f"模型从镜像源下载成功: {dest_path}")
            return dest_path
        else:
            logger.warning(f"从镜像源下载失败，尝试使用原始源")

    # 如果镜像下载失败或没有镜像，使用原始链接
    if download_file(original_url, dest_path):
        logger.info(f"模型从原始源下载成功: {dest_path}")
        return dest_path
    else:
        logger.error(f"模型下载失败: {model_name}")
        return None


def download_torchvision_model(model_name):
    """
    触发下载Torchvision预训练模型到指定目录

    Args:
        model_name: 模型名称，例如fasterrcnn_resnet50_fpn

    Returns:
        bool: 下载是否成功
    """
    try:
        # 确保torch home目录已经设置为_automation/torch
        set_torch_home()

        import torch
        import torchvision.models.detection as detection_models

        logger.info(f"开始下载Torchvision模型: {model_name}")
        logger.info(f"模型将保存到: {os.environ.get('TORCH_HOME')}")

        # 先尝试使用weights参数（新版本torchvision）
        try:
            # 根据模型名称选择不同的模型进行下载
            if model_name == "fasterrcnn_resnet50_fpn":
                model = detection_models.fasterrcnn_resnet50_fpn(
                    weights="DEFAULT")
            elif model_name == "maskrcnn_resnet50_fpn":
                model = detection_models.maskrcnn_resnet50_fpn(
                    weights="DEFAULT")
            elif model_name == "retinanet_resnet50_fpn":
                model = detection_models.retinanet_resnet50_fpn(
                    weights="DEFAULT")
            elif model_name == "keypointrcnn_resnet50_fpn":
                model = detection_models.keypointrcnn_resnet50_fpn(
                    weights="DEFAULT")
            else:
                logger.error(f"未知的Torchvision模型: {model_name}")
                return False
        except TypeError as e:
            logger.warning(
                f"使用weights参数下载模型失败: {e}，尝试使用旧版接口 (pretrained=True)")

            # 如果新接口失败，尝试使用旧接口
            if model_name == "fasterrcnn_resnet50_fpn":
                model = detection_models.fasterrcnn_resnet50_fpn(
                    pretrained=True)
            elif model_name == "maskrcnn_resnet50_fpn":
                model = detection_models.maskrcnn_resnet50_fpn(pretrained=True)
            elif model_name == "retinanet_resnet50_fpn":
                model = detection_models.retinanet_resnet50_fpn(
                    pretrained=True)
            elif model_name == "keypointrcnn_resnet50_fpn":
                model = detection_models.keypointrcnn_resnet50_fpn(
                    pretrained=True)
            else:
                logger.error(f"未知的Torchvision模型: {model_name}")
                return False

        logger.info(f"Torchvision模型下载成功: {model_name}")
        return True
    except Exception as e:
        logger.error(f"下载Torchvision模型失败: {e}")
        return False


def download_model_gui(parent=None, model_type="yolov7", model_name="yolov7.pt", dest_dir="models"):
    """
    使用GUI下载模型

    Args:
        parent: 父窗口
        model_type: 模型类型，可选值: yolov7, rtmpose, rtmdet, torchvision
        model_name: 模型名称
        dest_dir: 目标目录

    Returns:
        str: 模型路径，如果下载失败则返回None
    """
    try:
        from PyQt5 import QtWidgets, QtCore

        # 确保目标目录存在
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)

        # 创建进度对话框
        progress_dialog = QtWidgets.QProgressDialog(
            f"正在下载{model_type}模型...", "取消", 0, 100, parent
        )
        progress_dialog.setWindowTitle("下载模型")
        progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        progress_dialog.setAutoClose(True)
        progress_dialog.setAutoReset(True)

        # 创建下载线程
        class DownloadThread(QtCore.QThread):
            progress_signal = QtCore.pyqtSignal(int)
            finished_signal = QtCore.pyqtSignal(str)
            error_signal = QtCore.pyqtSignal(str)

            def __init__(self, model_type, model_name, dest_dir):
                super().__init__()
                self.model_type = model_type
                self.model_name = model_name
                self.dest_dir = dest_dir

            def run(self):
                try:
                    if self.model_type == "yolov7":
                        if self.model_name not in YOLOV7_MODELS:
                            self.error_signal.emit(
                                f"未知的模型名称: {self.model_name}")
                            return

                        url = YOLOV7_MODELS[self.model_name]
                        dest_path = os.path.join(
                            self.dest_dir, self.model_name)

                        # 如果模型已存在，则直接返回路径
                        if os.path.exists(dest_path):
                            self.finished_signal.emit(dest_path)
                            return

                        # 下载模型
                        response = requests.get(url, stream=True)
                        response.raise_for_status()

                        total_size = int(
                            response.headers.get('content-length', 0))
                        downloaded_size = 0

                        # 确保目标目录存在
                        os.makedirs(os.path.dirname(
                            os.path.abspath(dest_path)), exist_ok=True)

                        with open(dest_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    size = f.write(chunk)
                                    downloaded_size += size
                                    progress = int(
                                        downloaded_size / total_size * 100)
                                    self.progress_signal.emit(progress)

                        self.finished_signal.emit(dest_path)
                    elif self.model_type == "rtmpose":
                        result = download_rtmpose_model(
                            self.model_name, self.dest_dir)
                        if result:
                            self.finished_signal.emit(result)
                        else:
                            self.error_signal.emit(
                                f"下载RTMPose模型失败: {self.model_name}")
                    elif self.model_type == "rtmdet":
                        result = download_rtmdet_model(
                            self.model_name, self.dest_dir)
                        if result:
                            self.finished_signal.emit(result)
                        else:
                            self.error_signal.emit(
                                f"下载RTMDet模型失败: {self.model_name}")
                    elif self.model_type == "torchvision":
                        result = download_torchvision_model(self.model_name)
                        if result:
                            self.finished_signal.emit("success")
                        else:
                            self.error_signal.emit(
                                f"下载Torchvision模型失败: {self.model_name}")
                    else:
                        self.error_signal.emit(f"不支持的模型类型: {self.model_type}")
                except Exception as e:
                    self.error_signal.emit(str(e))

        # 创建并启动下载线程
        download_thread = DownloadThread(model_type, model_name, dest_dir)
        download_thread.progress_signal.connect(progress_dialog.setValue)
        download_thread.finished_signal.connect(progress_dialog.close)
        download_thread.error_signal.connect(
            lambda msg: QtWidgets.QMessageBox.critical(parent, "下载错误", msg))

        download_thread.start()
        progress_dialog.exec_()

        # 等待线程完成
        if download_thread.isRunning():
            download_thread.wait()

        # 返回模型路径
        if model_type == "yolov7":
            dest_path = os.path.join(dest_dir, model_name)
            if os.path.exists(dest_path):
                return dest_path
        elif model_type == "torchvision":
            return "success"  # Torchvision模型由PyTorch自动管理

        return None
    except Exception as e:
        logger.error(f"GUI下载模型失败: {e}")
        # 回退到命令行下载
        if model_type == "yolov7":
            return download_yolov7_model(model_name, dest_dir)
        elif model_type == "rtmpose":
            return download_rtmpose_model(model_name, dest_dir)
        elif model_type == "rtmdet":
            return download_rtmdet_model(model_name, dest_dir)
        elif model_type == "torchvision":
            return download_torchvision_model(model_name)
        else:
            return None


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 解析命令行参数
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "yolov7.pt"

    # 下载模型
    download_yolov7_model(model_name)
