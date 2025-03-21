import os
import sys
import requests
import logging
from tqdm import tqdm
from pathlib import Path

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

# RTMPose模型下载链接
RTMPOSE_MODELS = {
    "rtmpose_tiny": "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-t_simcc-aic-coco_pt-aic-coco_420e-256x192-e0c9327b_20230127.pth",
    "rtmpose_s": "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230127.pth",
    "rtmpose_m": "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth",
    "rtmpose_l": "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-1f9a0168_20230126.pth"
}

# 国内镜像站RTMPose模型下载链接 (OpenMMLab模型已经可以在国内正常访问，这里提供备用镜像)
RTMPOSE_MODELS_MIRROR = {
    "rtmpose_tiny": "https://mirror.openmmlab.com/v1/projects/rtmpose/rtmpose-t_simcc-aic-coco_pt-aic-coco_420e-256x192-e0c9327b_20230127.pth",
    "rtmpose_s": "https://mirror.openmmlab.com/v1/projects/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230127.pth",
    "rtmpose_m": "https://mirror.openmmlab.com/v1/projects/rtmpose/rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth",
    "rtmpose_l": "https://mirror.openmmlab.com/v1/projects/rtmpose/rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-1f9a0168_20230126.pth"
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


def set_torch_home():
    """设置PyTorch模型下载镜像源"""
    # 注意：这需要在import torch之前设置
    os.environ['TORCH_HOME'] = os.path.join(
        os.path.expanduser('~'), '.cache', 'torch')
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


def download_yolov7_model(model_name="yolov7.pt", dest_dir="models"):
    """
    下载YOLOv7模型，优先使用镜像源

    Args:
        model_name: 模型名称，可选值: yolov7.pt, yolov7-tiny.pt, yolov7x.pt, yolov7-w6.pt, yolov7-e6.pt, yolov7-d6.pt, yolov7-e6e.pt
        dest_dir: 目标目录

    Returns:
        str: 模型路径，如果下载失败则返回None
    """
    if model_name not in YOLOV7_MODELS:
        logger.error(
            f"未知的模型名称: {model_name}，可用模型: {', '.join(YOLOV7_MODELS.keys())}")
        return None

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


def download_rtmpose_model(model_name="rtmpose_s", dest_dir=None):
    """
    下载RTMPose模型，优先使用镜像源

    Args:
        model_name: 模型名称，可选值: rtmpose_tiny, rtmpose_s, rtmpose_m, rtmpose_l
        dest_dir: 目标目录，如果为None则使用默认的MMPose缓存目录

    Returns:
        str: 模型路径，如果下载失败则返回None
    """
    if model_name not in RTMPOSE_MODELS:
        logger.error(
            f"未知的模型名称: {model_name}，可用模型: {', '.join(RTMPOSE_MODELS.keys())}")
        return None

    # 如果未指定目标目录，使用默认的缓存目录
    if dest_dir is None:
        dest_dir = os.path.join(os.path.expanduser(
            "~"), ".cache", "torch", "hub", "checkpoints")

    # 确保目标目录存在
    os.makedirs(dest_dir, exist_ok=True)

    # 获取模型文件名
    if model_name == "rtmpose_tiny":
        file_name = "rtmpose-t_simcc-aic-coco_pt-aic-coco_420e-256x192-e0c9327b_20230127.pth"
    elif model_name == "rtmpose_s":
        file_name = "rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230127.pth"
    elif model_name == "rtmpose_m":
        file_name = "rtmpose-m_simcc-aic-coco_pt-aic-coco_420e-256x192-63eb25f7_20230126.pth"
    elif model_name == "rtmpose_l":
        file_name = "rtmpose-l_simcc-aic-coco_pt-aic-coco_420e-256x192-1f9a0168_20230126.pth"
    else:
        file_name = "rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230127.pth"

    dest_path = os.path.join(dest_dir, file_name)

    # 如果模型已存在，则直接返回路径
    if os.path.exists(dest_path):
        logger.info(f"模型已存在: {dest_path}")
        return dest_path

    # 优先使用镜像链接
    mirror_url = RTMPOSE_MODELS_MIRROR.get(model_name)
    original_url = RTMPOSE_MODELS.get(model_name)

    logger.info(f"开始下载RTMPose模型: {model_name}")

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


def download_rtmdet_model(model_name="rtmdet_s", dest_dir=None):
    """
    下载RTMDet模型，优先使用镜像源

    Args:
        model_name: 模型名称，可选值: rtmdet_tiny, rtmdet_s, rtmdet_m, rtmdet_l
        dest_dir: 目标目录，如果为None则使用默认的MMDet缓存目录

    Returns:
        str: 模型路径，如果下载失败则返回None
    """
    if model_name not in RTMDET_MODELS:
        logger.error(
            f"未知的模型名称: {model_name}，可用模型: {', '.join(RTMDET_MODELS.keys())}")
        return None

    # 如果未指定目标目录，使用默认的缓存目录
    if dest_dir is None:
        dest_dir = os.path.join(os.path.expanduser(
            "~"), ".cache", "torch", "hub", "checkpoints")

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
    触发下载Torchvision预训练模型

    Args:
        model_name: 模型名称，例如fasterrcnn_resnet50_fpn

    Returns:
        bool: 下载是否成功
    """
    try:
        import torch
        import torchvision.models.detection as detection_models

        logger.info(f"开始下载Torchvision模型: {model_name}")

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
