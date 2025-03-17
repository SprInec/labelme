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


def download_file(url, dest_path, chunk_size=8192):
    """
    下载文件并显示进度条

    Args:
        url: 下载链接
        dest_path: 目标路径
        chunk_size: 块大小

    Returns:
        bool: 下载是否成功
    """
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
    下载YOLOv7模型

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

    url = YOLOV7_MODELS[model_name]
    dest_path = os.path.join(dest_dir, model_name)

    # 如果模型已存在，则直接返回路径
    if os.path.exists(dest_path):
        logger.info(f"模型已存在: {dest_path}")
        return dest_path

    logger.info(f"开始下载模型: {model_name}")
    if download_file(url, dest_path):
        logger.info(f"模型下载成功: {dest_path}")
        return dest_path
    else:
        logger.error(f"模型下载失败: {model_name}")
        return None


def download_model_gui(parent=None, model_type="yolov7", model_name="yolov7.pt", dest_dir="models"):
    """
    使用GUI下载模型

    Args:
        parent: 父窗口
        model_type: 模型类型，可选值: yolov7, pose
        model_name: 模型名称
        dest_dir: 目标目录

    Returns:
        str: 模型路径，如果下载失败则返回None
    """
    try:
        from PyQt5 import QtWidgets, QtCore

        # 确保目标目录存在
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
        dest_path = os.path.join(dest_dir, model_name)
        if os.path.exists(dest_path):
            return dest_path
        else:
            return None
    except Exception as e:
        logger.error(f"GUI下载模型失败: {e}")
        # 回退到命令行下载
        if model_type == "yolov7":
            return download_yolov7_model(model_name, dest_dir)
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
