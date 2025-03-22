"""
YOLOv7 PyTorch工具函数
"""
import torch
import logging
import math
import os
import time

logger = logging.getLogger(__name__)


def select_device(device=''):
    """选择设备（CPU或CUDA）"""
    # 设置device为'0'或'0,1,2,3'
    s = f'YOLOv7 🚀 torch {torch.__version__} '
    device = str(device).strip().lower().replace(
        'cuda:', '')  # 'cuda:0' -> '0'
    cpu = device == 'cpu'
    if cpu:
        logger.info(s + '使用CPU')
        return torch.device('cpu')
    else:
        # 检查是否支持CUDA
        cuda = not cpu and torch.cuda.is_available()
        if cuda:
            n = torch.cuda.device_count()
            if n > 1 and len(device) and device != 'all':  # 如果设置了特定的GPU
                device_ids = [int(device)] if len(device) == 1 else [
                    int(x) for x in device.split(',')]
                device_id = device_ids[0]
                assert device_id < n, f'指定的GPU ID {device_id} 超出范围，只有 {n} 个GPU可用'
                device = torch.device(f'cuda:{device_id}')
            else:
                # 使用所有GPU或第一个GPU
                device = torch.device('cuda:0')
            logger.info(f'{s} 使用CUDA 设备{device}')
            return device
        else:
            logger.info(f'{s} CUDA不可用，使用CPU')
            return torch.device('cpu')


def time_synchronized():
    """使用cuda.synchronize()获得更精确的时间"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def initialize_weights(model):
    """初始化模型权重"""
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


def model_info(model, verbose=False):
    """打印模型信息"""
    # 模型信息，n_p参数数量，n_g层数
    n_p = sum(x.numel() for x in model.parameters())  # 参数数量
    n_g = sum(1 for x in model.parameters() if x.requires_grad)  # 梯度参数数量
    if verbose:
        logger.info(
            f"{'层数':>15s}{'名称':>15s}{'梯度':>15s}{'参数':>15s}{'形状':>30s}{'输出形状':>30s}")

    return n_p, n_g
