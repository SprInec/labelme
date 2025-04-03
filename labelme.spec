# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from pathlib import Path
import site
import osam

block_cipher = None

# 项目根目录
WORK_PATH = Path.cwd()
LABELME_PATH = WORK_PATH / 'labelme'

# 查找osam包路径
OSAM_PATH = Path(osam.__file__).parent

# 收集所有数据文件
datas = [
    (str(LABELME_PATH / 'icons'), 'labelme/icons'),
    (str(LABELME_PATH / 'translate'), 'labelme/translate'),
    (str(LABELME_PATH / 'config'), 'labelme/config'),
]

# 添加CLIP词汇文件
clip_vocab_file = str(OSAM_PATH / '_models/yoloworld/clip/bpe_simple_vocab_16e6.txt.gz')
clip_dest_dir = 'osam/_models/yoloworld/clip'
datas.append((clip_vocab_file, clip_dest_dir))

# 查找并添加osam包的所有数据文件
for root, dirs, files in os.walk(OSAM_PATH):
    for file in files:
        if file.endswith(('.pth', '.yaml', '.json', '.txt', '.gz', '.bin')):
            file_path = os.path.join(root, file)
            rel_dir = os.path.relpath(os.path.dirname(file_path), os.path.dirname(OSAM_PATH))
            dest_dir = os.path.join('osam', rel_dir)
            datas.append((file_path, dest_dir))

# 收集所有必要模块
hiddenimports = [
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'imgviz',
    'natsort',
    'matplotlib',
    'matplotlib.backends.backend_qt5agg',
    'matplotlib.backends.backend_qt5',
    'numpy',
    'scipy',
    'scikit-image',
    'PIL',
    'PIL._imagingtk',
    'PIL._tkinter_finder',
    'yaml',
    'loguru',
    'osam',
    'osam._models',
    'osam._models.yoloworld',
    'osam._models.yoloworld.clip',
    'shapely',
    'torch',
    'torchvision',
]

# 添加模型文件目录作为数据文件
model_dirs = [
    ('labelme/_automation/yolov7/checkpoints', 'labelme/_automation/yolov7/checkpoints'),
    ('labelme/_automation/mmpose/checkpoints', 'labelme/_automation/mmpose/checkpoints'),
    ('labelme/_automation/mmdetection/checkpoints', 'labelme/_automation/mmdetection/checkpoints'),
    ('labelme/_automation/torch', 'labelme/_automation/torch'),
]
datas.extend(model_dirs)

# 添加与AI模型相关的隐藏导入
hiddenimports.extend([
    'osam',
    'osam._models',
    'osam._models.yoloworld',
    'osam._models.yoloworld.clip',
    'torch',
    'torchvision',
    'numpy',
    'PIL',
    'mmcv',
    'mmdet',
    'mmpose',
])

# 主程序分析
a = Analysis(
    [str(LABELME_PATH / '__main__.py')],
    pathex=[str(WORK_PATH)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 创建PYZ归档文件
pyz = PYZ(
    a.pure, 
    a.zipped_data,
    cipher=block_cipher
)

# 生成可执行文件
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='labelme',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # 设置为True可以看到错误信息
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(LABELME_PATH / 'icons' / 'icon.ico') if os.path.exists(str(LABELME_PATH / 'icons' / 'icon.ico')) else None,
)

# 收集所有文件到一个目录
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='labelme',
)