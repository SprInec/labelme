from mmdet.apis import inference_detector, init_detector
from mmpose.structures import merge_data_samples
from mmpose.registry import VISUALIZERS
from mmpose.evaluation.functional import nms
from mmpose.apis import init_model as init_pose_estimator
from mmpose.apis import inference_topdown
from mmengine.registry import init_default_scope
import mmengine
from mmcv import imread
import mmcv
import torch
import cv2
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt


# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

img_path = 'B0001.png'

# detector = init_detector(
#     'labelme/_automation/mmpose/projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py',
#     'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',
#     device=device
# )

# pose_estimator = init_pose_estimator(
#     'labelme/_automation/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py',
#     'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
#     device=device,
#     cfg_options={'model': {'test_cfg': {'output_heatmaps': True}}}
# )

detector = init_detector(
    'labelme/_automation/mmpose/projects/rtmpose/rtmdet/person/rtmdet_m_640-8xb32_coco-person.py',
    'labelme/_automation/mmpose/checkpoints/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',
    device=device
)

pose_estimator = init_pose_estimator(
    'labelme/_automation/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_udp-8xb64-210e_coco-256x192.py',
    'labelme/_automation/mmpose/checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
    device=device,
    cfg_options={'model': {'test_cfg': {'output_heatmaps': True}}}
)

# pose_estimator = init_pose_estimator(
#     'labelme/_automation/mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py',
#     'labelme/_automation/mmpose/checkpoints/rtmpose_m.pth',
#     device=device,
#     cfg_options={'model': {'test_cfg': {'output_heatmaps': True}}}
# )

init_default_scope(detector.cfg.get('default_scope', 'mmdet'))

# 获取目标检测预测结果
detect_result = inference_detector(detector, img_path)

# 置信度阈值
CONF_THRES = 0.5

pred_instance = detect_result.pred_instances.cpu().numpy()
bboxes = np.concatenate(
    (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
bboxes = bboxes[np.logical_and(
    pred_instance.labels == 0, pred_instance.scores > CONF_THRES)]
bboxes = bboxes[nms(bboxes, 0.3)][:, :4]

# 获取每个 bbox 的关键点预测结果
pose_results = inference_topdown(pose_estimator, img_path, bboxes)

# 把多个 bbox 的 pose 结果打包到一起
data_samples = merge_data_samples(pose_results)

idx_point = 13
heatmap = data_samples.pred_fields.heatmaps[idx_point, :, :]

# 索引为 idx 的关键点，在全图上的预测热力图
plt.imshow(heatmap)
plt.show()

# 半径
pose_estimator.cfg.visualizer.radius = 10
# 线宽
pose_estimator.cfg.visualizer.line_width = 8
visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
# 元数据
visualizer.set_dataset_meta(pose_estimator.dataset_meta)

img = mmcv.imread(img_path)
img = mmcv.imconvert(img, 'bgr', 'rgb')

img_output = visualizer.add_datasample(
    'result',
    img,
    data_sample=data_samples,
    draw_gt=False,
    draw_heatmap=False,
    draw_bbox=True,
    show_kpt_idx=True,
    show=False,
    wait_time=0,
    out_file='B0001opt.jpg'
)

plt.figure(figsize=(10, 10))
plt.imshow(img_output)
plt.show()
