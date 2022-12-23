import time

import numpy as np
from mindspore import ops
import mindspore as ms
from mindspore_xai.explainer import GradCAM

from utils import get_part_score, get_ranking_loss, get_bbox


def cam_loop(model, dataset, args):
    """
    网络对训练集数据训练一次
    @param model:网络模型
    @param dataset:训练集
    @param loss_fn:损失函数
    @param optimizer:优化器
    @param args:基础参数
    @return:NULL
    """
    size = dataset.get_dataset_size()  # 获取数据集大小，用于输出监督进度
    grad_cam = GradCAM(model, layer="layer4") # 定义grad_cam
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        time_start = time.time()  # 开始计时
        data = data.squeeze(1)
        label = label.astype('int32')
        # gradcam找到位置
        # /root/anaconda3/envs/mindspore/lib/python3.8/site-packages/mindspore_xai/explainer/backprop
        saliency = grad_cam(data, label)
        xy_list = get_bbox(args.batch_size, saliency, rate=0.2)
        time_eclapse = time.time() - time_start
        print('CAM time:' + str(time_eclapse) + '\n')  # 训练一轮时间

    return xy_list
