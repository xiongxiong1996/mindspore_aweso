import time

import numpy as np
from mindspore import ops
import mindspore as ms
from mindspore_xai.explainer import GradCAM

from utils import get_part_score, get_ranking_loss, get_bbox


def train_loop(model, dataset, loss_fn, optimizer, args):
    """
    网络对训练集数据训练一次
    @param model:网络模型
    @param dataset:训练集
    @param loss_fn:损失函数
    @param optimizer:优化器
    @param args:基础参数
    @return:NULL
    """
    def forward_fn(data, label):
        """
        反向传播
        @param data: 输入数据
        @param label: 真实标签
        @return:loss,logits 损失及分类结果
        """
        logits1_avg, logits2_max, logits3_concat1, logits4_box, logits5_topn, logits6_parts, logits7_transfer \
            , logits8_concat2, logits9_gate = model(data)
        # repeate label
        label_np = label.asnumpy()
        label_repeat = np.repeat(label_np, args.topk)
        label_repeat = ms.Tensor(label_repeat)
        part_score = get_part_score(logits6_parts, label_repeat, args)  # （bs,topk）
        loss1_avg = loss_fn(logits1_avg, label)
        loss2_max = loss_fn(logits2_max, label)
        loss3_concat1 = loss_fn(logits3_concat1, label)
        loss4_box = loss_fn(logits4_box, label)
        loss5_topn = get_ranking_loss(logits5_topn, part_score, args)
        loss6_parts = loss_fn(logits6_parts, label_repeat)
        loss7_transfer = loss_fn(logits7_transfer, label_repeat)
        loss8_concat2 = loss_fn(logits8_concat2, label)
        loss9_gate = loss_fn(logits9_gate, label)

        logits = logits9_gate
        loss = loss1_avg + loss2_max + loss3_concat1 + loss4_box + \
               loss5_topn + loss6_parts + loss7_transfer + loss8_concat2 + loss9_gate
        # loss1_avg, loss2_max,loss3_concat1,loss4_box,loss5_topn,loss6_parts,loss7_transfer,loss8_concat2,loss9_gate = \
        #     loss1_avg.asnumpy(),loss2_max.asnumpy(),loss3_concat1.asnumpy(),loss4_box.asnumpy(),loss5_topn.asnumpy(),\
        #     loss6_parts.asnumpy(),loss7_transfer.asnumpy(),loss8_concat2.asnumpy(),loss9_gate.asnumpy()
        # loss_str = f"Train: \n loss1_avg: {loss1_avg} \nloss2_max: {loss2_max} \n" \
        #            f"loss3_concat1: {loss3_concat1} \nloss4_box: {loss4_box} \n" \
        #            f"loss5_topn: {loss5_topn} \nloss6_parts: {loss6_parts} \n" \
        #            f"loss7_transfer: {loss7_transfer} \nloss8_concat2: {loss8_concat2} \n" \
        #            f"loss9_gate: {loss9_gate} \n "
        # print(loss_str)

        return loss, logits

    # grad_fn
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(data, label):
        """
        单步训练，优化
        @param data: 输入数据
        @param label: 真实标签
        @return: loss
        """
        (loss, logits), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss, logits

    size = dataset.get_dataset_size()  # 获取数据集大小，用于输出监督进度
    model.set_train()  # 设置网络模式为训练模式
    time_start = time.time()  # 开始计时
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        data = data.squeeze(1)
        label = label.astype('int32')
        loss, logits = train_step(data, label)
        if batch % 100 == 0:
            time_eclapse = time.time() - time_start
            print('train time:' + str(time_eclapse) + '\n')  # 训练一轮时间
            time_start = time.time()  # 重新开始计时
            loss, current = loss.asnumpy(), batch
            train_str = f"Train: \n loss: {loss:>7f} [{current:>3d}/{size:>3d}] \n"
            print(train_str)  # 监控进度
            with open(args.resultpath + '/results_train.txt', 'a') as file:  # 存入文件
                file.write(train_str)
