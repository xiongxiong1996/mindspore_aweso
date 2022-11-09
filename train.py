# ==========================================
# Copyright 2022  . All Rights Reserved.
# E-mail:shaohuaduan@bjtu.edu.com
# 北京交通大学 计算机与信息技术学院 信息所 MePro 张淳杰 段韶华
# ==========================================
import argparse
import os
import time

import mindspore as ms
from mindspore import nn, Model, LossMonitor, TimeMonitor, ops
from mindspore.ops import reshape
from mindvision.classification import Mnist

from dataset import get_loader
from models import get_model, BaseNet
from mindvision.classification  import resnet50

# 设置参数--------------------------------------------------------------------
parser = argparse.ArgumentParser('MindSpore_Awesome', add_help=False)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--resume', default='', type=str, metavar='path', help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=2e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--data', default='CUB', type=str, help='choice database')
parser.add_argument('--arch', default='resnet50', type=str, help='arch default:resnet50')

# 设置参数--------------------------------------------------------------------
ms.set_context(device_id=0)
config_ck = ms.CheckpointConfig(save_checkpoint_steps=32, keep_checkpoint_max=10)
# 设置dataset固定参数--------------------------------------------------------------------
data_config = {"AIR": [100, "../Data/fgvc-aircraft-2013b"],
               "CAR": [196, "../Data/stanford_cars"],
               "DOG": [120, "../Data/StanfordDogs"],
               "CUB": [200, "/opt/data/private/wenyu/dataset/CUB_200_2011/"],
               }
# 设置dataset固定参数--------------------------------------------------------------------


def main():
    global args  # 定义全局变量args
    args = parser.parse_args()
    # 判断是哪个数据集，并选择参数，待优化---------------------------
    if args.data == "CUB":
        print("load CUB config")
        args.num_classes = data_config['CUB'][0]
        args.data_root = data_config['CUB'][1]
    elif args.data == "AIR":
        print("load AIR config")
        args.num_classes = data_config['AIR'][0]
        args.data_root = data_config['AIR'][1]
    else:
        print("error:unknow dataset")
    # 判断是哪个数据集，并选择参数，待优化---------------------------
    nowtime = time.strftime("%Y-%m-%d-%H", time.localtime())
    exp_dir = nowtime + '_' + args.data + '_'+str(args.batch_size)+'_'+str(args.lr)
    os.makedirs(exp_dir, exist_ok=True)

    # dataloader
    train_dataset = get_loader(args, shuffle=True, train=True)
    eval_dataset = get_loader(args, shuffle=False, train=False)
    # 获取网络
    # net = BaseNet(args)
    net = BaseNet(args)
    # 损失函数定义
    loss_ce = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    # loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')
    # loss =nn.MSELoss()
    # 连接前向网络与损失函数
    net_with_loss = nn.WithLossCell(net, loss_ce)
    # 优化函数定义
    opt = nn.Momentum(params=net.trainable_params(), learning_rate=args.lr, momentum=0.9)
    # 定义训练网络，封装网络和优化器
    train_net = nn.TrainOneStepCell(net_with_loss, opt)
    # 设置网络为训练模式
    train_net.set_train()
    # 实例化模型：其中包含了网络模型，使用的损失函数，使用的优化函数等。
    # model = Model(net, loss_ce, opt, metrics={"Accuracy": nn.Accuracy()})
    # 设置验证网络
    eval_net = nn.WithEvalCell(net, loss_ce)
    eval_net.set_train(False)
    equal = ops.Equal()
    # 调用model.train进行训练
    # 其中包含了训练的epochs，训练数据集等。增加callbacks，用于实现训练过程中的监控。
    # model.train(args.epochs, train_dataset, callbacks=[LossMonitor(0.01, 1875)])
    # model.train(args.epochs, train_ds, callbacks=[TimeMonitor()])


    train_total = 5994
    test_total = 5794
    for epoch in range(args.epochs):
        time_start = time.time()  # 开始计时
        num_correct = [0] * 5
        for d in train_dataset.create_dict_iterator():
            data = d["data"]
            label = d["label"]
            # print(data.shape,label.shape)
            # 未知原因，此处得到的data的维度不正确，多了一个维度。
            bs = data.shape[0]
            data = reshape(data, (bs,3,448,448))
            result = train_net(data, label)
            num_correct[0] += result.sum()
        result_str=(f"Epoch: [{epoch} / {args.epochs}], "
              f"loss: {100. * float(num_correct[0])/train_total}"+"\n")
        print(result_str)
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(result_str)
        time_eclapse = time.time() - time_start
        print('train time:' + str(time_eclapse))  # 输出训练时间

        if epoch < 5 or epoch % 10 == 0:
            time_start = time.time()  # 开始计时
            # 默认情况下就会共享权重了
            # 真正验证迭代过程

            for d in eval_dataset.create_dict_iterator():
                data = d["data"]
                label = d["label"]
                # print(data.shape,label.shape)
                # 位置原因，此处得到的data的维度不正确，多了一个维度。
                bs = data.shape[0]
                data = reshape(data, (bs, 3, 448, 448))
                outputs = eval_net(data, label)
                num_correct[1] += outputs[0].sum()
                # index, value = ops.ArgMaxWithValue()(reshape(outputs[1], (args.num_classes,bs)))
                # transpose(outputs[1])转置
                # test acc 怎么求
                # index, value = ops.ArgMaxWithValue()(outputs[1].transpose())
                # num_correct[2] += equal(index,label).sum()
            # 评估结果
            test_loss = 100. * float(num_correct[1])/test_total
            test_acc = 0.
            test_str= (f"test_loss:  {test_loss}, "
            f"test_acc: {test_acc}"+"\n")
            print(test_str)
            with open(exp_dir + '/results_test.txt', 'a') as file:
                file.write(test_str)
            time_eclapse = time.time() - time_start
            print('test time:' + str(time_eclapse))  # 输出训练时间




if __name__ == '__main__':
    main()
