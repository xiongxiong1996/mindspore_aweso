# ==========================================
# Copyright 2022  Institute of Information Science, Beijing Jiaotong University. All Rights Reserved.
# E-mail:shaohuaduan@bjtu.edu.com
# 北京交通大学 计算机与信息技术学院 信息科学研究所 MePro   张淳杰 段韶华
# ==========================================
import argparse
import os
import time
import mindspore as ms
from examples.common.resnet import resnet50
from mindspore import nn, PYNATIVE_MODE
from mindspore_xai.explainer import GradCAM

from dataset import get_loader
from models.cpnet import CpNet
from test_loop import test_loop
from train_loop import train_loop


# 网络基本参数------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser('MindSpore_Awesome', add_help=False)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch_size', default=12, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--resume', default='', type=str, metavar='path', help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs', default=11, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--topk', default=4, type=int, metavar='N', help='number of topk')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--data', default='CUB', type=str, help='choice database')
parser.add_argument('--backbone', default='resnet50', type=str, help='arch default:resnet50')
parser.add_argument('--resultpath', default='', type=str, metavar='path', help='path to result (No setting required)')

# MindSpore基本参数------------------------------------------------------------------------------------------------------
ms.set_context(device_id=0)
ms.set_context(mode=PYNATIVE_MODE)  # 动态图模式，否则无法使用GradCam
# dataset固定参数设置-----------------------------------------------------------------------------------------------------
data_config = {"AIR": [100, "../Data/fgvc-aircraft-2013b"],
               "CAR": [196, "../Data/stanford_cars"],
               "DOG": [120, "../Data/StanfordDogs"],
               "CUB": [200, "/opt/data/private/wenyu/dataset/CUB_200_2011/"],
               }


# main------------------------------------------------------------------------------------------------------------------
def main():
    global args  # 定义全局变量args
    args = parser.parse_args()
    # args.resume = './2022-11-22-02_CUB_16_0.0002/model.ckpt' # 临时设置
    # 判断数据集，读入固定参数  注：python 并不支持 switch 语句，所以只能用多个if------------------------------------------------
    if args.data == "CUB":
        print("load CUB config")
        args.num_classes = data_config['CUB'][0]
        args.data_root = data_config['CUB'][1]
    elif args.data == "AIR":
        print("load AIR config")
        args.num_classes = data_config['AIR'][0]
        args.data_root = data_config['AIR'][1]
    elif args.data == "CAR":
        print("load CAR config")
        args.num_classes = data_config['CAR'][0]
        args.data_root = data_config['CAR'][1]
    elif args.data == "DOG":
        print("load DOG config")
        args.num_classes = data_config['DOG'][0]
        args.data_root = data_config['DOG'][1]
    else:
        print("error: unknow dataset")
    # 创建result存储文件夹------------------------------------------------------------------------------------------------
    nowtime = time.strftime("%Y%m%d%H%M", time.localtime())
    exp_dir = nowtime + '_' + args.data + '_' + str(args.batch_size) + '_' + str(args.lr)
    os.makedirs(exp_dir, exist_ok=True)
    args.resultpath = exp_dir  # args中加入resultpath
    # 读取数据集---------------------------------------------------------------------------------------------------------
    train_dataset = get_loader(args, shuffle=True, train=True)
    test_dataset = get_loader(args, shuffle=False, train=False)
    # 定义网络模型，若存在resume参数，则读取checkpoint的网络参数---------------------------------------------------------------
    model = CpNet(args)
    for m in model.parameters_and_names():
        print(m)
    # net = resnet50(args.num_classes)
    grad_cam = GradCAM(model, layer="layer4")
    # saliency = grad_cam(boat_image, targets=3, show=False)
    if args.resume != "":
        param_dict = ms.load_checkpoint(args.resume)
        param_not_load = ms.load_param_into_net(model, param_dict)
        print(param_not_load)  # 输出没有被加载参数的层，如果都加载完毕则输出为空 []

    # 定义损失函数--------------------------------------------------------------------------------------------------------
    # loss_ce = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss_ce = nn.CrossEntropyLoss()
    # 定义优化器---------------------------------------------------------------------------------------------------------
    optimizer = nn.Momentum(params=model.trainable_params(), learning_rate=args.lr, momentum=0.9)
    # optimizer = nn.SGD(model.trainable_params(), learning_rate=args.lr)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!# 定义gradcam----------------------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # gradcam = GradCAM(model, layer="layer4")
    # 进行训练及测试------------------------------------------------------------------------------------------------------
    max_acc = 0  # 用于判断当前模型是否最优
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        # 开始训练-------------------------------------------------------------------------------------------------------
        time_start = time.time()  # 开始计时
        train_loop(model, train_dataset, loss_ce, optimizer, args)  # 单次训练
        time_eclapse = time.time() - time_start
        print('train time:' + str(time_eclapse) + '\n')  # 训练一轮时间
        # 开始测试-------------------------------------------------------------------------------------------------------
        if epoch < 3 or epoch % 5 == 0:
            time_start = time.time()  # 开始计时
            correct = test_loop(model, test_dataset, loss_ce, args)  # 单词测试
            time_eclapse = time.time() - time_start
            print('test time:' + str(time_eclapse) + '\n')  # 测试一轮时间
            # 判断是否最优模型，进行存储------------------------------------------------------------------------------------
            if correct > max_acc:
                max_acc = correct
                ms.save_checkpoint(model, exp_dir + '/bestmodel.ckpt')
    print("Done!")


if __name__ == '__main__':
    main()
