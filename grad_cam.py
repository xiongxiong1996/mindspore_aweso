# ==========================================
# Copyright 2022  . All Rights Reserved.
# E-mail:shaohuaduan@bjtu.edu.com
# 北京交通大学 计算机与信息技术学院 信息所 MePro 张淳杰 段韶华
# ==========================================
import argparse
import time


import numpy
import mindspore as ms
import numpy as np
from mindspore import nn, Model, LossMonitor, TimeMonitor, ops, PYNATIVE_MODE
from mindspore_xai.explainer import GradCAM,Gradient
from dataset import get_loader
from models import get_model, BaseNet, resnet_mindspore
from mindvision.classification import resnet50, LeNet5

from utils import get_bbox

# 设置参数--------------------------------------------------------------------
parser = argparse.ArgumentParser('MindSpore_Awesome', add_help=False)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--resume', default='', type=str, metavar='path', help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--data', default='CUB', type=str, help='choice database')
parser.add_argument('--arch', default='resnet50', type=str, help='arch default:resnet50')

# 设置参数--------------------------------------------------------------------
ms.set_context(device_id=0)
# config_ck = ms.CheckpointConfig(save_checkpoint_steps=32, keep_checkpoint_max=10)
ms.set_context(mode=PYNATIVE_MODE) # 动态图模式
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
    args.resume = './2022-11-14-02_CUB_32_0.0002/model.ckpt'
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

    # nowtime = time.strftime("%Y-%m-%d-%H", time.localtime())
    # exp_dir = nowtime + '_' + args.data + '_'+str(args.batch_size)+'_'+str(args.lr)
    # os.makedirs(exp_dir, exist_ok=True)

    # dataloader
    train_dataset = get_loader(args, shuffle=False, train=True)
    test_dataset = get_loader(args, shuffle=False, train=False)
    # 获取网络
    model = BaseNet(args)
    if args.resume != "":
        param_dict = ms.load_checkpoint(args.resume)
        param_not_load = ms.load_param_into_net(model, param_dict)
        print(param_not_load)
    gradcam = GradCAM(model, layer="conv_1")


    def cam_loop(dataset):
        '''
        调用接口实现CAM
        '''
        # Define forward function
        # saliency_tensor = ms.Tensor()
        xy_list_all = []
        for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
            data = data.squeeze(1)
            # [N, 3, 224, 224] Tensor
            label = label.astype('int32')
            label = label.squeeze()
            saliency = gradcam(data, label, show=False)
            # 仅getbox
            # if 'saliency_numpy' in vars():
            #     saliency_numpy = np.append(saliency_numpy, saliency.asnumpy(), axis = 0) # numpy.append 需要设置axis，否则会被展平，如果多维的话就需要指定按照哪个维度合并
            #     # 测试
            #     # np.save("saliency.npy", saliency_numpy)
            # else:
            #     saliency_numpy = saliency.asnumpy()
            xy_list = get_bbox(data.shape[0], saliency, rate=0.09)

            xy_list_all.append(xy_list)
            # 测试
            # xy_list_np = np.array(xy_list_all)
            # np.save("xy_list.npy", xy_list_np)
            # xy_list_np = np.load('xy_list.npy', allow_pickle=True)
            # xy_list = xy_list_np.tolist()
            # print(xy_list)
        # np.save("saliency.npy", saliency_numpy)
        xy_list_np = np.array(xy_list_all)
        np.save("xy_list_np.npy", xy_list_np)
        # 读取使用
        # np.load('a.npy')
        # a = a.tolist()


    time_start = time.time()  # 开始计时
    cam_loop(train_dataset)
    time_eclapse = time.time() - time_start
    print('train time:' + str(time_eclapse) + '\n')  # 输出训练时间
    print("Done!")




if __name__ == '__main__':
    main()
