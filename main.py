# ==========================================
# Copyright 2022  . All Rights Reserved.
# E-mail:shaohuaduan@bjtu.edu.com
# 北京交通大学 计算机与信息技术学院 信息所 MePro 张淳杰 段韶华
# ==========================================
import argparse
from dataset import get_loader
from models import get_model

parser = argparse.ArgumentParser('MindSpore_Awesome', add_help=False)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--resume', default='', type=str, metavar='path', help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--data', default='CUB', type=str, help='choice database')
parser.add_argument('--arch', default='resnet50', type=str, help='arch default:resnet50')
parser.add_argument('--model', default=0, type=str, help='choice model default:0')  # 选择哪个模型，0为resnet，后续可以再加其它模型
# parser.add_argument('--num_classes', default=0, type=str, help='num_classes')
# parser.add_argument('--data_root', default=0, type=str, help='num_classes')

data_config = {"AIR": [100, "../Data/fgvc-aircraft-2013b"],
               "CAR": [196, "../Data/stanford_cars"],
               "DOG": [120, "../Data/StanfordDogs"],
               "CUB": [200, "/opt/data/private/wenyu/dataset/CUB_200_2011/"],
               }


def main():
    global args  # 定义全局变量args
    args = parser.parse_args()
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
    model = get_model(args)
    # model = nn.DataParallel(model).cuda()
    #
    #   如果args.resume 存在，读取存档点。
    # #

    # dataloader
    iterator = get_loader(args, shuffle=True, train=True)
    print(next(iter(iterator)))

    print("end")


if __name__ == '__main__':
    main()
