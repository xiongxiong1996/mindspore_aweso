# ==========================================
# Copyright 2022  . All Rights Reserved.
# E-mail:shaohuaduan@bjtu.edu.com
# 北京交通大学 计算机与信息技术学院 信息所 MePro 张淳杰 段韶华
# ==========================================
import argparse
import os
import time

import mindspore as ms
from mindspore import nn, Model, LossMonitor, TimeMonitor, ops, PYNATIVE_MODE
from mindspore.ops import reshape
from mindvision.classification import Mnist

from dataset import get_loader
from models import get_model, BaseNet
from mindvision.classification  import resnet50

# 设置参数--------------------------------------------------------------------
parser = argparse.ArgumentParser('MindSpore_Awesome', add_help=False)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=2, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--resume', default='', type=str, metavar='path', help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--data', default='CUB', type=str, help='choice database')
parser.add_argument('--arch', default='resnet50', type=str, help='arch default:resnet50')

# 设置参数--------------------------------------------------------------------
ms.set_context(device_id=0)
# config_ck = ms.CheckpointConfig(save_checkpoint_steps=32, keep_checkpoint_max=10)
ms.set_context(mode=PYNATIVE_MODE)
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
    # args.resume = './2022-11-08-14_CUB_16_0.0002/model.ckpt'
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
    test_dataset = get_loader(args, shuffle=False, train=False)
    # 获取网络
    model = BaseNet(args)
    if args.resume != "":
        param_dict = ms.load_checkpoint(args.resume)
        param_not_load = ms.load_param_into_net(model, param_dict)
        print(param_not_load)
    # 损失函数定义
    # loss_ce = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss_ce = nn.CrossEntropyLoss()
    # 优化器定义
    optimizer = nn.Momentum(params=model.trainable_params(), learning_rate=args.lr, momentum=0.9)
    # optimizer = nn.SGD(model.trainable_params(), learning_rate=args.lr)

    def train_loop(model, dataset, loss_fn, optimizer):
        # Define forward function
        def forward_fn(data, label):
            logits = model(data)
            loss = loss_fn(logits, label)
            return loss, logits

        # Get gradient function
        grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

        # Define function of one-step training
        def train_step(data, label):
            (loss, _), grads = grad_fn(data, label)
            loss = ops.depend(loss, optimizer(grads))
            return loss

        size = dataset.get_dataset_size()
        model.set_train()
        for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
            data = data.squeeze()
            label = label.astype('int32')
            loss = train_step(data, label)

            if batch % 100 == 0:
                loss, current = loss.asnumpy(), batch
                train_str =(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")
                print(train_str)
                with open(exp_dir + '/results_train.txt', 'a') as file:
                    file.write(train_str)

    def test_loop(model, dataset, loss_fn):
        num_batches = dataset.get_dataset_size()
        model.set_train(False)
        total, test_loss, correct = 0, 0, 0
        for data, label in dataset.create_tuple_iterator():
            data = data.squeeze()
            label = label.astype('int32')
            pred = model(data)
            total += len(data)
            test_loss += loss_fn(pred, label).asnumpy()
            correct += (pred.argmax(1) == label).asnumpy().sum()
        test_loss /= num_batches
        correct /= total
        test_str = (f"Test: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        print(test_str)
        with open(exp_dir + '/results_test.txt', 'a') as file:
            file.write(test_str)
        return  correct

    max_acc=0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        # 开始训练
        time_start = time.time()  # 开始计时
        train_loop(model, train_dataset, loss_ce, optimizer)
        time_eclapse = time.time() - time_start
        print('train time:' + str(time_eclapse)+'\n')  # 输出训练时间
        # 开始测试
        if epoch < 3 or epoch % 5 == 0:
            time_start = time.time()  # 开始计时
            correct = test_loop(model, test_dataset, loss_ce)
            time_eclapse = time.time() - time_start
            print('test time:' + str(time_eclapse) + '\n')  # 输出测试时间
            if correct > max_acc:
                max_acc = correct
                ms.save_checkpoint(model, exp_dir + '/model.ckpt')
    print("Done!")


    # train_total = 5994
    # test_total = 5794
    # for epoch in range(args.epochs):
    #     time_start = time.time()  # 开始计时
    #     num_correct = [0] * 5
    #     for d in train_dataset.create_dict_iterator():
    #         data = d["data"]
    #         label = d["label"]
    #         # print(data.shape,label.shape)
    #         # 未知原因，此处得到的data的维度不正确，多了一个维度。
    #         bs = data.shape[0]
    #         data = reshape(data, (bs,3,448,448))
    #         result = train_net(data, label)
    #         num_correct[0] += result.sum()
    #     result_str=(f"Epoch: [{epoch} / {args.epochs}], "
    #           f"loss: {100. * float(num_correct[0])/train_total}"+"\n")
    #     print(result_str)
    #     with open(exp_dir + '/results_train.txt', 'a') as file:
    #         file.write(result_str)
    #     time_eclapse = time.time() - time_start
    #     print('train time:' + str(time_eclapse))  # 输出训练时间
    #
    #     if epoch < 5 or epoch % 10 == 0:
    #         time_start = time.time()  # 开始计时
    #         # 默认情况下就会共享权重了
    #         # 真正验证迭代过程
    #
    #         for d in eval_dataset.create_dict_iterator():
    #             data = d["data"]
    #             label = d["label"]
    #             # print(data.shape,label.shape)
    #             # 位置原因，此处得到的data的维度不正确，多了一个维度。
    #             bs = data.shape[0]
    #             data = reshape(data, (bs, 3, 448, 448))
    #             outputs = eval_net(data, label)
    #             num_correct[1] += outputs[0].sum()
    #             # index, value = ops.ArgMaxWithValue()(reshape(outputs[1], (args.num_classes,bs)))
    #             # transpose(outputs[1])转置
    #             # test acc 怎么求
    #             # index, value = ops.ArgMaxWithValue()(outputs[1].transpose())
    #             # num_correct[2] += equal(index,label).sum()
    #         # 评估结果
    #         test_loss = 100. * float(num_correct[1])/test_total
    #         test_acc = 0.
    #         test_str= (f"test_loss:  {test_loss}, "
    #         f"test_acc: {test_acc}"+"\n")
    #         print(test_str)
    #         with open(exp_dir + '/results_test.txt', 'a') as file:
    #             file.write(test_str)
    #         time_eclapse = time.time() - time_start
    #         print('test time:' + str(time_eclapse))  # 输出训练时间




if __name__ == '__main__':
    main()
