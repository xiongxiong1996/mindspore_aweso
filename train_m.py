# ==========================================
# Copyright 2022  . All Rights Reserved.
# E-mail:shaohuaduan@bjtu.edu.com
# 北京交通大学 计算机与信息技术学院 信息所 MePro 张淳杰 段韶华
# ==========================================
import argparse
import os
import time
import mindspore as ms
from mindspore import nn, ops, PYNATIVE_MODE
from mindspore_xai.explainer import GradCAM

from dataset import get_loader
from models.base import CPCNN, BaseNet
from utils import get_bbox

# 设置网络参数--------------------------------------------------------------------
parser = argparse.ArgumentParser('MindSpore_Awesome', add_help=False)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--resume', default='', type=str, metavar='path', help='path to latest checkpoint (default: none)')
parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--data', default='CUB', type=str, help='choice database')
parser.add_argument('--arch', default='resnet50', type=str, help='arch default:resnet50')

# MindSpore设置--------------------------------------------------------------------
ms.set_context(device_id=0)
# config_ck = ms.CheckpointConfig(save_checkpoint_steps=32, keep_checkpoint_max=10)
ms.set_context(mode=PYNATIVE_MODE)
# dataset固定参数--------------------------------------------------------------------
data_config = {"AIR": [100, "../Data/fgvc-aircraft-2013b"],
               "CAR": [196, "../Data/stanford_cars"],
               "DOG": [120, "../Data/StanfordDogs"],
               "CUB": [200, "/opt/data/private/wenyu/dataset/CUB_200_2011/"],
               }


# 设置dataset固定参数--------------------------------------------------------------------

def main():
    global args  # 定义全局变量args
    args = parser.parse_args()
    args.resume = './2022-11-14-02_CUB_32_0.0002/model.ckpt' # 临时设置
    # 判断是哪个数据集，并选择读取数据的参数，待优化-----------------------------------------
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
    # 创建结果存储文件夹，检查文件是否存在--------------------------------------------------
    nowtime = time.strftime("%Y-%m-%d-%H", time.localtime())
    exp_dir = nowtime + '_' + args.data + '_' + str(args.batch_size) + '_' + str(args.lr)
    os.makedirs(exp_dir, exist_ok=True)

    # dataloader 数据集读取
    train_dataset = get_loader(args, shuffle=True, train=True)
    test_dataset = get_loader(args, shuffle=False, train=False)

    # 获取网络，如果有ckpt文件则进行读取
    model = BaseNet(args)
    if args.resume != "":
        param_dict = ms.load_checkpoint(args.resume)
        param_not_load = ms.load_param_into_net(model, param_dict)
        print(param_not_load) # 这里输出的是没有被加载的层参数，如果都加载了此处输出为空[]

    # 损失函数定义
    # loss_ce = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss_ce = nn.CrossEntropyLoss()

    # 优化器定义
    optimizer = nn.Momentum(params=model.trainable_params(), learning_rate=args.lr, momentum=0.9)
    # optimizer = nn.SGD(model.trainable_params(), learning_rate=args.lr)
    # 定义gradcam
    gradcam = GradCAM(model, layer="conv_1")
    def train_loop(model, dataset, loss_fn, optimizer):
        '''
        单次网络训练
        @param model:使用的网络模型
        @param dataset:训练的数据集
        @param loss_fn:损失函数
        @param optimizer:优化器
        @return:Null
        '''
        def forward_fn(data, label):
            '''
            前向传播计算损失
            @param data: 输入数据
            @param label: 真实标签
            @return:loss,logits 损失及分类结果
            '''
            # logits, logits_max, logits_cat = model(data)
            logits = model(data)
            # loss = loss_fn(logits, label) + loss_fn(logits_max, label) + loss_fn(logits_cat, label)
            loss = loss_fn(logits, label)
            return loss, logits

        # Get gradient function
        grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

        # Define function of one-step training
        def train_step(data, label):
            '''
            训练单步，进行优化
            @param data: 输入数据
            @param label: 真实标签
            @return: loss
            '''
            (loss, logits), grads = grad_fn(data, label)
            loss = ops.depend(loss, optimizer(grads))
            return loss, logits

        def train_box_step(data, label):
            '''
            训练单步，进行优化
            @param data: 输入数据
            @param label: 真实标签
            @return: loss
            '''
            (loss, logits), grads = grad_fn(data, label)
            loss = ops.depend(loss, optimizer(grads))
            return loss, logits

        size = dataset.get_dataset_size() # 获取数据集大小
        model.set_train() # 设置网络模式为训练模式
        resize = nn.ResizeBilinear()
        expand_dims = ops.ExpandDims()
        zeroslike = ops.ZerosLike()
        squeeze = ops.Squeeze(0)
        for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
            # 获取特征层 用于getbox
            data = data.squeeze(1)
            # 数据处理
            label = label.astype('int32')
            # 进行basenet的训练
            loss, logits = train_step(data, label)
            # 使用gradcam 取得CAMmap
            label = ops.Argmax(output_type=ms.int32)(logits)
            saliency = gradcam(data, label, show=False)
            # getbox获取关键区域
            xy_list = get_bbox(data, saliency, rate=0.09)
            # xy_list_int =[]
            # for x in xy_list:
            #     for
            #     xy_list_int.append(int(x)) # 转为int类型
            # print(xy_list_int)
            # 切分出区域部分
            range_data = zeroslike(data)
            for k in range(args.batch_size):
                [x1, x2, y1, y2] = xy_list[k]
                if x1 == x2 or y1 == y2:
                    range_data[k, :, :, :] = data[k, :, :, :]
                else:
                    tmp = data[k, :, y1:y2, x1:x2]
                    tmp = expand_dims(tmp,0)
                    tmp = resize(tmp,(448,448)) # 上采样，必须4维
                    tmp = squeeze(tmp)# 三维才能写入range_data
                    range_data[k,:,:,:] = tmp

            # 训练区域网络
            loss_re, logits_re = train_box_step(range_data, label)


            if batch % 100 == 0:
                loss, loss_re, current = loss.asnumpy(), loss_re.asnumpy(), batch
                train_str = (f"loss: {loss:>7f} loss_re: {loss_re:>7f} [{current:>3d}/{size:>3d}]")
                print(train_str)
                with open(exp_dir + '/results_train.txt', 'a') as file:
                    file.write(train_str)

    def test_loop(model, dataset, loss_fn):
        num_batches = dataset.get_dataset_size()
        model.set_train(False)
        total, test_loss, correct = 0, 0, 0
        for data, label in dataset.create_tuple_iterator():
            data = data.squeeze(1)
            label = label.astype('int32')
            # pred = model(data)
            # logits, logits_max, logits_cat = model(data)
            logits = model(data)
            total += len(data)

            # test_loss += loss_fn(pred, label).asnumpy()
            # correct += (pred.argmax(1) == label).asnumpy().sum()
            # test_loss += (loss_fn(logits, label) + loss_fn(logits_max, label) + loss_fn(logits_cat, label)).asnumpy()
            # pre = logits + logits_max + logits_cat
            test_loss += (loss_fn(logits, label)).asnumpy()
            pre = logits
            correct += (pre.argmax(1) == label).asnumpy().sum()
        test_loss /= num_batches
        correct /= total
        test_str = (f"Test: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        print(test_str)
        with open(exp_dir + '/results_test.txt', 'a') as file:
            file.write(test_str)
        return correct



    max_acc = 0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        # 开始训练
        time_start = time.time()  # 开始计时
        train_loop(model, train_dataset, loss_ce, optimizer)

        time_eclapse = time.time() - time_start
        print('train time:' + str(time_eclapse) + '\n')  # 输出训练时间
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
