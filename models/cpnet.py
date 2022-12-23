import time

import mindspore.nn as nn
import mindspore as ms
import numpy as np
from mindspore import Tensor, Parameter, ops, load_checkpoint, load_param_into_net
from mindspore.dataset import vision
from mindspore.ops import stop_gradient
from mindspore_xai.explainer import GradCAM
import mindspore.dataset as ds

# from mindvision.classification import resnet50
from models import my_resnet, resnetCam
from mindspore.ops import functional as F
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype

from models.utils import generate_default_anchor_maps, SearchTransfer, filter_checkpoint_parameter_by_list, \
    ContextBlock, FeatureEnhanceBlock, get_parts
from utils import l2Norm, get_bbox


class CpNet(nn.Cell):
    def __init__(self, args):
        super(CpNet, self).__init__()
        # Resnet 预训练
        # my_resnet
        rensnet = my_resnet.resnet50(num_classes=args.num_classes)
        param_dict = ms.load_checkpoint('/opt/data/private/shaohua/PretrainedModel/ResNet_79.46.ckpt')
        param_not_load = ms.load_param_into_net(rensnet, param_dict)
        print(param_not_load)  # 输出没有被加载参数的层，如果都加载完毕则输出为空 []
        # resnetCam
        # resnet_cam = resnetCam.resnet50(num_classes=args.num_classes)
        # param_dict = ms.load_checkpoint('/opt/data/private/shaohua/PretrainedModel/ResNet_79.46.ckpt')
        # param_not_load = ms.load_param_into_net(resnet_cam, param_dict)
        # print(param_not_load)  # 输出没有被加载参数的层，如果都加载完毕则输出为空 []

        # 参数
        self.args = args
        # 常规操作，变换维度，concat等
        self.opReshape = ops.Reshape()  # reshape
        self.opConcat_1 = ops.Concat(axis=1)
        self.opDropout = nn.Dropout(keep_prob=0.5)  # 注：PyTorch中P参数为丢弃参数的概率。 MindSpore中keep_prob参数为保留参数的概率。
        self.opGatherD = ops.GatherD()
        self.opStack = ops.Stack(axis=-1)
        self.opSum = ops.ReduceSum(keep_dims=False)
        # resnet 特征提取
        self.myresnet1 = rensnet
        self.myresnet3 = rensnet
        # self.resnet_cam = resnet_cam


        self.block = my_resnet.ResidualBlock  # 用于构造resnet的layer4, 用于grandcam
        self.layer4 = my_resnet.make_layer(256 * self.block.expansion, self.block, 512, 3, stride=2)
        # 自己定义的卷积
        self.myConv1 = nn.Conv2d(1024, 10 * args.num_classes, 1, 1, pad_mode='pad', padding=1, has_bias=True,
                                 weight_init='normal')
        # 池化层
        # self.aap1 = nn.AdaptiveAvgPool2D(1)  # 全局平均池化。找不到...
        # self.amp1 = nn.AdaptiveMaxPool2D(1)  # 全局最大池化。找不到...
        self.aap1 = ops.AdaptiveAvgPool2D(1)  # 全局平均池化
        self.amp_30 = nn.MaxPool2d(30, 1, pad_mode='valid')  # 最大池化 k=30
        # 线性层
        self.classifier1_avg = nn.Dense(2048, args.num_classes)
        self.classifier2_max = nn.Dense(10 * args.num_classes, args.num_classes)
        self.classifier3_concat1 = nn.Dense(2048 + 10 * args.num_classes, args.num_classes)
        self.classifier4_box = nn.Dense(2048, args.num_classes)
        self.classifier7_transfer = nn.Dense(2048, args.num_classes)
        self.classifier8_concat2 = nn.Dense(2048 + args.topk * 2048, args.num_classes)
        # 特征增强块 FeatureEnhanceBlock
        self.FeatureEnhanceBlock = FeatureEnhanceBlock()
        # context
        self.GlobalContext = ContextBlock()
        # transfer
        self.SearchTransfer1 = SearchTransfer()
        self.SearchTransfer2 = SearchTransfer()
        self.SearchTransfer3 = SearchTransfer()
        self.concat_op_0 = ops.Concat(axis=0)
    # 反向传播 不是forward 而是 construct MindSpore的写法
    def construct(self, x, y=None):  # x(bs,3,448,448)
        # logits1_avg logits2_max logits3_concat1--------------------------------------------------------------time：0.2
        # output_1(bs,200) feature_low_1(bs,1024,28,28) feature1_1(bs,2048,14,14) feature2_1(bs,2048)
        _, feature_low_1, _, _ = self.myresnet1(x)
        batch_size = x.shape[0]
        x1 = self.layer4(feature_low_1)  # (bs,2048,14,14)
        x1 = self.aap1(x1)  # (bs,2048,1,1)
        x1 = self.opReshape(x1, (batch_size, -1))  # (bs,2048)
        logits1_avg = self.classifier1_avg(x1)  # (bs,class_n)
        feature_low_1 = stop_gradient(feature_low_1)  # (bs,1024,28,28)
        x2 = self.myConv1(feature_low_1)  # (bs,10*class_n,30,30)
        x2 = self.amp_30(x2)  # (bs,10*class_n,1,1)
        # x2 = self.opReshape(x2, (x2.shape[0], -1))  # (bs,10*class_n)
        x2 = self.opReshape(x2, (batch_size, -1))  # (bs,10*class_n)
        logits2_max = self.classifier2_max(x2)  # (bs,class_n)
        x1 = stop_gradient(x1)
        x1 = l2Norm(x1)  # (bs,2048)
        x2 = stop_gradient(x2)  # (bs,10*class_n)
        x3 = self.opConcat_1((x1, x2))  # (bs,2048+ 10*class_n)
        logits3_concat1 = self.classifier3_concat1(x3)
        # GradCam getbox -----------------------------------------------------------------------------------------------
        time_start = time.time()  # 开始计时
        input_box = x
        time_eclapse = time.time() - time_start
        print('gradcam time:' + str(time_eclapse) + '\n')  # 训练一轮时间

        # logits_cam = self.resnet_cam(x)
        # grad_cam = GradCAM(self.resnet_cam, layer="layer4")  # 定义grad_cam
        # label = logits_cam.argmax(1)
        # saliency = grad_cam(x, label)
        # xy_list = get_bbox(self.args.batch_size, saliency, rate=0.2)
        # # 裁剪box
        # for i in range(x.shape[0]):
        #     [x1, x2, y1, y2] = xy_list[i]
        #     image = x[i]
        #     # [c,w,h]=image.shape
        #     # image = self.opReshape(image, (w,h,c))
        #     image = image.swapaxes(0, 2)
        #     image = image.swapaxes(0, 1)
        #     image_np = image.asnumpy()
        #     x_p = ds.vision.Crop((x1,y1),(x2-x1,y2-y1))(image_np)
        #     x_p = Tensor(x_p)
        #     # x_p = self.opReshape(x_p, (c, w, h))
        #     x_p = x_p.swapaxes(0, 2)
        #     x_p = x_p.swapaxes(1, 2)
        #     [c, w, h] = x_p.shape
        #     x_p = self.opReshape(x_p,(1,c,w,h))
        #     x_p = ops.interpolate(x_p, None, None, (448, 448), mode="bilinear") # 上采样
        #     [_, c, w, h] = x_p.shape
        #     x_p = self.opReshape(x_p, (c, w, h))
        #     input_box[i,:,:,:] = x_p
        # logits4_box ----------------------------------------------------------------------------------------time：0.16
        # input_box(bs,3,448,448)
        # output_2(bs,200) feature_low_2(bs,1024,28,28) feature1_2(bs,2048,14,14) feature2_2(bs,2048)
        output_2, feature_low_2, feature1_2, _ = self.myresnet1(input_box)
        rpn_feature = feature1_2
        x_box = self.aap1(rpn_feature)
        x_box = self.opDropout(x_box)
        box_feature = x_box
        x_box = self.opReshape(x_box, (batch_size, -1))
        logits4_box = self.classifier4_box(x_box)  # (bs,class_n)
        # FEBlock --------------------------------------------------------------------------------------------time：0.01
        rpn_score = self.FeatureEnhanceBlock(rpn_feature)  # (bs,1614,1)
        # get_parts  logits5_topn ----------------------------------------------------------------------------time：0.25
        part_imgs, topk_index = get_parts(self.args, rpn_score, input_box)  # partimgs bs*topk,3,224,224 topk_index 16*4
        rpn_score = self.opReshape(rpn_score, (batch_size, -1))  # (bs,1614)
        logits5_topn = self.opGatherD(rpn_score, 1, topk_index)  # (bs,topk)
        # parts features  logits6_parts-----------------------------------------------------------------------time：0.16
        # output_3(bs*topk,200)  feature1_3(bs*topk,2048,7,7)
        output_3, _, feature1_3, _ = self.myresnet3(part_imgs)
        logits6_parts = output_3  # (bs*topN,class_n)
        # Transfer & GlobalContext Logits7_transfer-----------------------------------------------------------time：0.02
        [_, c, w, h] = feature1_3.shape
        parts_features = self.opReshape(feature1_3, (batch_size, self.args.topk, c, w, h))  # (bs,topk,2048,7,7)
        # parts_features_transfer = Transfer(parts_features)  # (bs*topk,2048,7,7)  # 耗时1.5m, 最耗时的部分
        part_features_all = parts_features
        part_features_I0 = part_features_all[:, 0, ...]
        part_features_I1 = part_features_all[:, 1, ...]
        part_features_I2 = part_features_all[:, 2, ...]
        part_features_I3 = part_features_all[:, 3, ...]
        S1 = self.SearchTransfer1(part_features_I0, part_features_I1)
        # 跨特征增强
        S2 = self.SearchTransfer2(part_features_I0, part_features_I2)
        # 跨特征增强
        S3 = self.SearchTransfer3(part_features_I0, part_features_I3)
        # 对part imgs进行特征提取
        parts_features_transfer = self.concat_op_0((part_features_I0, S1, S2, S3))
        #  GlobalContext
        transfer_feature = self.GlobalContext(parts_features_transfer)
        transfer_feature = self.aap1(transfer_feature)  # (bs*topk,2048,1,1)
        transfer_feature1 = self.opReshape(transfer_feature, (batch_size * self.args.topk, -1))
        transfer_feature2 = self.opReshape(transfer_feature, (batch_size, -1))
        # transfer_feature1[:,:] = 0.
        # transfer_feature2[:,:] = 0.
        transfer_feature1 = l2Norm(transfer_feature1)
        logits7_transfer = self.classifier7_transfer(transfer_feature1)  # (bs*topk,class_n)
        # Concat all loss Logits8_concat2 -------------------------------------------------------------------time：0.001
        box_feature = self.opReshape(box_feature, (batch_size, -1))
        box_feature = l2Norm(box_feature)
        transfer_feature2 = l2Norm(transfer_feature2)
        concat_feature = self.opConcat_1((box_feature, transfer_feature2))
        logits8_concat2 = self.classifier8_concat2(concat_feature)
        # logits9_gate --------------------------------------------------------------------------------------time：0.001
        logits9_gate = self.opStack(
            [stop_gradient(logits3_concat1), stop_gradient(logits4_box),
             stop_gradient(logits8_concat2)])  # (bs,class_n,3)
        logits9_gate = self.opSum(logits9_gate, -1)  # (bs,class_n)
        return logits1_avg, logits2_max, logits3_concat1, logits4_box, logits5_topn, logits6_parts, logits7_transfer \
            , logits8_concat2, logits9_gate
