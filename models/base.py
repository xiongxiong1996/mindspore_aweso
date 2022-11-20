import copy
from copy import deepcopy

import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor, Parameter, ops
from mindspore.ops import stop_gradient
from mindspore_xai.explainer import GradCAM

# from mindvision.classification import resnet50
from models import resnet_mindspore

from utils import l2Norm


class BaseNet(nn.Cell):
    def __init__(self, args):
        super(BaseNet, self).__init__()
        # basenet = resnet50(pretrained=True)  # default number_class=1000
        basenet = resnet_mindspore.resnet50(200,False)
        # self.val = Parameter(Tensor(1.0, ms.float32), name="var")
        # resnet_list = list(basenet.cells())
        # list1_1 =resnet_list[0][-2]
        # list1_2 = resnet_list[0][-1]
        # list2 = resnet_list[1]
        # list3 = resnet_list[2]




        resnet = basenet.cells()
        # print(resnet)
        # list1 = list(basenet.cells())[0]
        self.conv_1 = nn.SequentialCell(list(basenet.cells())[:-2])   # resnet去掉pooling层和线性层
        self.aap1 = ops.AdaptiveAvgPool2D(1)
        self.amp1 = ops.AdaptiveMaxPool2D(1)
        self.classifier = nn.Dense(2048, args.num_classes) # 线性层，最后一层用于分类
        # self.cls_max = nn.Dense(2048, args.num_classes)
        # self.cls_cat = nn.Dense(2048+2048, args.num_classes)

        #self.basenet = nn.SequentialCell(list(basenet.cells())[:])
    # 反向传播不是forward 而是 construct
    def construct(self, x, y=None):
        # 获取基础logits
        x1 = self.conv_1(x)
        x2 = self.aap1(x1)
        x2 = x2.squeeze(2)
        x2 = x2.squeeze(2)
        logits = self.classifier(x2)

        # # 获取max_logits
        # mx0 = stop_gradient(x1) # 阻止梯度回传
        # mx1 = self.aap1(mx0)  # amp必须是Int64???
        # mx1 = mx1.squeeze(2)
        # mx1 = mx1.squeeze(2)
        # logits_max = self.cls_max(mx1)
        #
        # # 获取cat_logits
        # # x2 = l2_norm_v2(x2)
        # cx1 = l2Norm(x2)
        # cx1 = stop_gradient(cx1)
        # # cx1 = stop_gradient(l2_norm_v2(x2))
        # cx2 = stop_gradient(mx1)
        # op = ops.Concat(1)
        # cx3 = op((cx1, cx2))
        # logits_cat = self.cls_cat(cx3)

        # Grad_cam & getbox
        # 使用图片输入和标准损失，通过gradcam生成权重，来进行box的获取

        # self.gradcam = GradCAM(self, layer="conv_1")
        # label = ops.Argmax(output_type=ms.int32)(logits)
        # saliency = self.gradcam(x, label, show=False)
        # print(saliency)
        # return logits, logits_max, logits_cat
        return logits

    # get params 获取参数
    def get_params(self, prefix='extractor'):
        extractor_params = list(self.conv5.parameters())
        extractor_params_ids = list(map(id, self.conv5.parameters()))
        classifier_params = filter(lambda p: id(p) not in extractor_params_ids, self.parameters())
        if prefix in ['extractor', 'extract']:
            return extractor_params
        elif prefix in ['classifier']:
            return classifier_params




class CPCNN(nn.Cell):
    def __init__(self, net, args):
        super(CPCNN, self).__init__()
        self.conv = nn.SequentialCell(list(net.cells())[:-1])
        # GradCam定义

        #self.basenet = nn.SequentialCell(list(basenet.cells())[:])
    # 反向传播不是forward 而是 construct
    def construct(self, x, y=None):

        feature = self.conv(x)

        return feature

    # get params 获取参数
    def get_params(self, prefix='extractor'):
        extractor_params = list(self.conv5.parameters())
        extractor_params_ids = list(map(id, self.conv5.parameters()))
        classifier_params = filter(lambda p: id(p) not in extractor_params_ids, self.parameters())
        if prefix in ['extractor', 'extract']:
            return extractor_params
        elif prefix in ['classifier']:
            return classifier_params
