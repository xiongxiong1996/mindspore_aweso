import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor, Parameter, ops
from mindspore.ops import stop_gradient

from mindvision.classification import resnet50, GlobalAvgPooling, ConvNormActivation

from utils import l2Norm


# class WeightNorm(nn.Cell):
#
#     def __init__(self, module, dim: int = 0):
#         super().__init__()
#
#         if dim is None:
#             dim = -1
#
#         self.dim = dim
#         self.module = module
#         self.assign = P.Assign()
#         # add g and v as new parameters and express w as g/||v|| * v
#         self.param_g = Parameter(Tensor(norm_except_dim(self.module.weight, 2, dim)))
#         self.param_v = Parameter(Tensor(self.module.weight.data))
#         self.module.weight.set_data(_weight_norm(self.param_v, self.param_g, self.dim))
#         self.use_weight_norm = True
#
#
#     def construct(self, *inputs, **kwargs):
#         if not self.use_weight_norm:
#             return self.module(*inputs, **kwargs)
#
#         self.assign(self.module.weight, _weight_norm(self.param_v, self.param_g, self.dim))
#         return self.module(*inputs, **kwargs)
#
#
#     def remove_weight_norm(self):
#         self.assign(self.module.weight, _weight_norm(self.param_v, self.param_g, self.dim))
#         self.use_weight_norm = False





class BaseNet(nn.Cell):
    def __init__(self, args):
        super(BaseNet, self).__init__()
        basenet = resnet50(pretrained=True)  # default number_class=1000
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
        self.cls_max = nn.Dense(2048, args.num_classes)
        self.cls_cat = nn.Dense(2048+2048, args.num_classes)
        #self.basenet = nn.SequentialCell(list(basenet.cells())[:])
    # 反向传播不是forward 而是 construct
    def construct(self, x, y=None):
        # 获取基础logits
        x1 = self.conv_1(x)
        x2 = self.aap1(x1)
        x2 = x2.squeeze()
        logits = self.classifier(x2)

        # 获取max_logits
        mx0 = stop_gradient(x1) # 阻止梯度回传
        mx1 = self.amp1(mx0)
        mx1 = mx1.squeeze()
        logits_max = self.cls_max(mx1)

        # 获取cat_logits
        # x2 = l2_norm_v2(x2)
        cx1 = l2Norm(x2)
        cx1 = stop_gradient(cx1)
        # cx1 = stop_gradient(l2_norm_v2(x2))
        cx2 = stop_gradient(mx1)
        op = ops.Concat(1)
        cx3 = op((cx1, cx2))
        logits_cat = self.cls_cat(cx3)
        return logits, logits_max, logits_cat

    # get params 获取参数
    def get_params(self, prefix='extractor'):
        extractor_params = list(self.conv5.parameters())
        extractor_params_ids = list(map(id, self.conv5.parameters()))
        classifier_params = filter(lambda p: id(p) not in extractor_params_ids, self.parameters())
        if prefix in ['extractor', 'extract']:
            return extractor_params
        elif prefix in ['classifier']:
            return classifier_params
