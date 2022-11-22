import copy
from copy import deepcopy

import mindspore.nn as nn
import mindspore as ms
import numpy as np
from mindspore import Tensor, Parameter, ops
from mindspore.ops import stop_gradient
from mindspore_xai.explainer import GradCAM

# from mindvision.classification import resnet50
from models import resnet_mindspore, my_resnet
from mindspore.ops import functional as F
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype

from models.utils import Navigator, generate_default_anchor_maps, SearchTransfer
from utils import l2Norm


class BaseNet(nn.Cell):
    def __init__(self, args):
        super(BaseNet, self).__init__()
        # basenet = resnet50(pretrained=True)  # default number_class=1000
        # basenet = resnet_mindspore.resnet50(200,False)
        # self.val = Parameter(Tensor(1.0, ms.float32), name="var")
        # resnet_list = list(basenet.cells())
        # list1_1 =resnet_list[0][-2]
        # list1_2 = resnet_list[0][-1]
        # list2 = resnet_list[1]
        # list3 = resnet_list[2]

        # resnet = basenet.cells()
        # print(resnet)
        # list1 = list(basenet.cells())[0]
        # self.conv_1 = nn.SequentialCell(list(basenet.cells())[:-2])   # resnet去掉pooling层和线性层
        # resnet50 去掉layer4和后面的pooling层及线性层
        self.conv_1 = my_resnet.resnet50(args.num_classes)
        # resnet50的layer4
        self.block = my_resnet.ResidualBlock
        self.layer4 = my_resnet.make_layer(256 * self.block.expansion, self.block, 512, 3, stride=2)
        # 全局平均池化
        self.aap1 = ops.AdaptiveAvgPool2D(1)
        # 全局最大池化
        self.amp1 = ops.AdaptiveMaxPool2D(1)
        # 线性层1
        self.classifier = nn.Dense(2048, args.num_classes)
        # self.cls_max = nn.Dense(2048, args.num_classes)
        # self.cls_cat = nn.Dense(2048+2048, args.num_classes)

        # self.basenet = nn.SequentialCell(list(basenet.cells())[:])

    # 反向传播不是forward 而是 construct
    def construct(self, x, y=None):
        # 获取基础logits
        # x1 = self.conv_1(x)
        output, x, feature1, feature2 = self.conv_1(x)
        # feature = self.conv(x)
        x = self.layer4(x)
        x = self.aap1(x)
        x = x.squeeze(2)
        x = x.squeeze(2)
        logits = self.classifier(x)

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
    def __init__(self, args):
        super(CPCNN, self).__init__()

        self.conv_1 = my_resnet.resnet50(args.num_classes)
        # resnet50的layer4
        self.block = my_resnet.ResidualBlock
        self.layer4 = my_resnet.make_layer(256 * self.block.expansion, self.block, 512, 3, stride=2)
        # 全局平均池化
        self.aap1 = ops.AdaptiveAvgPool2D(1)
        # 全局最大池化
        self.amp1 = ops.AdaptiveMaxPool2D(1)
        # 线性层1
        self.classifier = nn.Dense(2048, args.num_classes)
        # self.cls_max = nn.Dense(2048, args.num_classes)
        # self.cls_cat = nn.Dense(2048+2048, args.num_classes)
        # NTS-Net
        self.topK = args.topk
        self.navigator = Navigator()  # navigator
        _, edge_anchors, _ = generate_default_anchor_maps()  # 生成默认锚点maps
        self.np_edge_anchors = edge_anchors + 224  # 锚点
        self.edge_anchors = Tensor(self.np_edge_anchors, mstype.float32)
        self.opReshape = ops.Reshape()  # reshape
        self.squeezeop = P.Squeeze()  # squeeze
        self.sortop = ops.Sort(descending=True)  # sort descending 为True，则根据value对元素进行降序排序,默认为False，升序排序
        self.gatherND = ops.GatherNd()
        self.concat_op = ops.Concat(axis=1)
        self.nms = P.NMSWithMask(0.3)  # NMS最大抑制
        self.min_float_num = -65536.0
        self.selected_mask_shape = (1614,)
        self.unchosen_score = Tensor(self.min_float_num * np.ones(self.selected_mask_shape, np.float32),
                                     mstype.float32)
        self.select = P.Select()
        self.topK_op = ops.TopK(sorted=True)
        self.zeros = ops.Zeros()
        self.resize = nn.ResizeBilinear()
        self.expand_dims = ops.ExpandDims()

        # self.basenet = nn.SequentialCell(list(basenet.cells())[:])

    # 反向传播不是forward 而是 construct
    def construct(self, x, y=None):
        # 获取feature1_rpn feature2 及logits, 这里和ntsnet中extractfeature对应。
        output, x1, feature1, feature2 = self.conv_1(x)
        # layer4
        x1 = self.layer4(x1)
        rpn_feature = x1
        # aap
        x1 = self.aap1(x1)
        x1 = x1.squeeze(2)
        x1 = x1.squeeze(2)
        feature2 = x1
        logits = self.classifier(x1)

        batch_size = x.shape[0]
        rpn_feature = F.stop_gradient(rpn_feature)
        # 获得rpn_score
        rpn_score = self.navigator(rpn_feature)
        # 获得anchors
        edge_anchors = self.edge_anchors
        # 用于存放top_k信息
        for i in range(batch_size):
            rpn_score_current_img = self.opReshape(rpn_score[i:i + 1:1, ::], (-1, 1))  # 取第i个，并reshape
            bbox_score = self.squeezeop(rpn_score_current_img)  # squeeze 去掉为1的维度
            bbox_score_sorted, bbox_score_sorted_indices = self.sortop(bbox_score)  # 按照降序排序，输出值和索引
            bbox_score_sorted_concat = self.opReshape(bbox_score_sorted, (-1, 1))  # reshape升维，和extend一样
            edge_anchors_sorted_concat = self.gatherND(edge_anchors,
                                                       self.opReshape(bbox_score_sorted_indices, (1614,
                                                                                                  1)))  # 根据indices描述的索引，提取params上的元素， 重新构建一个tensor   实现edge_anchors排序
            bbox = self.concat_op(
                (edge_anchors_sorted_concat, bbox_score_sorted_concat))  # edge和bbox concat按照维度(1614,5)
            _, _, selected_mask = self.nms(bbox)  # nms抑制，selected_mask(1614,),全为true或false
            selected_mask = F.stop_gradient(selected_mask)  # 停止梯度计算
            # bbox_score = self.squeezeop(bbox_score_sorted_concat) # 还不如直接用 bbox_score_sorted
            # scores_using = self.select(selected_mask, bbox_score, self.unchosen_score)# 根据mask，看从谁中进行选择。False的被给予了最小的浮点数
            scores_using = self.select(selected_mask, bbox_score_sorted, self.unchosen_score)
            # select the topk anchors and scores after nms
            _, topK_indices = self.topK_op(scores_using, self.topK)  # 找四个最大元素，返回indices
            topK_indices = self.opReshape(topK_indices, (self.topK, 1))
            bbox_topk = self.gatherND(bbox, topK_indices)  # 按照topk_indices从bbox取出值
            # 生成partimage的list用于取part imgs
            part_list = []
            bbox_topk = bbox_topk.astype("int64")
            for i in range(bbox_topk.shape[0]):
                [y0, x0, y1, x1] = bbox_topk[i, 0:4]
                part_list.append([int(y0), int(x0), int(y1), int(x1)])

        # 对part imgs进行特征提取

        return logits, bbox_topk, part_list

    # get params 获取参数
    def get_params(self, prefix='extractor'):
        extractor_params = list(self.conv5.parameters())
        extractor_params_ids = list(map(id, self.conv5.parameters()))
        classifier_params = filter(lambda p: id(p) not in extractor_params_ids, self.parameters())
        if prefix in ['extractor', 'extract']:
            return extractor_params
        elif prefix in ['classifier']:
            return classifier_params


class partNet(nn.Cell):
    def __init__(self, args):
        super(partNet, self).__init__()

        self.conv_1 = my_resnet.resnet50(args.num_classes)
        # 全局平均池化
        self.aap1 = ops.AdaptiveAvgPool2D(1)
        # 全局最大池化
        self.amp1 = ops.AdaptiveMaxPool2D(1)
        # 线性层1
        self.classifier = nn.Dense(2048, args.num_classes)
        # Transfer
        self.SearchTransfer1 = SearchTransfer()
        self.SearchTransfer2 = SearchTransfer()
        self.SearchTransfer3 = SearchTransfer()
        # NTS-Net
        self.topK = args.topk
        self.batch_size = args.batch_size
        self.opReshape = ops.Reshape()  # reshape
        self.squeezeop = P.Squeeze()  # squeeze

        self.concat_op = ops.Concat(axis=1)

        self.zeros = ops.Zeros()
        self.resize = nn.ResizeBilinear()
        self.expand_dims = ops.ExpandDims()

        # self.basenet = nn.SequentialCell(list(basenet.cells())[:])

    # 反向传播不是forward 而是 construct
    def construct(self, x, y=None):
        # 获取feature1_rpn feature2 及logits, 这里和ntsnet中extractfeature对应。
        batch_size = self.batch_size
        output, feature_low, feature1, feature2 = self.conv_1(x)
        part_feature = self.aap1(feature1)
        part_feature_rank = self.squeezeop(part_feature,-1)  # bs*topk,2048

        part_features_all = self.opReshape(feature1,(batch_size,self.topK,-1)) # bs,topN,2048,7,7
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
        output = output[0:batch_size-1,:] # 只是为了能跑
        return output
