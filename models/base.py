import copy
from copy import deepcopy

import mindspore.nn as nn
import mindspore as ms
import numpy as np
from mindspore import Tensor, Parameter, ops, load_checkpoint, load_param_into_net
from mindspore.ops import stop_gradient
from mindspore_xai.explainer import GradCAM

# from mindvision.classification import resnet50
from models import resnet_mindspore, my_resnet
from mindspore.ops import functional as F
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype

from models.utils import Navigator, generate_default_anchor_maps, SearchTransfer, filter_checkpoint_parameter_by_list, \
    ContextBlock, FeatureEnhanceBlock
from utils import l2Norm


class BaseNet(nn.Cell):
    def __init__(self, args):
        super(BaseNet, self).__init__()
        # 设定resnet50,并加载与训练参数
        feature_extractor = my_resnet.resnet50(args.num_classes)
        # 加载预训练模型
        param_dict = load_checkpoint('./resnet50.ckpt')
        # 获取最后一层参数的名字
        filter_list = [x.name for x in feature_extractor.fc.get_parameters()]
        # 删除预训练模型最后一层的参数
        filter_checkpoint_parameter_by_list(param_dict, filter_list)
        load_param_into_net(feature_extractor, param_dict)
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
        # self.block = my_resnet.ResidualBlock
        # self.layer4 = my_resnet.make_layer(256 * self.block.expansion, self.block, 512, 3, stride=2)
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
        self.FeatureEnhanceBlock = FeatureEnhanceBlock() # FeatureEnhanceBlock
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

        self.squeeze = ops.Squeeze(0)
        self.pad_side = 224  # pad参数
        self.Pad_ops = ops.Pad(((0, 0), (0, 0), (self.pad_side, self.pad_side), (self.pad_side, self.pad_side)))

        # self.basenet = nn.SequentialCell(list(basenet.cells())[:])

    # 反向传播不是forward 而是 construct
    def construct(self, x, y=None):
        # 获取feature1_rpn feature2 及logits, 这里和ntsnet中extractfeature对应。
        resnet_logits, _, feature1, feature2 = self.conv_1(x)
        rpn_feature = feature1
        batch_size = x.shape[0]
        rpn_feature = F.stop_gradient(rpn_feature)
        # 获得rpn_score
        # rpn_score = self.navigator(rpn_feature)
        rpn_score = self.FeatureEnhanceBlock(rpn_feature)
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
        # 测试可不可以
        x_pad = self.Pad_ops(x)
        part_imgs = self.zeros((16 * self.topK, 3, 224, 224), mstype.float32)
        for i in range(len(part_list)):
            [y0, x0, y1, x1] = part_list[i]
            part = x_pad[i, :, y0:y1, x0:x1]
            part = self.expand_dims(part, 0)
            part = self.resize(part, (224, 224))
            part = self.squeeze(part)
            part_imgs[i, :] = part
        # 对part imgs进行特征提取
        return resnet_logits, part_list

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

        self.concat_op_0 = ops.Concat(axis=0)

        self.zeros = ops.Zeros()
        self.resize = nn.ResizeBilinear()
        self.expand_dims = ops.ExpandDims()
        # self.basenet = nn.SequentialCell(list(basenet.cells())[:])
        self.part_classifier = nn.Dense(2048, args.num_classes)
        self.trans_classifier = nn.Dense(2048, args.num_classes)
        self.global_context = ContextBlock()
    # 反向传播不是forward 而是 construct
    def construct(self, x, y=None):
        # 获取feature1_rpn feature2 及logits, 这里和ntsnet中extractfeature对应。
        batch_size = self.batch_size
        _, feature_low, feature1, feature2 = self.conv_1(x)
        part_feature = self.aap1(feature1)
        # avgpooling
        part_feature_rank = self.opReshape(part_feature,(part_feature.shape[0],-1))
        # part 基本损失
        part_logits = self.part_classifier(part_feature_rank)
        # part_logits = self.opReshape(part_logits,(self.batch_size, self.topK, -1))

        part_features_all = self.opReshape(feature1,(batch_size,self.topK,feature1.shape[1],feature1.shape[2],feature1.shape[3])) # bs,topN,2048,7,7
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
        part_features_tran = self.concat_op_0((part_features_I0, S1, S2, S3), )
        # 进行全局上下文
        part_features_tran = self.global_context(part_features_tran)
        # avgpooling
        global_features = self.aap1(part_features_tran)
        part_features = self.opReshape(global_features, (global_features.shape[0], -1))
        part_features = self.opReshape(part_features, (self.batch_size* self.topK, -1))
        # transfer后的part损失
        trans_logits = self.trans_classifier(l2Norm(part_features))
        # part的trans_logits
        # trans_logits = self.opReshape(trans_logits, (self.batch_size, self.topK, -1))
        # 少一个part_logits和boxx的feature的融合。
        # ！！！！！！！！！#
        return part_logits, trans_logits
