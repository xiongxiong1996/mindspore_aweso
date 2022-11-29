import time

import numpy as np
import math
from mindspore import ops, Tensor, nn
import mindspore.common.dtype as mstype
from mindspore.ops import functional as F
from mindspore.ops import operations as P
import mindspore as ms

_default_anchors_setting = (
    dict(layer='p3', stride=32, size=48, scale=[2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
    dict(layer='p4', stride=64, size=96, scale=[2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
    dict(layer='p5', stride=128, size=192, scale=[1, 2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
)

def generate_default_anchor_maps(anchors_setting=None, input_shape=(448, 448)):
    """
    generate default anchor

    :param anchors_setting: all information of anchors
    :param input_shape: shape of input images, e.g. (h, w)
    :return: center_anchors: # anchors * 4 (oy, ox, h, w)
             edge_anchors: # anchors * 4 (y0, x0, y1, x1)
             anchor_area: # anchors * 1 (area)
    """
    if anchors_setting is None:
        anchors_setting = _default_anchors_setting

    center_anchors = np.zeros((0, 4), dtype=np.float32)
    edge_anchors = np.zeros((0, 4), dtype=np.float32)
    anchor_areas = np.zeros((0,), dtype=np.float32)
    input_shape = np.array(input_shape, dtype=int)

    for anchor_info in anchors_setting:
        stride = anchor_info['stride']
        size = anchor_info['size']
        scales = anchor_info['scale']
        aspect_ratios = anchor_info['aspect_ratio']

        output_map_shape = np.ceil(input_shape.astype(np.float32) / stride)
        output_map_shape = output_map_shape.astype(np.int)
        output_shape = tuple(output_map_shape) + (4,)
        ostart = stride / 2.
        oy = np.arange(ostart, ostart + stride * output_shape[0], stride)
        oy = oy.reshape(output_shape[0], 1)
        ox = np.arange(ostart, ostart + stride * output_shape[1], stride)
        ox = ox.reshape(1, output_shape[1])
        center_anchor_map_template = np.zeros(output_shape, dtype=np.float32)
        center_anchor_map_template[:, :, 0] = oy
        center_anchor_map_template[:, :, 1] = ox
        for scale in scales:
            for aspect_ratio in aspect_ratios:
                center_anchor_map = center_anchor_map_template.copy()
                center_anchor_map[:, :, 2] = size * scale / float(aspect_ratio) ** 0.5
                center_anchor_map[:, :, 3] = size * scale * float(aspect_ratio) ** 0.5
                edge_anchor_map = np.concatenate((center_anchor_map[..., :2] - center_anchor_map[..., 2:4] / 2.,
                                                  center_anchor_map[..., :2] + center_anchor_map[..., 2:4] / 2.),
                                                 axis=-1)
                anchor_area_map = center_anchor_map[..., 2] * center_anchor_map[..., 3]
                center_anchors = np.concatenate((center_anchors, center_anchor_map.reshape(-1, 4)))
                edge_anchors = np.concatenate((edge_anchors, edge_anchor_map.reshape(-1, 4)))
                anchor_areas = np.concatenate((anchor_areas, anchor_area_map.reshape(-1)))
    return center_anchors, edge_anchors, anchor_areas

def filter_checkpoint_parameter_by_list(origin_dict, param_filter):
    '''
    用于读取resnet50模型参数时去掉线性层参数
    @param origin_dict: 原始参数
    @param param_filter: 需要去除的参数
    @return:
    '''
    for key in list(origin_dict.keys()):
        for name in param_filter:
            if name in key:
                print("Delete parameter from checkpoint: ", key)
                del origin_dict[key]
                break


def rerange(input, index):
    '''

    @param input:需要重新排序的tensor  N,X,W
    @param dim:排序的维度 2
    @param index:排序索引 N,W 需要广播
    @return:排序完的tensor，和输入同维度。
    '''
    opReshape = ops.Reshape()
    index = opReshape(index, (index.shape[0], 1, index.shape[1]))
    index = F.broadcast_to(index, (index.shape[0], input.shape[1], index.shape[2]))
    output = ops.GatherD()(input, 2, index)
    return output


class SearchTransfer(nn.Cell):
    def __init__(self):
        super(SearchTransfer, self).__init__()
        # 1*1卷积核降维
        self.conv_trans = nn.Conv2d(4096, 2048, 1, stride=1, pad_mode='valid', has_bias=True, weight_init='normal')
        self.flod = nn.Conv2d(18432, 2048, 1, stride=1, pad_mode='valid', has_bias=True, weight_init='normal')
        self.opReshape = ops.Reshape()  # reshape
        self.concat_op = ops.Concat(axis=1)
        self.zeros = ops.Zeros()
        self.resize = nn.ResizeBilinear()
        self.expand_dims = ops.ExpandDims()
        self.transpose = ops.Transpose()
        self.unfold = nn.Unfold(ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='same')
        self.l2_normalize = ops.L2Normalize(axis=1)
        self.batmatmul = ops.BatchMatMul()
        self.argmax = ops.ArgMaxWithValue(axis=1)

    # 反向传播不是forward 而是 construct
    def construct(self, part_ref, part_target):
        '''

        @param part_ref: part_feature_I0
        @param part_target: part_feature_I1
        @return:
        '''
        # 进行unfold
        part_ref_unfold1 = self.unfold(part_ref)  # K=V 维度 N, C*k*k, Hr*Wr
        part_target_unfold = self.unfold(part_target)  # Q 维度 N, C*k*k, H*W
        # 进行reshape,合并最后两个维度,因为mindspore和pytorch的区别
        # K=V   (bs,18432,49)
        part_ref_unfold1 = self.opReshape(part_ref_unfold1,
                                          (part_ref_unfold1.shape[0], part_ref_unfold1.shape[1], -1))
        # Q   (bs,18432,49)
        part_target_unfold = self.opReshape(part_target_unfold,
                                            (part_target_unfold.shape[0], part_target_unfold.shape[1], -1))
        # L2归一化
        part_ref_unfold = self.l2_normalize(part_ref_unfold1)  # N, C*k*k, Hr*Wr
        part_target_unfold = self.l2_normalize(part_target_unfold)  # N, C*k*k, H*W
        # 转置，方便后面相乘 (bs,49,18432)
        part_ref_unfold = self.transpose(part_ref_unfold, (0, 2, 1))  # N,Hr*Wr, C*k*k # 改变顺序
        # 进行相乘,Cross relevance  (bs,49,49)
        R_part = self.batmatmul(part_ref_unfold, part_target_unfold)  # [N, Hr*Wr, H*W]
        # 取最大值 (bs,49) (bs,49)
        max_index, max_value = self.argmax(R_part)  # [N, H*W]  最大值的索引, 最大值
        # (bs,18432,49)
        part_ref_rerang_unflod = rerange(part_ref_unfold1, max_index)
        # 没有flod算子，使用卷积代替，先运行着。  (bs,2048,7,7)
        part_ref_rerang = self.opReshape(part_ref_rerang_unflod,
                                         (part_ref_rerang_unflod.shape[0], part_ref_rerang_unflod.shape[1], 7, 7))
        part_ref_rerang = self.flod(part_ref_rerang)
        # V^和part_features_I1融合
        con_res = self.concat_op((part_ref_rerang, part_target))
        # 维度转换 4096->2048 1*1卷积
        part_res = self.conv_trans(con_res)
        # maxvalue生成Mash bs,1,7,7
        mask = self.opReshape(max_value, (max_value.shape[0], 1, part_ref_rerang.shape[2], part_ref_rerang.shape[3]))
        # part_res 和 mask相乘再和part_feature_I1相加
        part_res = part_res * mask
        part_res = part_res + part_target
        return part_res


class ContextBlock(nn.Cell):
    def __init__(self):
        super(ContextBlock, self).__init__()
        # 1*1卷积核降维
        self.opReshape = ops.Reshape()  # reshape
        self.concat_op = ops.Concat(axis=-1)  # axis=-1

        self.zeros = ops.Zeros()
        self.resize = nn.ResizeBilinear()
        self.expand_dims = ops.ExpandDims()
        self.transpose = ops.Transpose()
        self.l2_normalize = ops.L2Normalize(axis=1)
        self.argmax = ops.ArgMaxWithValue(axis=1)
        self.aap1 = ops.AdaptiveAvgPool2D(1)
        self.aap3 = ops.AdaptiveAvgPool2D(3)
        self.aap5 = ops.AdaptiveAvgPool2D(5)
        self.softmax = ops.Softmax(axis=2)
        self.conv1 = nn.Conv2d(2048, 8192, 1, stride=1, pad_mode='valid', has_bias=True, weight_init='normal')
        self.layernorm = nn.LayerNorm([8192, 7, 7], begin_norm_axis=1,
                                      begin_params_axis=1)  # begin_norm_axis=1, begin_params_axis=1 这两个参数还没搞清楚
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8192, 2048, 1, stride=1, pad_mode='valid', has_bias=True, weight_init='normal')

    # 反向传播不是forward 而是 construct
    def construct(self, x):  # x : part_features_tran
        # psp
        batch, channel, height, width = x.shape
        # (bs*4,2048,1)
        aap1 = self.aap1(x)
        aap1 = self.opReshape(aap1, (aap1.shape[0], aap1.shape[1], -1))
        # (bs*4,2048,9)
        aap3 = self.aap3(x)
        aap3 = self.opReshape(aap3, (aap3.shape[0], aap3.shape[1], -1))
        # (bs*4,2048,25)
        aap5 = self.aap5(x)
        aap5 = self.opReshape(aap5, (aap5.shape[0], aap5.shape[1], -1))
        # (bs*4,2048,35)
        psp_feature = self.concat_op((aap1, aap3, aap5))  # axis = -1
        psp_feature = self.softmax(psp_feature)  # axis =2
        # 将要x  reshape
        input_x = self.opReshape(x, (batch, height * width, channel))
        # x 乘 psp_feature，再乘psp_feature的转置。    x * pf * pf^T
        psp_feature_T = self.opReshape(psp_feature, (psp_feature.shape[0], psp_feature.shape[2], psp_feature.shape[1]))
        # bs*4,49,2048,2048
        context_mask = ops.matmul(psp_feature, psp_feature_T)
        # bs*4,49,2048
        context_mask = ops.matmul(input_x, context_mask)
        # psp_feature转置
        contex = self.opReshape(context_mask, (batch, channel, height, width))
        # channel_add_conv
        # conv2d
        conv_contex = self.conv1(contex)
        conv_contex = self.layernorm(conv_contex)  # layernorm 的两个参数没特别清楚。
        conv_contex = self.relu(conv_contex)
        conv_contex = self.conv2(conv_contex)
        output = x + conv_contex
        return output


class FeatureEnhanceBlock(nn.Cell):
    def __init__(self):
        super(FeatureEnhanceBlock, self).__init__()
        # 1*1卷积核降维
        self.opReshape = ops.Reshape()  # reshape
        self.concat_op = ops.Concat(axis=-1)  # axis=-1
        self.opConcat = ops.Concat(axis=1)
        self.zeros = ops.Zeros()
        self.resize = nn.ResizeBilinear()
        self.expand_dims = ops.ExpandDims()
        self.transpose = ops.Transpose()
        self.l2_normalize = ops.L2Normalize(axis=1)
        self.argmax = ops.ArgMaxWithValue(axis=1)
        self.aap1 = ops.AdaptiveAvgPool2D(1)
        self.softmax = ops.Softmax(axis=1)  # dim=1
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(2048, 128, 3, stride=1, pad_mode='same', has_bias=True, weight_init='normal')
        self.conv2 = nn.Conv2d(128, 128, 3, stride=2, pad_mode='same', has_bias=True, weight_init='normal')
        self.conv3 = nn.Conv2d(128, 128, 3, stride=2, pad_mode='same', has_bias=True, weight_init='normal')
        self.downsample = nn.Conv2d(128, 128, 3, stride=2, pad_mode='same', has_bias=True, weight_init='normal')
        self.tidy1 = nn.Conv2d(128, 6, 1, stride=1, pad_mode='same', has_bias=True, weight_init='normal')
        self.tidy2 = nn.Conv2d(128, 6, 1, stride=1, pad_mode='same', has_bias=True, weight_init='normal')
        self.tidy3 = nn.Conv2d(128, 9, 1, stride=1, pad_mode='same', has_bias=True, weight_init='normal')

    # 反向传播不是forward 而是 construct
    def construct(self, x):  # x : feature bs,2048,14,14
        batch_size = x.shape[0]
        d1 = self.relu(self.conv1(x))
        d2 = self.relu(self.conv2(d1))
        d3 = self.relu(self.conv3(d2))
        d2_1 = self.softmax(self.aap1(d2))  # bs,128,4,1
        e2_1 = d2_1 * d1  # setp1  bs,128,14,14
        d1_final = d1 - e2_1  # setp2 bs,128,14,14
        d2_2 = d2 + self.downsample(e2_1)  # setp3 bs,128,7,7
        d3_1 = self.softmax(self.aap1(d3))  # bs,128,1,1
        e3_1 = d3_1 * d2_2  # step4 bs,128,7,7
        d2_final = d2_2 - e3_1  # step5 bs,128,7,7
        d3_final = d3 + self.downsample(e3_1)  # step6 bs,128,4,4

        t1 = self.tidy1(d1_final)
        t2 = self.tidy2(d2_final)
        t3 = self.tidy3(d3_final)
        t1 = self.opReshape(t1, (batch_size, -1, 1))
        t2 = self.opReshape(t2, (batch_size, -1, 1))
        t3 = self.opReshape(t3, (batch_size, -1, 1))
        output = self.opConcat((t1, t2, t3))
        return output


def l2Norm(input):
    """
    L2归一化
    @param input: 需要归一化的层
    @return: 归一化后的结果
    """
    input_size = input.shape  # 不能用size，要用shape。。
    lp = ops.LpNorm(axis=0, p=2, keep_dims=True)
    _output = input / (lp(input))  # torch.norm 求范数 dim=-1
    output = _output.view(input_size)  # 看起来没用
    return output


def get_bbox(data_shape, cammap, rate=0.001):
    '''
    通过gradecam获得热力图，并获取权重，使用权重来获得box
    :param x: 输入图像 bs*3*448*448
    :param cammap bs*1*448*448
    :param rate:  getbox的rate
    :return: xy_list 一个存放了x, y坐标的list。使用时读取list进行切分即可。
    '''
    reshape = ops.Reshape()  # 定义reshape
    argmax_0 = ops.ArgMaxWithValue(axis=0)  # 定义argmax
    argmin_0 = ops.ArgMinWithValue(axis=0)  # 定义argmin

    # sign = ops.Sign() # 得自己定义sign否则会报错明天做@！！@！！@@！cao
    def sign(input):
        input = input.asnumpy()
        output = np.sign(input)
        output = ms.Tensor(output)
        return output

    # 两次阶跃函数，使得mask中只有0 和1 .有点意思
    # mask = sign(sign(mask-rate)+1)
    mask = sign(sign(cammap - rate) + 1)
    mask = reshape(mask, (mask.shape[0], 1, 448, 448))
    xy_list = []
    for k in range(data_shape):
        indices = mask[k].nonzero()
        indices_numpy = indices.asnumpy()
        if indices_numpy.any():
            min_numpy = indices_numpy.min(axis=0)
            max_numpy = indices_numpy.max(axis=0)
            y1 = min_numpy[-2]
            x1 = min_numpy[-1]
            y2 = max_numpy[-2]
            x2 = max_numpy[-1]
        else:
            y1 = 0
            x1 = 0
            y2 = 447
            x2 = 447
        xy_list.append([int(x1), int(x2), int(y1), int(y2)])
    return xy_list


def get_parts(args, rpn_score, imgs):
    batch_size = imgs.shape[0]
    # 基本变换
    opReshape = ops.Reshape()
    opSort = ops.Sort(descending=True)  # sort
    opSqueeze = P.Squeeze()
    gatherND = ops.GatherNd()
    opConcat_1 = ops.Concat(axis=1)
    opTopk = ops.TopK(sorted=True)
    pad_side = 224  # pad参数
    opPad = ops.Pad(((0, 0), (0, 0), (pad_side, pad_side), (pad_side, pad_side)))
    opZeros = ops.Zeros()
    resize = nn.ResizeBilinear() # 上采样
    opExpand = ops.ExpandDims()
    opSqueeze = ops.Squeeze()
    # nms
    nms = P.NMSWithMask(0.3)
    # select
    select = P.Select()

    # 生成锚点
    _, edge_anchors, _ = generate_default_anchor_maps()  # 生成默认锚点maps
    np_edge_anchors = edge_anchors + 224  # 锚点
    edge_anchors = Tensor(np_edge_anchors, mstype.float32)
    # unchosen_score
    selected_mask_shape = (1614,)
    min_float_num = -65536.0
    unchosen_score = Tensor(min_float_num * np.ones(selected_mask_shape, np.float32),
                            mstype.float32)
    # pad
    x_pad = opPad(imgs)
    part_imgs = opZeros((batch_size * args.topk, 3, 224, 224), mstype.float32)
    topk_indices_all = opZeros((batch_size , args.topk), mstype.int32)
    # 用于存放top_k信息
    for i in range(batch_size):
        rpn_score_current_img = opReshape(rpn_score[i:i + 1:1, ::], (-1, 1))  # 取第i个，并reshape
        bbox_score = opSqueeze(rpn_score_current_img)  # squeeze 去掉为1的维度
        bbox_score_sorted, bbox_score_sorted_indices = opSort(bbox_score)  # 按照降序排序，输出值和索引
        bbox_score_sorted_concat = opReshape(bbox_score_sorted, (-1, 1))  # reshape升维，和extend一样
        # 根据indices描述的索引，提取params上的元素， 重新构建一个tensor   实现edge_anchors排序
        edge_anchors_sorted_concat = gatherND(edge_anchors, opReshape(bbox_score_sorted_indices, (1614, 1)))
        bbox = opConcat_1(
            (edge_anchors_sorted_concat, bbox_score_sorted_concat))  # edge和bbox concat按照维度(1614,5)
        _, _, selected_mask = nms(bbox)  # nms抑制，selected_mask(1614,),全为true或false
        selected_mask = F.stop_gradient(selected_mask)  # 停止梯度计算
        # bbox_score = self.squeezeop(bbox_score_sorted_concat) # 还不如直接用 bbox_score_sorted
        # scores_using = self.select(selected_mask, bbox_score, self.unchosen_score)# 根据mask，看从谁中进行选择。False的被给予了最小的浮点数
        scores_using = select(selected_mask, bbox_score_sorted, unchosen_score)
        # select the topk anchors and scores after nms
        _, topK_indices = opTopk(scores_using, args.topk)  # 找四个最大元素，返回indices
        topk_indices_all[i,:] = topK_indices
        topK_indices = opReshape(topK_indices, (args.topk, 1))
        bbox_topk = gatherND(bbox, topK_indices)  # 按照topk_indices从bbox取出值
        # 生成partimage的list用于取part imgs
        part_list = []
        bbox_topk = bbox_topk.astype("int64")
        # for j in range(bbox_topk.shape[0]):
        #     [y0, x0, y1, x1] = bbox_topk[j, 0:4]
        #     part_list.append([int(y0), int(x0), int(y1), int(x1)])
        # # part imgs
        for k in range(len(part_list)):
            # [y0, x0, y1, x1] = part_list[k]
            [y0, x0, y1, x1] = bbox_topk[k, 0:4]
            part = x_pad[i, :, y0:y1, x0:x1]
            part = opExpand(part, 0)
            part = resize(part, (224, 224))
            part = opSqueeze(part)
            part_imgs[args.topk*i+k, :] = part
    return part_imgs, topk_indices_all
