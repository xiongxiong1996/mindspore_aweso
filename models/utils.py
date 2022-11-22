import numpy as np
import math
from mindspore import ops, Tensor, nn
import mindspore.common.dtype as mstype

_default_anchors_setting = (
    dict(layer='p3', stride=32, size=48, scale=[2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
    dict(layer='p4', stride=64, size=96, scale=[2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
    dict(layer='p5', stride=128, size=192, scale=[1, 2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
)


def _fc(in_channel, out_channel):
    '''Weight init for dense cell'''
    stdv = 1 / math.sqrt(in_channel)
    weight = Tensor(np.random.uniform(-stdv, stdv, (out_channel, in_channel)).astype(np.float32))
    bias = Tensor(np.random.uniform(-stdv, stdv, (out_channel)).astype(np.float32))
    return nn.Dense(in_channel, out_channel, has_bias=True,
                    weight_init=weight, bias_init=bias).to_float(mstype.float32)


def _conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, pad_mode='pad'):
    """Conv2D wrapper."""
    shape = (out_channels, in_channels, kernel_size, kernel_size)
    stdv = 1 / math.sqrt(in_channels * kernel_size * kernel_size)
    weights = Tensor(np.random.uniform(-stdv, stdv, shape).astype(np.float32))
    shape_bias = (out_channels,)
    biass = Tensor(np.random.uniform(-stdv, stdv, shape_bias).astype(np.float32))
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     pad_mode=pad_mode, weight_init=weights, has_bias=True, bias_init=biass)


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


class Navigator(nn.Cell):
    """Navigator"""

    def __init__(self):
        """Navigator init"""
        super(Navigator, self).__init__()
        self.down1 = _conv(2048, 128, 3, 1, padding=1, pad_mode='pad')
        self.down2 = _conv(128, 128, 3, 2, padding=1, pad_mode='pad')
        self.down3 = _conv(128, 128, 3, 2, padding=1, pad_mode='pad')
        self.ReLU = nn.ReLU()
        self.tidy1 = _conv(128, 6, 1, 1, padding=0, pad_mode='same')
        self.tidy2 = _conv(128, 6, 1, 1, padding=0, pad_mode='same')
        self.tidy3 = _conv(128, 9, 1, 1, padding=0, pad_mode='same')
        self.opConcat = ops.Concat(axis=1)
        self.opReshape = ops.Reshape()

    def construct(self, x):
        """Navigator construct"""
        batch_size = x.shape[0]
        d1 = self.ReLU(self.down1(x))
        d2 = self.ReLU(self.down2(d1))
        d3 = self.ReLU(self.down3(d2))
        t1 = self.tidy1(d1)
        t2 = self.tidy2(d2)
        t3 = self.tidy3(d3)
        t1 = self.opReshape(t1, (batch_size, -1, 1))
        t2 = self.opReshape(t2, (batch_size, -1, 1))
        t3 = self.opReshape(t3, (batch_size, -1, 1))
        return self.opConcat((t1, t2, t3))


class SearchTransfer(nn.Cell):
    def __init__(self):
        super(SearchTransfer, self).__init__()
        # 1*1卷积核降维
        self.conv_trans = nn.Conv2d(4096, 2048, 1, stride=1, pad_mode='valid', has_bias=True, weight_init='normal')

        self.opReshape = ops.Reshape()  # reshape
        self.concat_op = ops.Concat(axis=1)
        self.zeros = ops.Zeros()
        self.resize = nn.ResizeBilinear()
        self.expand_dims = ops.ExpandDims()
        self.transpose = ops.Transpose()
        self.l2_normalize = ops.L2Normalize()
        self.batmatmul = ops.BatchMatMul()
        self.argmax = ops.ArgMaxWithValue()

    # 反向传播不是forward 而是 construct
    def construct(self, part_ref, part_target):
        unfold = nn.Unfold(ksizes=[1, 3, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='valid')
        # 进行unfold
        part_ref_unfold1 = unfold(part_ref)  # N, C*k*k, Hr*Wr
        part_target_unfold = unfold(part_target)  # N, C*k*k, H*W

        # 进行reshape,合并最后两个维度
        part_ref_unfold1 = self.opReshape(part_ref_unfold1,
                                         (part_ref_unfold1.shape[0], part_ref_unfold1.shape[1], -1))
        part_target_unfold = self.opReshape(part_target_unfold,
                                            (part_target_unfold.shape[0], part_target_unfold.shape[1], -1))
        # L2归一化
        part_ref_unfold = self.l2_normalize(part_ref_unfold1, axis=1) # N, C*k*k, Hr*Wr
        part_target_unfold = self.l2_normalize(part_target_unfold, axis=1)   # N, C*k*k, H*W
        # 转置，方便后面相乘
        part_ref_unfold = self.transpose(part_ref_unfold, (0, 2, 1))  # N,Hr*Wr, C*k*k # 改变顺序
        # 进行相乘
        R_part = self.batmatmul(part_ref_unfold, part_target_unfold)  # [N, Hr*Wr, H*W]
        # 取最大值
        R_part_arg, R_part_star = self.argmax(R_part, axis=1)  # [N, H*W]  最大值的索引, 最大值

        return R_part_arg
