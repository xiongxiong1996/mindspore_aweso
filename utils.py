from matplotlib import transforms
from mindspore import ops, nn, numpy
import mindspore as ms

def l2Norm(input):
    '''
    l2 归一化 已测试，完全正确
    '''
    input_size = input.shape  # 不能用size，要用shape。。
    lp = ops.LpNorm(axis=0, p=2, keep_dims=True)
    _output = input / (lp(input))  # torch.norm 求范数 dim=-1
    output = _output.view(input_size)  # 看起来没用
    return output


def get_bbox(x, cammap, rate=0.001):
    '''
    通过gradecam获得热力图，并获取权重，使用权重来获得box
    :param x: 输入图像 bs*3*448*448
    :param cammap bs*1*448*448
    :param rate:  getbox的rate
    :return: xy_list 一个存放了x, y坐标的list。使用时读取list进行切分即可。
    '''
    # .view(conv_layer.size(0), conv_layer.size(1), 14 * 14)

    resize = nn.ResizeBilinear()
    # mean = ops.ReduceMean(keep_dims=True)
    # expand_dims = ops.ExpandDims()
    # zeroslike = ops.ZerosLike()
    # argmax = ops.ArgMaxWithValue(axis=-1, keep_dims=True)
    # argmin = ops.ArgMinWithValue(axis=-1, keep_dims=True)
    reshape = ops.Reshape()
    argmax_0 = ops.ArgMaxWithValue(axis=0)
    argmin_0 = ops.ArgMinWithValue(axis=0)
    sign = ops.Sign()
    #
    # layer_weights = mean(layer_weights, -1)
    # layer_weights = layer_weights.squeeze(-1)
    # layer_weights = mean(layer_weights, -1)
    # # layer_weights = layer_weights.squeeze(-1)
    # # layer_weights = expand_dims(layer_weights, -1)
    # conv_layer = reshape(conv_layer, (conv_layer.shape[0], conv_layer.shape[1], conv_layer.shape[2] * conv_layer.shape[3]))
    # conv_cam = conv_layer * layer_weights
    # mask = reshape(cammap, (cammap.shape[0], -1))
    # # bs,1       max-min
    # _, mask_min = argmin(mask)
    # _, mask_max = argmax(mask)
    #
    # x_range = mask_max - mask_min
    # # bs,448*448 归一化
    # mask_norm = (mask - mask_min / x_range)
    mask = cammap
    # 两次阶跃函数，使得mask中只有0 和1 .有点意思
    mask = sign(sign(mask-rate)+1)
    mask = reshape(mask,(mask.shape[0], 1, 448, 448))

    # input_box = zeroslike(x)
    xy_list = []
    for k in range(x.shape[0]):
        indices = mask[k].nonzero()
        _, indices_min = argmin_0(indices.astype(ms.float32))
        _, indices_max = argmax_0(indices.astype(ms.float32))
        indices_min = indices_min.astype(ms.int64)
        indices_max = indices_max.astype(ms.int64)
        y1 = indices_min.item(-2)
        x1 = indices_min.item(-1)
        y2 = indices_max.item(-2)
        x2 = indices_max.item(-1)
        # tmp = x[k, :, y1:y2, x1:x2]
        # if x1 == x2 or y1 == y2:
        #     tmp = x[k, :, :, :]
        xy_list.append([x1, x2, y1, y2])
    return xy_list
