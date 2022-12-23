import numpy as np
from mindspore import ops, nn
import mindspore as ms
import mindspore.common.dtype as mstype


opReshape = ops.Reshape()
zeros = ops.Zeros()
opExpand_dims = ops.ExpandDims()
opSum = ops.ReduceSum(keep_dims=False)
relu = ops.ReLU()

def l2Norm(input):
    '''
    l2 归一化 已测试，完全正确
    '''
    input_size = input.shape  # 不能用size，要用shape。。
    lp = ops.LpNorm(axis=0, p=2, keep_dims=True)
    _output = input / (lp(input))  # torch.norm 求范数 dim=-1
    output = _output.view(input_size)
    return output

def get_part_score(logits, targets, args):
    batch_size = int(targets.shape[0] / args.topk)
    log_softmax = nn.LogSoftmax()
    temp = log_softmax(logits)  # (bs*topn,calss_n)
    score = zeros((batch_size * args.topk, 1), mstype.float32)
    for i in range(logits.shape[0]):
        list_item = -temp[i][targets[i]]
        score[i] = list_item
    score = opReshape(score, (batch_size, args.topk))
    return score


def get_ranking_loss(score, part_score, args):
    loss = ms.Tensor(0.)
    batch_size = score.shape[0]
    for i in range(args.topk):
        part_score_i = opExpand_dims(part_score[:, i], 1)
        targets_p = (part_score > part_score_i)
        pivot = opExpand_dims(score[:, i], 1)
        loss_p = (1 - pivot + score) * targets_p
        loss_p = opSum(relu(loss_p))
        loss += loss_p
    loss = loss / batch_size
    return loss


def get_bbox(batch_szie, cammap, rate=0.001):
    '''
    通过gradecam获得热力图，并获取权重，使用权重来获得box
    :param batch_szie: 输入图像的 bs
    :param cammap bs*1*448*448
    :param rate:  getbox的rate
    :return: xy_list 一个存放了x, y坐标的list。使用时读取list进行切分即可。
    '''

    reshape = ops.Reshape() # 定义reshape
    def sign(input):
        input = input.asnumpy()
        output = np.sign(input)
        output = ms.Tensor(output)
        return output
    # 两次阶跃函数，使得mask中只有0 和1 .有点意思
    mask = sign(sign(cammap-rate)+1)
    mask = reshape(mask,(mask.shape[0], 1, 448, 448))
    xy_list = []
    for k in range(batch_szie):
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