from mindspore import ops, nn
import mindspore as ms
import mindspore.common.dtype as mstype


opReshape = ops.Reshape()
zeros = ops.Zeros()
opExpand_dims = ops.ExpandDims()
opSum = ops.ReduceSum(keep_dims=False)
relu = ops.ReLU()


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
