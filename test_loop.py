import numpy as np
import mindspore as ms

from utils import get_part_score, get_ranking_loss


def test_loop(model, dataset, loss_fn, args):
    """
    网络对测试集数据测试一次
    @param model:使用网络模型
    @param dataset:测试集
    @param loss_fn:损失函数
    @param args:基础参数
    @return:correct准确率
    """
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0., 0., 0.
    for data, label in dataset.create_tuple_iterator():
        data = data.squeeze(1)
        label = label.astype('int32')
        logits1_avg, logits2_max, logits3_concat1, logits4_box, logits5_topn, logits6_parts, logits7_transfer \
            , logits8_concat2, logits9_gate = model(data)
        # repeate label
        label_np = label.asnumpy()
        label_repeat = np.repeat(label_np, args.topk)
        label_repeat = ms.Tensor(label_repeat)
        part_score = get_part_score(logits6_parts, label_repeat, args)  # （bs,topk）
        loss1_avg = loss_fn(logits1_avg, label)
        loss2_max = loss_fn(logits2_max, label)
        loss3_concat1 = loss_fn(logits3_concat1, label)
        loss4_box = loss_fn(logits4_box, label)
        loss5_topn = get_ranking_loss(logits5_topn, part_score, args)
        loss6_parts = loss_fn(logits6_parts, label_repeat)
        loss7_transfer = loss_fn(logits7_transfer, label_repeat)
        loss8_concat2 = loss_fn(logits8_concat2, label)
        loss9_gate = loss_fn(logits9_gate, label)

        test_loss += loss1_avg + loss2_max + loss3_concat1 + loss4_box + \
               loss5_topn + loss6_parts + loss7_transfer + loss8_concat2 + loss9_gate
        total += len(data)
        pre = logits9_gate
        correct += (pre.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    test_str = f"Test: \n Accuracy: {correct}%, Avg loss: {test_loss} \n"
    print(test_str)
    with open(args.resultpath + '/results_test.txt', 'a') as file:
        file.write(test_str)
    return correct
