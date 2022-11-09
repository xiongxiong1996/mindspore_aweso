import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor, Parameter

from mindvision.classification import resnet50, GlobalAvgPooling, ConvNormActivation


class BaseNet(nn.Cell):
    def __init__(self, args):
        super(BaseNet, self).__init__()
        basenet = resnet50(pretrained=True)  # default number_class=1000
        # self.val = Parameter(Tensor(1.0, ms.float32), name="var")
        self.conv_1 = nn.SequentialCell(list(basenet.cells())[:-3]) # resnet去掉后三层
        self.conv_2 = nn.CellList(list(basenet.cells())[-3]) # resnet倒数第三层
        self.gap = GlobalAvgPooling()
        self.classifier = nn.Dense(2048, args.num_classes) # 线性层，最后一层用于分类
        #self.basenet = nn.SequentialCell(list(basenet.cells())[:])
    # 反向传播不是forward 而是 construct
    def construct(self, x, y=None):
        # res1 = self.conv_1(x)
        # res2 = self.gap(res1)
        # out = self.classifier(res2)
        x1 = self.conv_1(x)
        x2 = self.conv_2(x1)
        x3 = self.gap(x2)
        out = self.classifier(x3)
        return out

    # get params 获取参数
    def get_params(self, prefix='extractor'):
        extractor_params = list(self.conv5.parameters())
        extractor_params_ids = list(map(id, self.conv5.parameters()))
        classifier_params = filter(lambda p: id(p) not in extractor_params_ids, self.parameters())
        if prefix in ['extractor', 'extract']:
            return extractor_params
        elif prefix in ['classifier']:
            return classifier_params
