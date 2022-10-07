import mindspore.nn as nn
import mindspore.ops as ops

from importlib import import_module

from mindvision.classification import resnet50
from mindvision.classification.models.classifiers import BaseClassifier


class BaseNet(nn.Cell):
    def __init__(self, args):
        super(BaseNet, self).__init__()

        basenet = resnet50(pretrained=True)  # default number_class=1000
        # print(list(basenet.cells())[:-2]) # 去掉最后两层
        self.conv5 = nn.SequentialCell(list(basenet.cells())[:-2])
        self.pool = ops.AdaptiveAvgPool2D(1)
        self.classifier = nn.Dense(2048, args.num_classes) # 线性层，最后一层用于分类

    def forward(self, x, y=None):
        conv5 = self.conv5(x)
        conv5_pool = self.pool(conv5)
        fea = conv5_pool.view(conv5_pool.size(0), -1)
        logits = self.classifier(fea)
        outputs = {'logits': [logits]}
        return outputs

    # get params 获取参数
    def get_params(self, prefix='extractor'):
        extractor_params = list(self.conv5.parameters())
        extractor_params_ids = list(map(id, self.conv5.parameters()))
        classifier_params = filter(lambda p: id(p) not in extractor_params_ids, self.parameters())
        if prefix in ['extractor', 'extract']:
            return extractor_params
        elif prefix in ['classifier']:
            return classifier_params
