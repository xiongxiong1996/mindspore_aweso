# from .base import BaseNet
from models.base import BaseNet


def get_model(args):
    net = BaseNet(args)
    return net
