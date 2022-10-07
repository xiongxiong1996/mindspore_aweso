#from .base import BaseNet
from models.base import BaseNet


def get_model(args):
    if args.model == 0:
        print("model 0 ")
        net = BaseNet(args)
    elif args.model == 1:
        print("model 1 ")
        net = BaseNet(args)
    else:
        print("model 2 ")
        net = BaseNet(args)
    return net