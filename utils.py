from mindspore import ops


def l2Norm(input):
    '''
    l2 归一化 需要测试是否正确  LpNorm和torch.norm！！！！！！！！！！！！！！！
    '''
    input_size = input.shape
    lp = ops.LpNorm(axis=0, p=2, keep_dims=True)
    _output = input / (lp(input))   # torch.norm 求范数 dim=-1
    output = _output.view(input_size) # 看起来没用
    return output