from mindspore import ops


def l2Norm(input):
    '''
    l2 归一化 已测试，完全正确
    '''
    input_size = input.shape # 不能用size，要用shape。。
    lp = ops.LpNorm(axis=0, p=2, keep_dims=True)
    _output = input / (lp(input))   # torch.norm 求范数 dim=-1
    output = _output.view(input_size) # 看起来没用
    return output