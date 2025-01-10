import math
from typing import List, Optional, Tuple, Union
from typing_extensions import deprecated

import torch
from torch import nn
from torch import Tensor
from torch._torch_docs import reproducibility_notes
from torch.nn import functional as F, init
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.parameter import Parameter, UninitializedParameter

from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple, _single, _triple


def Linear_CIM(input, weight, bias=None):

    # print(input.shape)
    # print(weight.shape)

    height, width = input.shape
    output_features = weight.shape[1]

    num_vec = width // 256

    # 初始化输出矩阵
    output = torch.zeros(height, output_features, device=input.device)

    # 将权重矩阵分成小块
    num_h = weight.shape[0] // 256 # 遍历次数
    num_w = weight.shape[1] // 64
    input_weight = torch.zeros(256, 64, device=input.device)
    input_vec = torch.zeros(1, 256, device=input.device)

    # 遍历每个权重块
    for i in range(num_h):
        for j in range(num_w):
            # 提取输入的相应部分
            input_weight = weight[ i*256:(i+1)*256 , j*64:(j+1)*64 ]
            # 遍历输入
            for h in range(height):
                for w in range(num_vec):
                    input_vec = input[ h, w*256:(w+1)*256 ]
                    # 执行矩阵乘法，模拟计算单元
                    output[ h , j*64:(j+1)*64] += torch.matmul(input_vec, input_weight)

    # 如果存在偏置，添加偏置
    if bias is not None:
        output += bias

    return output

class Linear(Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
    # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
    # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
    # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # 应用线性变换
        # return torch.matmul(input, self.weight.T) + self.bias if self.bias is not None else torch.matmul(input, self.weight.T)
        return Linear_CIM(input, self.weight.T, self.bias)

