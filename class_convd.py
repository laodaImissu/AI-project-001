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
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple, _single, _triple

# 定义算子函数
# 支持输入256个数，权重为256x64矩阵
def CIM( input, output, kernel, bias = 0 ):

    cim_call_time = 0
    x = output.shape[0]
    y = output.shape[1]
    z = output.shape[2]

    batch_size = kernel.shape[0] 
    in_channels = kernel.shape[1] 

    # 放满每个计算单元 
    for i in range( batch_size // 64 ):
        for j in range(9):
            move_h = j // 3
            move_w = j % 3
            for h in range(y):
                for w in range(z):
                    # 模拟CIM计算单元
                    output[ i*64:(i+1)*64 ,h, w] += torch.matmul(kernel[ i*64:(i+1)*64 , : , 
                        move_h , move_w ], input[ : ,move_h+h , move_w+w ])
                    cim_call_time +=1
    return output,cim_call_time



# 用矩阵运算实现二维卷积
def matrix_conv2d(input, kernel,bias = 0, stride=1, padding=0, dilation=1, groups=1):

    #print(padding,stride,dilation,groups)
    #print(input.shape)
    cim_call_time = 0
    with open('CIM_call_time.txt', 'r') as file: 
        cim_call_time = file.read()
        if cim_call_time== '':
            cim_call_time = 0
        else :
            cim_call_time = int(cim_call_time)
     
    if isinstance(padding, tuple):
        padding = padding[0]  # 假设 padding 是一个对称的元组 (p, p)

    if isinstance(stride, tuple):
        stride = stride[0]

    # 如果 dilation 是元组，取第一个元素作为值
    if isinstance(dilation, tuple):
        dilation = dilation[0]

    # 如果 groups 是元组，取第一个元素作为值
    if isinstance(groups, tuple):
        groups = groups[0]

    if padding > 0:
        input = F.pad(input, (padding, padding, padding, padding))
    # 计算输出大小
    batch_size, channels, input_h, input_w = input.shape
    kernel_h, kernel_w = kernel.shape[-2:]

    # 考虑扩张
    effective_kernel_h = (kernel_h - 1) * dilation + 1
    effective_kernel_w = (kernel_w - 1) * dilation + 1

    output_h = (math.floor((input_h - effective_kernel_h) / stride) + 1)
    output_w = (math.floor((input_w - effective_kernel_w) / stride) + 1)

    # 初始化输出矩阵
    output = torch.zeros(batch_size, kernel.shape[0], output_h, output_w).to(input.device)

    return_matrix = torch.zeros(output.shape).to(input.device)
    for i in range(batch_size):
        temp_time = 0
        return_matrix[i, :, : , :], temp_time = CIM(input[i, :, : , :], output[i,:,:,:], kernel, bias)
        cim_call_time += temp_time

    with open('CIM_call_time.txt', 'w') as file:
        file.write(str(cim_call_time))

    return return_matrix

class CConv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        #print(weight.shape)
        if self.padding_mode != "zeros":
            return matrix_conv2d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return matrix_conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)