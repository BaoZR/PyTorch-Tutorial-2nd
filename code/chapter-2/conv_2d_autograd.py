import torch
from torch.autograd.function import once_differentiable
import torch.nn.functional as F


def convolution_backward(grad_out, X, weight):
    """
    将反向传播功能用函数包装起来，返回的参数个数与forward接收的参数个数保持一致，为2个
    """
    grad_input = F.conv2d(X.transpose(0, 1), grad_out.transpose(0, 1)).transpose(0, 1)
    grad_X = F.conv_transpose2d(grad_out, weight)
    return grad_X, grad_input

class MyConv2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight):
        ctx.save_for_backward(X, weight)

        # ============== step1: 函数功能实现 ==============
        ret = F.conv2d(X, weight)
        # ============== step1: 函数功能实现 ==============
        return ret

    @staticmethod
    def backward(ctx, grad_out):
        X, weight = ctx.saved_tensors
        return convolution_backward(grad_out, X, weight)











