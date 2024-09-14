import torch
from torch.autograd.function import Function


class Exp(Function):
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        grad_results = grad_output * result
        return grad_results


x = torch.tensor([1.], requires_grad=True)
y = Exp.apply(x)
print(y)
y.backward()
print(x.grad)
