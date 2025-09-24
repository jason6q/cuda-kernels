import torch
from torch import Tensor
from torch.library import register_fake, register_autograd

__all__ = ["test", "matmul_naive"]

"""
    Test operator setup
"""

def test(a: Tensor, b:Tensor) -> Tensor:
    """
        Run the test kernel.
    """
    return torch.ops.j6q_cu_ext.test.default(a,b)
    
# Helps torch know the metadata of the input/outputs of the tensor like shape, stride, etc...
@register_fake("j6q_cu_ext::test")
def _(a,b):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)

"""
    Matmul operator setup
"""
def matmul_naive(a: Tensor, b: Tensor) -> Tensor:
    """
    """
    return torch.ops.j6q_cu_ext.matmul_naive.default(a,b)

def matmul_naive_backward(ctx, grad_out):
    x,w,out = ctx.saved_tensors
    a,b = torch.ops.j6q_cu_ext.matmul_naive_backward(grad_out,x,w,out)
    return a, b

def matmul_naive_setup_ctx(ctx, inputs, output):
    # Setup anything we want to use in the backward pass here.
    x, w = inputs
    ctx.save_for_backward(x,w,output)

@register_fake("j6q_cu_ext::matmul_naive")
def _(a,b):
    # TODO: Support arbitrary matmul shape.
    torch._check(a.shape[1] == b.shape[0])
    torch._check(a.dtype == b.dtype)
    torch._check(a.device == b.device)
    return torch.empty((a.shape[0], b.shape[1]), device=a.device)

@register_fake("j6q_cu_ext::matmul_naive_backward")
def _(grad_out, a, b, out):
    return a.new_empty(a.shape), b.new_empty(b.shape)

register_autograd("j6q_cu_ext::matmul_naive", matmul_naive_backward,
                setup_context=matmul_naive_setup_ctx)

"""
    Maxpool2d operator setup
"""

def maxpool2d_naive(a:Tensor, size: int):
    pass

def maxpool2d_naive_backward(ctx, grad_out):
    pass

def maxpool2d_naive_setup_ctx(ctx, inputs, outputs):
    return


@register_fake("j6q_cu_ext::maxpool2d_naive")
def _(a,b):
    # TODO: Support arbitrary matmul shape.
    return torch.empty(a.shape, device=a.device)

@register_fake("j6q_cu_ext::maxpool2d_naive_backward")
def _(a,b):
    return torch.empty(a.shape, device=a.device)

register_autograd("j6q_cu_ext::maxpool2d_naive", maxpool2d_naive_backward, setup_context=maxpool2d_naive_setup_ctx)