import torch
from torch import Tensor

__all__ = ["test", "matmul_naive"]

def test(a: Tensor, b:Tensor) -> Tensor:
    """
        Run the test kernel.
    """
    return torch.ops.j6q_cuda_ext.test.default(a,b)

def matmul_naive(a: Tensor, b: Tensor) -> Tensor:
    """
    """
    return torch.ops.j6q_cuda_ext.matmul_naive.default(a,b)
    
# Helps torch know the metadata of the input/outputs of the tensor like shape, stride, etc...
@torch.library.register_fake("j6q_cu_ext::test")
def _(a,b):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)

@torch.library.register_fake("j6q_cu_ext::matmul_naive")
def _(a,b):
    # TODO: Support arbitrary matmul shape.
    torch._check(a.shape[1] == b.shape[0])
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty((a.shape[0], b.shape[1]), device=a.device)