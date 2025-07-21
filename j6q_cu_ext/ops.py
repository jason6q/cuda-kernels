import torch
from torch import Tensor

__all__ = ["test"]

def test(a: Tensor, b:Tensor) -> Tensor:
    """
        Run the test kernel.
    """
    return torch.ops.j6q_cuda_ext.test.default(a,b)
    
@torch.library.register_fake("j6q_cu_ext::test")
def _(a,b):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)