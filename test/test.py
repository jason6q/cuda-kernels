import torch
from torch.testing._internal.common_utils import TestCase

import j6q_cu_ext

def sample_inputs(device, *, requires_grad=False):
    def make_tensor(*size):
        return torch.randn(size, device=device, requires_grad=requires_grad)

    return [
        [make_tensor(3), make_tensor(3)],
        [make_tensor(1024), make_tensor(1024)]
    ]

    
def reference_add(a,b):
    return a + b

device = torch.device("cuda:0")
samples = sample_inputs(device, requires_grad=False)
for args in samples:
    result = torch.ops.j6q_cu_ext.test(*args)
    expected = reference_add(*args)
    torch.testing.assert_close(result, expected)

    # Check for incorrect operator registration APIs
    torch.library.opcheck(torch.ops.j6q_cu_ext.test.default, args)