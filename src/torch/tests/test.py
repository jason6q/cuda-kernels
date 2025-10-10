import torch
from torch.testing._internal.common_utils import TestCase

import j6q_cu_ext
from utils import sample_1d_inputs

def reference_add(a,b):
    return a + b

if __name__ == '__main__':
    device = torch.device("cuda:0")
    samples = sample_1d_inputs(device, requires_grad=False)
    for args in samples:
        result = torch.ops.j6q_cu_ext.test(*args)
        expected = reference_add(*args)
        torch.testing.assert_close(result, expected)

        # Check for incorrect operator registration APIs
        torch.library.opcheck(torch.ops.j6q_cu_ext.test.default, args)