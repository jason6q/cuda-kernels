import torch

from torch.library import opcheck
from torch.autograd import gradcheck
from torch.testing import assert_close
import j6q_cu_ext # This must be imported.

from utils import sample_2d_inputs

def reference_matmul(a,b):
    return a @ b

if __name__ == '__main__':
    device = torch.device("cuda:0")
    samples = sample_2d_inputs(device)
    for args in samples:
        opcheck(torch.ops.j6q_cu_ext.matmul_naive.default, args)

        result = torch.ops.j6q_cu_ext.matmul_naive(*args)
        expected = reference_matmul(*args)
        assert_close(result, expected)