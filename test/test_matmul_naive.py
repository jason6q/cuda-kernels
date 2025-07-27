import torch

def reference_matmul(a,b):
    return a @ b

from utils import sample_2d_inputs

if __name__ == '__main__':
    device = torch.device("cuda:0")
    samples = sample_2d_inputs(device)
    for args in samples:
        result = torch.ops.j6q_cu_ext.matmul_naive(*args)
        expected = reference_matmul(*args)
        torch.testing.assert_close(result, expected)

        torch.library.opcheck(torch.ops.j6q_cu_ext.matmul_naive.default, args)