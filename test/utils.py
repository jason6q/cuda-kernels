import torch

def sample_1d_inputs(device, *, requires_grad=False):
    def make_tensor(*size):
        return torch.randn(size, device=device, requires_grad=requires_grad)

    return [
        [make_tensor(3), make_tensor(3)],
        [make_tensor(1024), make_tensor(1024)]
    ]

def sample_2d_inputs(device, *, requires_grad=False):
    def make_tensor(*size):
        return torch.randn(size, dtype=torch.float, device=device, requires_grad=requires_grad)
            

    return [
        [make_tensor(32,32), make_tensor(32,32)],
        [make_tensor(128,32), make_tensor(32,128)],
        [make_tensor(256,64), make_tensor(64,256)],
        [make_tensor(512,128), make_tensor(128,512)],
        [make_tensor(1024,256), make_tensor(256,1024)],
    ]