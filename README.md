# cuda-kernels
## TODO: Migrate private kernel repo here with torch support.
My custom implementations of common CUDA kernels such as: gemm, conv, alphafold trimul, batchnorm fusion, etc...

Most likely not as good as the kernels you find in cuDNN. This repo is primarily for my own personal exploration and practice around different kernel implementations.

These kernels were developed on an RTX A6000.


## Environment Setup

Need to use `python3.11` for pybind11

```
conda create -n cuda-kernels python=3.11
conda activate cuda-kernels
conda install python
pip install -r requirements.txt
```

### Dependencies
...

CUTLASS
NVTX

### Build Extension
```
pip install --no-build-isolation -e .
```

### DeviceQuery
If you have CUDA Samples installed there's a helpful script that should get you all
the information you need about your GPU to do things like warp, block, grid calculations.
```
deviceQuery
```

#### RTX A6000 Specs
| Hardware | Constraint|
|------|------|
|Threads per Warp | 32 threads|
|Number of Registers | 65536 registers |
|Size of Constant Memory|64kb|
|Size of Shared Memory per Block|48kb|
|Threads per Block| 1024 threads|
|Threads per SM| 1536 threads|
|Number of SM| 84|
|Number of CUDA Cores| 10752|
|Number of Tensor Cores| 336|
|Size of Memory| 48GB|



## Creating a Custom C++/CUDA Extension for PyTorch
Instructions to integrate all kernels into PyTorch's codebase. 

See [PyTorch Custom Operators](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html#custom-ops-landing-page) and [The Custom Operators Manual](https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#heading=h.ptttacy8y1u9) for a detailed reference.

### Helpful files to look at
```
torch/headeronly/macros/Macros.h
c10/util/Exception.h
torch/library.h
torch/testing/_internal
```

## Helpful References
- [Pytorch C++ API](https://docs.pytorch.org/cppdocs/)
- [Extension-CPP](https://github.com/pytorch/extension-cpp/tree/master)

# Implemented Kernels
Bunch of GEMMS
Bunch of Convs
Mixup
TriMul from AlphaFold
