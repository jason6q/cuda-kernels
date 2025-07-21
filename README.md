# cuda-kernels
## TODO: Migrate private kernel repo here with torch support.
My custom implementations of common CUDA kernels such as: gemm, conv, alphafold trimul, batchnorm fusion, etc...

Most likely not as good as the kernels you find in cuDNN. This repo is primarily for my own personal exploration and practice around different kernel implementations.

These kernels were developed on an RTX A6000.

### Kernels

Bunch of GEMMS
Bunch of Convs
Mixup
TriMul from AlphaFold

## Environment Setup

```
conda create -n cuda-kernels
conda activate cuda-kernels
conda install python
pip install -r requirements.txt
```

### Dependencies
...

### Build Extension
```
pip install --no-build-isolation -e .
```

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