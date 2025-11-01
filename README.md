# cuda-kernels
## TODO: Migrate private kernel repo here with torch support.
My custom implementations of common CUDA kernels such as: gemm, conv, alphafold trimul, batchnorm fusion, etc...

Most likely not as good as the kernels you find in cuDNN. This repo is primarily for my own personal exploration and practice around different kernel implementations.

These kernels were developed on an RTX A6000.

1. Add a kernel selector mechanism based off core::Device
2. Add int8, bfloat16, float16 Dtype support
3. Shape/View struct -> Building our way towards arbitrary shape support on the kernels.
4. Singleton Dispatch table / Dispatch Keyset for op/device kernel lookup


## Environment Setup

Need to use `python3.11` for pybind11

```
conda create -n cuda-kernels python=3.11
conda activate cuda-kernels
conda install python
pip install -r requirements.txt
```

### Dependencies
TODO: Add these dependencies
```
DLPack
CUTLASS
CuDNN
```

### Building the C++ (Torch Agnostic) Extension
Just run the build script. Make sure to modify the flags
inside it for CUDA_ARCH support and where the torch
library is installed.
```
./build.sh
```

### Building the PyTorch Extension
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

## Adding a new CUDA kernel
Add the CUDA kernel code here: `tensor/native/cuda/<OP>.cu`

Add the Tensor interface to the CUDA Kernel (Launcher) here: `tensor/native/<OP>.cu`

Add the test case here: `tensor/native/tests/<OP>.cpp`

Register the Tensor interface operator to: `tensor/native/kernels.cuh`

Finally, add the files to the `CMakeLists.txt`


## Creating a Custom C++/CUDA Extension for PyTorch
Instructions to integrate all kernels into PyTorch's codebase. 

See [PyTorch Custom Operators](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html#custom-ops-landing-page) and [The Custom Operators Manual](https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#heading=h.ptttacy8y1u9) for a detailed reference.

## Helpful References
- [Pytorch C++ API](https://docs.pytorch.org/cppdocs/)
- [Extension-CPP](https://github.com/pytorch/extension-cpp/tree/master)
- [Getting Started with CuTe](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/00_quickstart.html)
- [The Jacobian for JVP and VJP](https://wangkuiyi.github.io/jacobian.html)

# Kernels
Bunch of GEMMS
Bunch of Convs
Mixup
TriMul from AlphaFold
| Operator | CPU | CUDA | CUDA-CuTE
| ----------- | ----------- |----|----|
| arange |&#x2713;|&#x2713;|&#x2717;
| matmul_naive|&#x2717;|&#x2713;|&#x2717;
| matmul_tile|&#x2717;|&#x2713;|&#x2717;



## Issues
Compatibility issue with CUDA12.x headers and glibc 2.41 (Ubuntu 25.04)

Local Patch:

Edit `/usr/local/cuda/targets/x86-64-linux/include/crt/math_functions.h` and replace the lines with:
```
-extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double sinpi(double x);
-extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float  sinpif(float x);
-extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double cospi(double x);
-extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float  cospif(float x);
+extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double sinpi(double x) noexcept(true);
+extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float  sinpif(float x) noexcept(true);
+extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double cospi(double x) noexcept(true);
+extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float  cospif(float x) noexcept(true);
```
