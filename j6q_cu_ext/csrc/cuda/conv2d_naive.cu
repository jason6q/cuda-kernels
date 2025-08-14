#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace j6q_cu_ext{
    __global__ void conv2d_naive(const float* a, const float* b, const float* f, 
        const int n, const int c, const int h, const int w, const int kernel_radius){
    }
}

at::Tensor conv2d_naive(const at::Tensor& a, const at::Tensor& f){
    int n = a.size(0);
    int c = a.size(1);
    int h = a.size(2);
    int w = a.size(3);

    TORCH_CHECK(a.dtype() == at::kFloat, "a must be a float");
    TORCH_CHECK(f.dtype() == at::kFloat, "f must be a float");

    at::Tensor b = torch::empty({n,c,h,w});

    return b;
}

TORCH_LIBRARY_FRAGMENT(j6q_cu_ext, m){
    m.def("conv2d_naive(Tensor a, Tensor f) -> Tensor");
}

TORCH_LIBRARY_IMPL(j6q_cu_ext, CUDA, m){
    m.impl("conv2d_naive", &conv2d_naive);
}