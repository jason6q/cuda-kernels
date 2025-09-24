
/*
    matmul that uses CUTE
*/
#include <ATen/Operators.h>
#include <ATen/ATen.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace j6q_cu_ext{
    template<typename scalar_t>
    __global__ void matmul_cute_kernel(){

    }

    template<typename scalar_t>
    __global__ void matmul_cute_gradA_kernel(){

    }

    template<typename scalar_t>
    __global__ void matmul_cute_gradB_kernel(){

    }
}

void matmul_cute(){

}

void matmul_cute_backward(){

}

TORCH_LIBRARY_FRAGMENT(j6q_cu_ext, m){
    m.def("matmul_cute(Tensor a, Tensor b) -> Tensor");
    m.def("matmul_cute_backward(Tensor grad_out, Tensor a, Tensor b, Tensor out) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(j6q_cu_ext, CUDA, m){
    m.impl("matmul_cute", &matmul_cute);
    m.impl("matmul_cute_backward", &matmul_cute_backward);
}