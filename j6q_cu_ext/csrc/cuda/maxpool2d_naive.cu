#include <ATen/ATen.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace j6q_cu_ext{
    template<typename scalar_t>
    __global__ void maxpool2d_naive_kernel(const scalar_t* a, const int size){

    }
}

at::Tensor maxpool2d_naive(const at::Tensor& a, int size){
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);

    //cudaStream_t stream = at::cuda::GetCurrentCUDAStream(); 

}

at::Tensor maxpool2d_naive_backward(){

}

TORCH_LIBRARY_FRAGMENT(j6q_cu_ext, m){

}

TORCH_LIBRARY_IMPL(j6q_cu_ext, CUDA, m){

}