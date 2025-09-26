#include <ATen/Operators.h>
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

    __global__ void maxpool2d_naive_kernel(const scalar_t* a, const int size){

    }
}

at::Tensor maxpool2d_naive(const at::Tensor& a, const int size){
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);

    int m = a.size(0);
    int n = a.size(1);

    at::Tensor a_contig = a.contiguous();
    at::Tensor da_contig = torch::empty_like(a, a_contig.options());

    dim3 a_blockDim(16,16);
    dim3 a_gridDim((n+15) / 16, (m+15) / 16);

    cudaStream_t stream = at::cuda::GetCurrentCUDAStream(); 
    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "maxpool2d", [&]{
        const scalar_t* a_ptr = a_contig.data_ptr<scalar_t>();
        const scalar_t* da_ptr = da_contig.data_ptr<scalar_t>();

        // Forward Kernel
        j6q_cu_ext::maxpool2d_naive_kernel<scalar_t><<<a_gridDim, a_blockDim, 0, stream>>>(
        )
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });


}

/*
    Theres no weights to maxpool.
*/
at::Tensor maxpool2d_naive_backward(const at::Tensor& grad_out, const at::Tensor& a){
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);

    int m = a.size(0);
    int n = a.size(1);

    at::Tensor a_contig = a.contiguous();
    at::Tensor da_contig = torch::empty_like(a, a_contig.options());

    dim3 blockDim(16,16);
    dim3 gridDim((n+15) / 16, (m+15) / 16);

    cudaStream_t stream = at::cuda::GetCurrentCUDAStream(); 
    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "maxpool2d", [&]{
        const scalar_t* a_ptr = a_contig.data_ptr<scalar_t>();
        const scalar_t* da_ptr = da_contig.data_ptr<scalar_t>();

        j6q_cu_ext::maxpool2d_naive_kernel<scalar_t><<<a_gridDim, a_blockDim, 0, stream>>>(
        )
    });


}

TORCH_LIBRARY_FRAGMENT(j6q_cu_ext, m){

}

TORCH_LIBRARY_IMPL(j6q_cu_ext, CUDA, m){

}