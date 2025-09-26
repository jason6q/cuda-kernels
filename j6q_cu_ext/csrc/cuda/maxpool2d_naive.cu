#include <torch/all.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Operators.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace j6q_cu_ext{
    __device__ std::vector<int64_t> calc_maxpool2d_shape(c10::IntArrayRef a_shape, const int stride){
        int m = a_shape(0);
        int n = a_shape(1);

        std::vector<int64_t> b_shape = {}

        return b_shape;
    }

    template<typename scalar_t>
    __global__ void maxpool2d_naive_kernel(const scalar_t* a, const scalar_t* b, const int stride){

    }

    //__global__ void maxpool2d_naive_kernel_backward(const scalar_t* a, const scalar_t* b, const int stride){
    //}
}

at::Tensor maxpool2d_naive(const at::Tensor& a, const int stride){
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);

    int m = a.size(0);
    int n = a.size(1);
    TORCH_INTERNAL_ASSERT(m % 2 == 0 && n % 2 == 0 && stride % 2 == 0); // Make sure even

    // Calculate the shape of output based off stride.
    // Stride will also determine window to pool over. e.g: stride=2 is (2,2), stride=4 is (4,4)
    std::vector<int64_t> b_size = j6q_cu_ext::calc_maxpool2d_shape(a.sizes(), stride);

    at::Tensor a_contig = a.contiguous();
    at::Tensor b = torch::empty({}, a_contig.options());

    dim3 a_blockDim(16,16);
    dim3 a_gridDim((n+15) / 16, (m+15) / 16);

    cudaStream_t stream = at::cuda::GetCurrentCUDAStream(); 
    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "maxpool2d", [&]{
        const scalar_t* a_ptr = a_contig.data_ptr<scalar_t>();
        const scalar_t* b_ptr = b_contig.data_ptr<scalar_t>();

        // Forward Kernel
        j6q_cu_ext::maxpool2d_naive_kernel<scalar_t><<<a_gridDim, a_blockDim, 0, stream>>>(a, b, stride)
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });

}

/*
    Theres no weights to maxpool.
*/
//at::Tensor maxpool2d_naive_backward(const at::Tensor& grad_out, const at::Tensor& a, const int stride){
//    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
//
//    int m = a.size(0);
//    int n = a.size(1);
//
//    at::Tensor a_contig = a.contiguous();
//    at::Tensor da_contig = torch::empty_like(a, a_contig.options());
//
//    dim3 blockDim(16,16);
//    dim3 gridDim((n+15) / 16, (m+15) / 16);
//
//    cudaStream_t stream = at::cuda::GetCurrentCUDAStream(); 
//    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "maxpool2d", [&]{
//        const scalar_t* a_ptr = a_contig.data_ptr<scalar_t>();
//        const scalar_t* da_ptr = da_contig.data_ptr<scalar_t>();
//
//        j6q_cu_ext::maxpool2d_naive_kernel<scalar_t><<<a_gridDim, a_blockDim, 0, stream>>>(
//        )
//    });
//
//
//}

TORCH_LIBRARY_FRAGMENT(j6q_cu_ext, m){

}

TORCH_LIBRARY_IMPL(j6q_cu_ext, CUDA, m){

}