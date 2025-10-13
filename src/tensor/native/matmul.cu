#include <vector>

#include "cuda_runtime.h"

#include "core/data_ptr.h"
#include "core/device.h"
#include "tensor/macros.h"
#include "tensor/ops.h"
#include "tensor/tensor.h"
#include "tensor/native/kernels.cuh"

namespace jqTen{
    Tensor matmul_naive_cuda(const Tensor& a, const Tensor& b){
        JQ_ASSERT(a.device() == core::Device::CUDA, "Tensor a device not CUDA");
        JQ_ASSERT(b.device() == core::Device::CUDA, "Tensor b device not CUDA");

        std::vector<int32_t> a_shape = a.shape();
        std::vector<int32_t> b_shape = b.shape();

        //JQ_ASSERT(a_shape.size() > 1, "Test");
        //JQ_ASSERT(b_shape.size() > 1, "Test");

        int32_t m = a_shape.back();
        int32_t n = b_shape[b_shape.size()-1];
        int32_t k = b_shape.back();

        // Calculate c output dim
        // Take a = {..., M, N}, b = {..., N, K}, c = {..., M, K}
        Tensor c = empty({m,k});

        // TODO: Template the types here.
        // Allow this section of the code to handle different types.
        const float* a_buf = static_cast<const float*>(a.data());
        const float* b_buf = static_cast<const float*>(b.data());
        float* c_buf = static_cast<float*>(c.data());

        // Don't need this if tensor is already on CUDA
        //void* a_buf_d;
        //void* b_buf_d;
        //void* c_buf_d;
        //JQ_ASSERT_CUDA_ERR_CHECK(cudaMalloc((void**)&a_buf_d, a.nbytes()));
        //JQ_ASSERT_CUDA_ERR_CHECK(cudaMalloc((void**)&b_buf_d, b.nbytes()));
        //JQ_ASSERT_CUDA_ERR_CHECK(cudaMalloc((void**)&c_buf_d, c.nbytes()));
        //cudaMemcpy(a_buf_d, a_buf, a.nbytes(), cudaMemcpyHostToDevice);
        //cudaMemcpy(b_buf_d, b_buf, b.nbytes(), cudaMemcpyHostToDevice);
        //cudaMemcpy(c_buf_d, c_buf, c.nbytes(), cudaMemcpyHostToDevice);

        // Kernel launch
        dim3 blockDim = dim3(16,16);
        dim3 gridDim = dim3(16,16);

        // TODO: Specify templated scalar_t instead of just float.
        //       May need to make a similar macro like in torch.
        matmul_naive_kernel<float><<<gridDim, blockDim, 0>>>(a_buf, b_buf, c_buf, m,k,n);

        //cudaFree(a_buf_d);
        //cudaFree(b_buf_d);
        //cudaFree(c_buf_d);

        return c;
    }
}