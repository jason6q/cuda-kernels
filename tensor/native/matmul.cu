#include "cuda_runtime.h"
#include <vector>

#include "core/data_ptr.h"
#include "core/device.h"
#include "tensor/macros.h"
#include "tensor/ops.h"
#include "tensor/tensor.h"
#include "tensor/native/kernels.cuh"

namespace jqTen{
    Tensor matmul_tile_cuda(const Tensor& a, const Tensor& b){
        JQ_ASSERT(a.device() == core::Device::CUDA, "Tensor a device not CUDA");
        JQ_ASSERT(b.device() == core::Device::CUDA, "Tensor b device not CUDA");

        std::vector<int32_t> a_shape = a.shape();
        std::vector<int32_t> b_shape = a.shape();

        int32_t m = a_shape.back();                 // (..., m, n)
        int32_t n = b_shape[b_shape.size()-1];      // (... n, k)
        int32_t k = b_shape.back();

        Tensor c = empty({m,k});
        c.to(core::Device::CUDA);

        const float* a_buf = static_cast<const float*>(a.data());
        const float* b_buf = static_cast<const float*>(b.data());
        float* c_buf = static_cast<float*>(c.data());

        // Kernel Launch
        int32_t GRID_SIZE = 16;
        int32_t BLOCK_SIZE = 16;
        int32_t TILE_SIZE = BLOCK_SIZE;

        dim3 gridDim(16,16);
        dim3 blockDim(16,16); // Will be tile size

        // SHMEM Size
        int32_t shmem_size = 2*TILE_SIZE*TILE_SIZE*sizeof(float);
        matmul_tile_kernel<float><<<gridDim, blockDim, shmem_size>>>(a_buf, b_buf, c_buf, m, n, k, TILE_SIZE);
    }

    Tensor matmul_naive_cuda(const Tensor& a, const Tensor& b){
        JQ_ASSERT(a.device() == core::Device::CUDA, "Tensor a device not CUDA");
        JQ_ASSERT(b.device() == core::Device::CUDA, "Tensor b device not CUDA");

        std::vector<int32_t> a_shape = a.shape();
        std::vector<int32_t> b_shape = b.shape();

        int32_t m = a_shape.back();
        int32_t n = b_shape[b_shape.size()-1];
        int32_t k = b_shape.back();

        // Calculate c output dim
        // Take a = {..., M, N}, b = {..., N, K}, c = {..., M, K}
        Tensor c = empty({m,k});
        c.to(core::Device::CUDA);

        // TODO: Template the types here.
        // Allow this section of the code to handle different types.
        const float* a_buf = static_cast<const float*>(a.data());
        const float* b_buf = static_cast<const float*>(b.data());
        float* c_buf = static_cast<float*>(c.data());

        // Kernel launch
        dim3 blockDim = dim3(16,16);
        dim3 gridDim = dim3(16,16);

        // TODO: Specify templated scalar_t instead of just float.
        //       May need to make a similar macro like in torch.
        matmul_naive_kernel<float><<<gridDim, blockDim, 0>>>(a_buf, b_buf, c_buf, m,k,n);

        return c;
    }
}