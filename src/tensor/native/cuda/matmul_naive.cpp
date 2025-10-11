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

        auto a_buf = a.data();
        auto b_buf = b.data();
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

        return c;
    }
}