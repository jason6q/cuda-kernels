#include "core/device.h"
#include "tensor/tensor.h"
#include "tensor/native/kernels.cuh"

namespace jqTen{
    Tensor arange_cuda(int32_t n){
        Tensor a = empty({n});
        a.to(core::Device::CUDA);

        float* a_buf = static_cast<float*>(a.data());

        // TODO: Change this soon.
        dim3 gridDim = dim3((n - 1) / 32 + 1);
        dim3 blockDim = dim3(32);
        arange_kernel<float><<<gridDim, blockDim, 0>>>(a_buf, n);

        return a;
    }
}