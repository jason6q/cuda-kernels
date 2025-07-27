#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace j6q_cu_ext {
    __global__ void test_kernel(const float* a, const float* b, float* c, int size){
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < size){
            c[i] = a[i] + b[i];
        }
    }
}

at::Tensor test_cuda(const at::Tensor& a, const at::Tensor& b){
    TORCH_CHECK(a.dtype() == at::kFloat)
    TORCH_CHECK(b.dtype() == at::kFloat)
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA)
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA)

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor c = torch::empty(a_contig.sizes(), a_contig.options());

    const float* a_ptr = a_contig.data_ptr<float>();
    const float* b_ptr = b_contig.data_ptr<float>();
    float* c_ptr = c.data_ptr<float>();

    int numel = a_contig.numel();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(); // Whatever the current GPU device is
    j6q_cu_ext::test_kernel<<<(numel+255)/256, 256, 0>>>(a_ptr, b_ptr, c_ptr, numel);
    return c;
}

TORCH_LIBRARY_FRAGMENT(j6q_cu_ext, m){
    m.def("test(Tensor a, Tensor b) -> Tensor");
}

// Use TORCH_LIBRARY_IMPL macro for registering backends for the operator. We'll only do CUDA here.
// This registers our kernel to Pytorch's Python frontend.
TORCH_LIBRARY_IMPL(j6q_cu_ext, CUDA, m){
    m.impl("test", &test_cuda);
}