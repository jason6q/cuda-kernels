/*
    The most basic matmul there is without leveraging shared memory
    and indexing directly from global.
*/
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace j6q_cu_ext{
    __global__ void matmul_naive_kernel(const float* a, const float* b, float* c, int m, int k, int n){
        // MxK, KxN = MxN
        int x = (blockDim.x * blockIdx.x) + threadIdx.x;
        int y = (blockDim.y * blockIdx.y) + threadIdx.y;

        if(x < n && y < m){
            float sum = 0.f;
            for(int i = 0; i < k; ++i){
                sum  += a[y*k + i] * b[i*n + x];
            }
            c[y*n + x] = sum;
        }
    }

    __global__ void matmul_naive_backward_kernel(){

    }
}

at::Tensor matmul_naive(const at::Tensor& a, const at::Tensor& b){
    int m = a.size(0);
    int k = a.size(1);
    int n = b.size(1);

    //TORCH_CHECK(a.size(1) == b.size(0), "Number of columns of a must equal number of rows of b."));
    TORCH_CHECK(a.dtype() == at::kFloat, "a must be float");
    TORCH_CHECK(b.dtype() == at::kFloat, "b must be float");
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);

    TORCH_CHECK(b.size(0) == k, "Rows of a must equal columns of b");

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor c = torch::empty({m,n}, a_contig.options());

    const float* a_ptr = a_contig.data_ptr<float>();
    const float* b_ptr = b_contig.data_ptr<float>();
    float* c_ptr = c.data_ptr<float>();

    dim3 blockDim(16,16);
    dim3 gridDim((n + 15) / 16, (m + 15) / 16);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    j6q_cu_ext::matmul_naive_kernel<<<gridDim, blockDim, 0>>>(a_ptr, b_ptr, c_ptr, m, k, n);
    return c;
}

std::tuple<at::Tensor, at::Tensor> matmul_naive_backward(
    const at::Tensor& grad_out, const at::Tensor& a, const at::Tensor& b, const at::Tensor& out){

    return {a,b};
}


TORCH_LIBRARY_FRAGMENT(j6q_cu_ext, m){
    m.def("matmul_naive(Tensor a, Tensor b) -> Tensor");
    m.def("matmul_naive_backward(Tensor grad_out, Tensor a, Tensor b, Tensor out) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(j6q_cu_ext, CUDA, m){
    m.impl("matmul_naive", &matmul_naive);
    m.impl("matmul_naive_backward", &matmul_naive_backward);
}