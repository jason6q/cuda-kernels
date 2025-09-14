/*
    The most basic matmul there is without leveraging shared memory
    and indexing directly from global.
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
    __global__ void matmul_naive_kernel(const scalar_t* a, const scalar_t* b, scalar_t* c, int m, int k, int n){
        // MxK, KxN = MxN
        int x = (blockDim.x * blockIdx.x) + threadIdx.x;
        int y = (blockDim.y * blockIdx.y) + threadIdx.y;

        if(x < n && y < m){
            scalar_t sum = scalar_t(0);
            for(int i = 0; i < k; ++i){
                sum  += a[y*k + i] * b[i*n + x];
            }
            c[y*n + x] = sum;
        }
    }

    template<typename scalar_t>
    __global__ void matmul_naive_gradA_kernel(
        const scalar_t* grad_out, const scalar_t* b, scalar_t* da, int m, int k, int n){
            int x = (blockDim.x * blockIdx.x) + threadIdx.x;
            int y = (blockDim.y * blockIdx.y) + threadIdx.y;

            // grad_out -> (m,n)
            // a -> (m,k)
            // b -> (k,n)
            // da = grad_out @ b.T -> (m,n) @ (n,k) = (m,k)
            if(x >= k || y >= m) return;

            scalar_t sum = scalar_t(0);
            for(int i = 0; i < n; ++i){
                sum += grad_out[y*n + i]*b[x*n + i];
            }

            da[y*k + x] = sum;
    }

    template<typename scalar_t>
    __global__ void matmul_naive_gradB_kernel(
        const scalar_t* grad_out, const scalar_t* a, scalar_t* db, int m, int k, int n){
            int x = (blockDim.x * blockIdx.x) + threadIdx.x;
            int y = (blockDim.y * blockIdx.y) + threadIdx.y;

            // grad_out -> (m,n)
            // a -> (m,k)
            // b -> (k,n)
            // db = a.T @ grad_out = (k,m) @ (m,n) = (k,n)
            if(x >= n || y >= k) return;

            scalar_t sum = scalar_t(0);
            for(int i = 0; i < m; ++i){
                sum += a[i*k + y]*grad_out[i*n + x];
            }

            db[y*n + x] = sum;
    }
}

at::Tensor matmul_naive(const at::Tensor& a, const at::Tensor& b){
    int m = a.size(0);
    int k = a.size(1);
    int n = b.size(1);

    //TORCH_CHECK(a.size(1) == b.size(0), "Number of columns of a must equal number of rows of b."));
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "Dtype mismatch between a and b")
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);

    TORCH_CHECK(b.size(0) == k, "Rows of a must equal columns of b");

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor c = torch::empty({m,n}, a_contig.options());

    dim3 blockDim(16,16);
    dim3 gridDim((n + 15) / 16, (m + 15) / 16);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "matmul_naive", [&]{
        const scalar_t* a_ptr = a_contig.data_ptr<scalar_t>();
        const scalar_t* b_ptr = b_contig.data_ptr<scalar_t>();
        scalar_t* c_ptr = c.data_ptr<scalar_t>();
        j6q_cu_ext::matmul_naive_kernel<scalar_t><<<gridDim, blockDim, 0>>>(a_ptr, b_ptr, c_ptr, m, k, n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
    return c;
}

std::tuple<at::Tensor, at::Tensor> matmul_naive_backward(
    const at::Tensor& grad_out, const at::Tensor& a, const at::Tensor& b, const at::Tensor& out){
    int m = a.size(0);
    int k = a.size(1);
    int n = b.size(1);

    //TORCH_CHECK(a.size(1) == b.size(0), "Number of columns of a must equal number of rows of b."));
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "Dtype mismatch between a and b")
    TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);

    TORCH_CHECK(b.size(0) == k, "Shape mismatch");

    at::Tensor a_contig = a.contiguous();
    at::Tensor b_contig = b.contiguous();
    at::Tensor grad_out_contig = grad_out.contiguous();

    at::Tensor da = torch::empty_like(a, a_contig.options());
    at::Tensor db = torch::empty_like(b, a_contig.options());

    dim3 da_blockDim(16,16);
    dim3 da_gridDim((k + 15) / 16, (m + 15) / 16);
    dim3 db_blockDim(16,16);
    dim3 db_gridDim((n + 15) / 16, (k + 15) / 16);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "matmul_naive", [&]{
        const scalar_t* a_ptr = a_contig.data_ptr<scalar_t>();
        const scalar_t* b_ptr = b_contig.data_ptr<scalar_t>();
        const scalar_t* grad_out_ptr = grad_out_contig.data_ptr<scalar_t>();

        scalar_t* da_ptr = da.data_ptr<scalar_t>();
        scalar_t* db_ptr = db.data_ptr<scalar_t>();

        j6q_cu_ext::matmul_naive_gradA_kernel<scalar_t><<<da_gridDim, da_blockDim, 0, stream>>>(
            grad_out_ptr, b_ptr, da_ptr, m, k, n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        j6q_cu_ext::matmul_naive_gradB_kernel<scalar_t><<<db_gridDim, db_blockDim, 0, stream>>>(
            grad_out_ptr, a_ptr, db_ptr, m, k, n);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });

    return {da, db};
}

TORCH_LIBRARY_FRAGMENT(j6q_cu_ext, m){
    m.def("matmul_naive(Tensor a, Tensor b) -> Tensor");
    m.def("matmul_naive_backward(Tensor grad_out, Tensor a, Tensor b, Tensor out) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(j6q_cu_ext, CUDA, m){
    m.impl("matmul_naive", &matmul_naive);
    m.impl("matmul_naive_backward", &matmul_naive_backward);
}