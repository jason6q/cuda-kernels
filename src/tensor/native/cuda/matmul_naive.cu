#include "tensor/native/kernels.cuh"
#include "tensor/tensor.h"

template<typename scalar_t>
__global__ void matmul_naive_kernel(const scalar_t* a, const scalar_t* b, scalar_t* c, int32_t m, int32_t k, int32_t n){
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
__global__ void matmul_naive_gradA_kernel(const scalar_t* grad_out, const scalar_t* b, scalar_t* da, int32_t m, int32_t k, int32_t n){
        int x = (blockDim.x * blockIdx.x) + threadIdx.x;
        int y = (blockDim.y * blockIdx.y) + threadIdx.y;

        // grad_out -> (m,n)
        // a -> (m,k)
        // b -> (k,n)
        // da = grad_out @ b.scalar_t -> (m,n) @ (n,k) = (m,k)
        if(x >= k || y >= m) return;

        scalar_t sum = scalar_t(0);
        for(int i = 0; i < n; ++i){
            sum += grad_out[y*n + i]*b[x*n + i];
        }

        da[y*k + x] = sum;
}

template<typename scalar_t>
__global__ void matmul_naive_gradB_kernel(
    const scalar_t* grad_out, const scalar_t* a, scalar_t* db, int32_t m, int32_t k, int32_t n){
        int x = (blockDim.x * blockIdx.x) + threadIdx.x;
        int y = (blockDim.y * blockIdx.y) + threadIdx.y;

        // grad_out -> (m,n)
        // a -> (m,k)
        // b -> (k,n)
        // db = a.scalar_t @ grad_out = (k,m) @ (m,n) = (k,n)
        if(x >= n || y >= k) return;

        scalar_t sum = scalar_t(0);
        for(int i = 0; i < m; ++i){
            sum += a[i*k + y]*grad_out[i*n + x];
        }

        db[y*n + x] = sum;
}

// TODO: Macro this for custom types.
template __global__ void matmul_naive_kernel<float>( const float*, const float*, float*, int32_t, int32_t, int32_t);
template __global__ void matmul_naive_gradA_kernel<float>( const float*, const float*, float*, int32_t, int32_t, int32_t);
template __global__ void matmul_naive_gradB_kernel<float>( const float*, const float*, float*, int32_t, int32_t, int32_t);