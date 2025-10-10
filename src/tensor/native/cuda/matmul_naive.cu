#include "kernels.cuh"
#include "tensor.h"

template<typename T>
__global__ void matmul_naive_kernel(const T* a, const T* b, const T* c, int m, int k, int n){
    // MxK, KxN = MxN
    int x = (blockDim.x * blockIdx.x) + threadIdx.x;
    int y = (blockDim.y * blockIdx.y) + threadIdx.y;

    if(x < n && y < m){
        T sum = T(0);
        for(int i = 0; i < k; ++i){
            sum  += a[y*k + i] * b[i*n + x];
        }
        c[y*n + x] = sum;
    }
}

template<typename T>
__global__ void matmul_naive_gradA_kernel(const T* grad_out, const T* b, T* da, int m, int k, int n){
        int x = (blockDim.x * blockIdx.x) + threadIdx.x;
        int y = (blockDim.y * blockIdx.y) + threadIdx.y;

        // grad_out -> (m,n)
        // a -> (m,k)
        // b -> (k,n)
        // da = grad_out @ b.T -> (m,n) @ (n,k) = (m,k)
        if(x >= k || y >= m) return;

        T sum = T(0);
        for(int i = 0; i < n; ++i){
            sum += grad_out[y*n + i]*b[x*n + i];
        }

        da[y*k + x] = sum;
}

template<typename T>
__global__ void matmul_naive_gradB_kernel(
    const T* grad_out, const T* a, T* db, int m, int k, int n){
        int x = (blockDim.x * blockIdx.x) + threadIdx.x;
        int y = (blockDim.y * blockIdx.y) + threadIdx.y;

        // grad_out -> (m,n)
        // a -> (m,k)
        // b -> (k,n)
        // db = a.T @ grad_out = (k,m) @ (m,n) = (k,n)
        if(x >= n || y >= k) return;

        T sum = T(0);
        for(int i = 0; i < m; ++i){
            sum += a[i*k + y]*grad_out[i*n + x];
        }

        db[y*n + x] = sum;
}