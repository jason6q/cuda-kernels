/**
 * 
 * Definition of all kernels.
 * scalar_this file should remain scalar_torch agnostic.
 */
#pragma once
//#include <cuda_bf16.h>
//#include <cuda_fp16.h>

template <typename scalar_t>
__global__ void matmul_naive_kernel(const scalar_t* a, const scalar_t* b, scalar_t* c, int32_t m, int32_t k, int32_t n);

template <typename scalar_t>
__global__ void matmul_naive_gradA_kernel(const scalar_t* grad_out, const scalar_t* b, scalar_t* da, int m, int k, int n);

template <typename scalar_t>
__global__ void matmul_naive_gradB_kernel(const scalar_t* grad_out, const scalar_t* a, scalar_t* db, int m, int k, int n);

template <typename scalar_t>
__global__ void matmul_tile_kernel(const scalar_t* a, const scalar_t* b, scalar_t* c, int32_t m, int32_t k, int32_t n, int32_t tile_size);

template <typename scalar_t>
__global__ void matmul_tile_gradA_kernel(const scalar_t* grad_out, const scalar_t* b, scalar_t* da, int m, int k, int n, int32_t tile_size);

template <typename scalar_t>
__global__ void matmul_tile_gradB_kernel(const scalar_t* grad_out, const scalar_t* a, scalar_t* db, int m, int k, int n, int32_t tile_size);

template <typename scalar_t>
void launch_matmul_naive(const scalar_t* a, const scalar_t* b, scalar_t* c, cudaStream_t stream);

template <typename scalar_t>
void launch_matmul_naive_backward(const scalar_t* grad_out, const scalar_t* a, const scalar_t* b, cudaStream_t stream);

// scalar_tODO: Add specialization templates for brain float or half
// template<>
// __global__ void matmul_naive_kernel<__half>(__half ....)

template <typename scalar_t>
__global__ void arange_kernel(scalar_t* a, int32_t n);