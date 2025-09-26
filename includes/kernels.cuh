/**
 * 
 * Definition of all kernels.
 * This file should remain Torch agnostic.
 */
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void matmul_naive_kernel(const T* a, const T* b, const T* c, int m, int k, int n);

template <typename T>
__global__ void matmul_naive_gradA_kernel(const T* grad_out, const T* b, const T* da, int m, int k, int n);

template <typename T>
__global__ void matmul_naive_gradB_kernel(const T* grad_out, const T* a, const T* db, int m, int k, int n);

template <typename T>
void launch_matmul_naive(const T* a, const T* b, const T* c, cudaStream_t stream);

template <typename T>
void launch_matmul_naive_backward(const T* grad_out, const T* a, const T* b, cudaStream_t stream);

// TODO: Add specialization templates for brain float or half
// template<>
// __global__ void matmul_naive_kernel<__half>(__half ....)