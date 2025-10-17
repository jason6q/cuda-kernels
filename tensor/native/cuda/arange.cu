template <typename scalar_t>
__global__ void arange_kernel(scalar_t* a, int32_t n){
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = scalar_t(i);
}

template __global__ void arange_kernel<float>(float* a, int32_t n);
template __global__ void arange_kernel<int32_t>(int32_t* a, int32_t n);