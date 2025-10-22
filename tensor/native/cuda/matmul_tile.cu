template<typename scalar_t>
__global__ void matmul_tile_kernel(const scalar_t* a, const scalar_t* b, scalar_t* c, 
    int32_t m, int32_t n, int32_t k, int32_t tile_size){

}

template<typename scalar_t>
__global__ void matmul_tile_gradA_kernel(const scalar_t* grad_out, const scalar_t* b, scalar_t* da, 
    int32_t m, int32_t n, int32_t k, int32_t tile_size){

}

template<typename scalar_t>
__global__ void matmul_tile_gradB_kernel(const scalar_t* grad_out, const scalar_t* a, scalar_t* db, 
    int32_t m, int32_t n, int32_t k, int32_t tile_size){

}

template __global__ void matmul_tile_kernel<float>(const float*, const float*, float*, int32_t, int32_t, int32_t, int32_t);
template __global__ void matmul_tile_gradA_kernel<float>(const float*, const float*, float*, int32_t, int32_t, int32_t, int32_t);
template __global__ void matmul_tile_gradB_kernel<float>(const float*, const float*, float*, int32_t, int32_t, int32_t, int32_t);