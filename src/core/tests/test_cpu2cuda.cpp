#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "core/device.h"
#include "core/data_ptr.h"
#include "tensor/tensor.h"

int main(int argc, char* argv[]){
    std::vector<int32_t> shape = {10,10};
    jqTen::Tensor tensor = jqTen::empty(shape);

    tensor.to(core::Device::CUDA);
    float* data_ptr = static_cast<float*>(tensor.data()); // WARNING: When you grab the data ptr it must be after the move to CUDA

    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, data_ptr);
    if(err != cudaSuccess){
        std::cerr << "Failed to get cuda attributes: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    if(attr.type == cudaMemoryTypeHost){
        std::cout << "Tensor failed to move to CUDA" << std::endl;
        return -1;
    }
    else{
        std::cout << "Tensor successfully moved to CUDA" << std::endl;
    }
    return 0;
}