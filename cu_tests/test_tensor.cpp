#include <iostream>
#include <cuda_runtime.h>

#include "tensor.h"
#include "data_ptr.h"
#include "device.h"


int main(int argc, char* argv[]){
    std::vector<int32_t> shape = {10};
    jq::Tensor tensor = jq::empty(shape);
    jq::Tensor cuda_tensor = jq::Tensor(shape, jq::DType::FP32, jq::Device::CUDA);
    jq::Tensor empty_tensor = jq::empty(shape);
    jq::DataPtr data_ptr = tensor.data_ptr();
    float* fptr = static_cast<float*>(data_ptr.get());

    tensor.to(jq::Device::CUDA);
    data_ptr = tensor.data_ptr();
    fptr = static_cast<float*>(data_ptr.get());
    std::cout << fptr << std::endl;

    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, data_ptr.get());
    if(err != cudaSuccess){
        std::cerr << "Failed to get cuda attributes: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    if(attr.type == cudaMemoryTypeHost){
        std::cout << "Tensor failed to move to CUDA" << std::endl;
    }
    else{
        std::cout << "Tensor successfully moved to CUDA" << std::endl;
    }

    return 0;
}