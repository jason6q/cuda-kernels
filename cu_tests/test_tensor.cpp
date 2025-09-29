#include <iostream>
#include <cuda_runtime.h>

#include "tensor.h"
#include "data_ptr.h"
#include "device.h"

int main(int argc, char* argv[]){
    std::vector<int32_t> shape = {1};
    jq::Tensor cpu_tensor = jq::empty(shape);
    //jq::Tensor cuda_tensor = jq::Tensor(shape, jq::DType::FP32, jq::Device::CUDA);
    //jq::Tensor empty_tensor = jq::empty(shape);
    jq::DataPtr data_ptr = cpu_tensor.data_ptr();

    float* fptr = static_cast<float*>(data_ptr.get());
    std::cout << *fptr << std::endl;

    return 0;
}