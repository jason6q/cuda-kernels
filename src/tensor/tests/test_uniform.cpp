#include <iostream>
#include <cuda_runtime.h>

#include "core/data_ptr.h"
#include "core/device.h"
#include "tensor/tensor.h"

int main(int argc, char* argv[]){
    std::vector<int32_t> shape = {1};
    jqTen::Tensor tensor = jqTen::random_uniform(shape);
    float* fptr = static_cast<float*>(tensor.data());

    /*
    int32_t numel = tensor.numel();

    for(int i = 0; i < numel; ++i){
        std::cout << fptr[i] << std::endl;
    }*/

    // Sort of a dumb test-case but with default seed the
    // following value should be generated
    // Use an epsilon for better reliability with floating
    if(std::abs(fptr[0] - 0.37454f) > 1e-6f){
        std::cout << "Incorrect values randomly generated! " << fptr[0] << std::endl;
        return -1;
    }

    return 0;
}