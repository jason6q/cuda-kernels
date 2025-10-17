#include <iostream>
#include <cuda_runtime_api.h>

#include "core/data_ptr.h"
#include "core/device.h"
#include "tensor/tensor.h"

int main(int argc, char* argv[]){
    std::vector<int32_t> shape = {10};
    jqTen::Tensor tensor = jqTen::empty(shape);
    float* fptr = static_cast<float*>(tensor.data());
    int32_t numel = tensor.numel();

    for(int i = 0; i < numel; ++i){
        if(fptr[i] != 0){
            std::cout << "Not any empty tensor!" << std::endl;
            return -1;
        }
    }

    return 0;
}