#include <iostream>
#include <cuda_runtime.h>

#include "tensor.h"
#include "data_ptr.h"
#include "device.h"

int main(int argc, char* argv[]){
    std::vector<int32_t> shape = {10};
    jqTen::Tensor tensor = jqTen::empty(shape);
    core::DataPtr data_ptr = tensor.data_ptr();
    float* fptr = static_cast<float*>(data_ptr.get());
    int32_t numel = tensor.numel();

    for(int i = 0; i < numel; ++i){
        if(fptr[i] != 0){
            std::cout << "Not any empty tensor!" << std::endl;
            return -1;
        }
    }

    return 0;
}