#include <iostream>
#include <cuda_runtime_api.h>

#include "tensor/tensor.h"

void test_matmul_naive(){
    jqTen::Tensor a = jqTen::random_uniform({10,10});
    jqTen::Tensor b = jqTen::random_uniform({10,10});
}

int main(int argc, char* argv[]){
    test_matmul_naive();
    return 0;
}