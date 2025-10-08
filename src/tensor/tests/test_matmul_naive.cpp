#include <iostream>
#include <cuda_runtime.h>

#include "tensor.h"
#include "kernels.cuh"

void test_matmul_naive(){
    jqTen::Tensor a = jqTen::random_uniform({10,10});
    jqTen::Tensor b = jqTen::random_uniform({10,10});
}

int main(int argc, char* argv[]){
    test_matmul_naive();
    return 0;
}