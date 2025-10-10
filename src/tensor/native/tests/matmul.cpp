/**
 * Test all variants of matmul kernels here.
 */
#include "tensor.h"

#include "ops.h"

int main(int argc, char* argv[]){
    jqTen::Tensor a = jqTen::random_uniform({10,10});
    jqTen::Tensor b = jqTen::random_uniform({10,10});
    jqTen::Tensor c = jqTen::matmul_naive(a, b);

    return 0;
