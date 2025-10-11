/**
 * Test all variants of matmul kernels here.
 */

#include "core/device.h"
#include "tensor/tensor.h"
#include "tensor/ops.h"

int main(int argc, char* argv[]){
    jqTen::Tensor a = jqTen::random_uniform({10,10});
    jqTen::Tensor b = jqTen::random_uniform({10,10});
    jqTen::Tensor c = jqTen::matmul_naive_cuda(a, b);

    return 0;
}
