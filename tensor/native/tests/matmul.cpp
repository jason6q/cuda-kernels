/**
 * Test all variants of matmul kernels here.
 */
#include "core/device.h"
#include "tensor/tensor.h"
#include "tensor/ops.h"

int main(int argc, char* argv[]){
    jqTen::Tensor a = jqTen::random_uniform({10,10});
    jqTen::Tensor b = jqTen::random_uniform({10,10});
    a.to(core::Device::CUDA);
    b.to(core::Device::CUDA);

    jqTen::Tensor c = jqTen::matmul_naive_cuda(a, b);
    a.print();
    b.print();
    c.print();

    return 0;
}