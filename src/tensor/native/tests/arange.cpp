#include "core/device.h"
#include "tensor/tensor.h"
#include "tensor/ops.h"

int main(int argc, char* argv[]){
    jqTen::Tensor a = jqTen::arange_cuda(10);
    a.print();
    return 0;
}