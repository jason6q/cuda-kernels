#include "core/device.h"
#include "tensor/tensor.h"
#include "tensor/ops.h"
#include "profiling/macros.h"

int main(int argc, char* argv[]){
    //NVTX_RANGE("arange test")
    jqTen::Tensor a = jqTen::arange_cuda(10);
    a.print();
    return 0;
}