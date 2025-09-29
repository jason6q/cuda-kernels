/**
 * Basic test case to make sure nothing crashes.
 */
#include <cuda_runtime.h>

#include "data_ptr.h"

int main(int argc, char* argv[]){
    size_t bytes = 1024 * sizeof(int64_t);
    size_t align = 32;
    bool zero = true;

    auto data_ptr_cpu = jq::DataPtr::cpu(bytes, zero, align);

    //cudaStream_t stream;
    //cudaError_t cudaErr = cudaStreamCreate(&stream);
    //if(cudaErr != cudaSuccess){
    //    return -1;
    //}
    auto data_ptr_cuda = jq::DataPtr::cuda(bytes, zero);

    return 0;
}