#include <random>
#include <vector>
#include <cstdint>
#include <algorithm>

#include <cuda_runtime.h>

#include "tensor.h"
#include "data_ptr.h"
#include "device.h"

namespace jq {
    size_t compute_nbytes(const std::vector<int32_t>& shape, DType dtype){
        // Calculate bytes
        size_t size = 1;
        for(int32_t i = 0; i < shape.size(); ++i){
            size *= shape[i];
        }
        if(dtype == DType::FP32){
            size *= sizeof(float);
        }
        else{
            size *= sizeof(float);
        }

        return size;
    }

    Tensor::Tensor(
        const std::vector<int32_t>& shape, 
        DType dtype,
        Device device): 
        shape_(shape), dtype_(dtype), device_(device), nbytes_(compute_nbytes(shape_, dtype_)),
        data_ptr_(device == Device::CUDA ? DataPtr::cuda(nbytes_, true) : DataPtr::cpu(nbytes_, true, 64)) // 64 byte alignment
        {

    }

    Tensor empty(
        const std::vector<int32_t>& shape, 
        std::optional<DType> dtype,
        std::optional<Device> device
    ){
        DType dtype_ = dtype.value_or(DType::FP32);
        Device device_ = device.value_or(Device::CPU);

        return Tensor(shape, dtype_, device_);
    }

    Tensor random_uniform(
        const std::vector<int32_t>& shape, 
        std::optional<DType> dtype,
        std::optional<int64_t> seed,
        std::optional<Device> device
    ){
        int32_t seed_num = seed.value_or(42);
        std::mt19937 gen(seed_num) ;// Mersenne Twister appears to be highly uniform

        // We'll init a CPU tensor than move it to CUDA.
        // TODO: RNG on CUDA variant?
        jq::Tensor tensor = empty(shape, dtype, device);

        // Calculate number of bytes
        size_t size = 1;
        for(int32_t i = 0; i < shape.size(); ++i){
            size *= shape[i];
        }
        size = size * sizeof(float);
    }
}