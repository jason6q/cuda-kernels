#include <random>
#include <vector>
#include <cstdint>

#include <cuda_runtime.h>

#include "tensor.h"

namespace jq {
    Tensor empty(
        std::vector<int32_t> shape, 
        std::optional<DType> dtype = DType::FP32,
        std::optional<Device> device = Device::CPU
    ){
        jq::Tensor empty_tensor;
        int32_t data_size = shape[0];
    }

    Tensor random_uniform(
        std::vector<int32_t> shape, 
        std::optional<DType> dtype = DType::FP32, 
        std::optional<int32_t> seed = std::nullopt,
        std::optional<Device> device = Device::CPU
    ){
        int32_t seed_num = seed.value_or(42);
        jq::Tensor tensor = empty(shape, dtype, device);
        std::mt19937 gen(seed_num) ;// Mersenne Twister appears to be highly uniform


    }
}