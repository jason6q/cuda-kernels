/**
 * 
 * Custom Tensor Library.
 * Keep this as minimal as possible. Using this mainly
 * for testing purposes.
 */
#pragma once
#include <vector>
#include <cstdint>
#include <optional>

namespace jq{
    enum DType: int32_t { 
        FP32=0 
    };

    enum Device: int32_t {
        CPU=0,
        CUDA=1
    };

    // Try to have this mimic the ATen Tensor minimally.
    struct Tensor{
        void *data_ptr;
        int64_t sizes[4];

        // Might migrate this to TensorOptions like in Torch.
        DType dtype;
        Device device;
    };

    Tensor empty(
        std::vector<int32_t> shape, 
        std::optional<DType> dtype = DType::FP32,
        std::optional<Device> device = Device::CPU
    );
    Tensor random_uniform(
        std::vector<int32_t> shape, 
        std::optional<DType> dtype = DType::FP32, 
        std::optional<int32_t> seed = std::nullopt,
        std::optional<Device> device = Device::CPU
    );
}