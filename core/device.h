#pragma once
#include <cstdint>
namespace core{
    // Forward Declarations

    enum DType: int32_t { 
        FP32
    };

    enum Device: int32_t {
        CPU,
        CUDA,
    };
}