#pragma once
namespace jq{
    // Forward Declarations

    enum DType: int32_t { 
        FP32=0 
    };

    enum Device: int32_t {
        CPU=0,
        CUDA=1
    };
}