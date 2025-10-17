#pragma once
#include <iostream>
#include <cstdlib>

#include "cuda_runtime.h"

#define JQ_ASSERT(cond, ...)                           \
    do {                                               \
        if (!(cond)) {                                 \
            std::cerr << "Assertion failed: " #cond    \
                      << std::endl;                    \
            std::abort();                              \
        }                                              \
    } while (0)

#define JQ_ASSERT_CUDA_ERR_CHECK(err) \
    do { \
        if(err != cudaSuccess){ \
            std::cerr << "CUDA Failed: " \
                      << cudaGetErrorString(err) << std::endl; \
            std::abort(); \
        } \
    } while(0)