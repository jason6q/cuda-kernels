#pragma once
#include <iostream>
#include <cstdlib>

#define JQ_ASSERT(cond, ...)                           \
    do {                                               \
        if (!(cond)) {                                 \
            std::cerr << "Assertion failed: " #cond    \
                      << std::endl;                    \
            std::abort();                              \
        }                                              \
    } while (0)
