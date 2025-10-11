#pragma once
#include <iostream>

#define JQ_ASSERT(cond, ...) \
    if (!cond){ \
        std::cerr << "Assertion failed." << std::endl; \
        std::abort(); \
    }