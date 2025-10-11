#pragma once
#include <iostream>

#define JQ_ASSERT(cond, msg) \
    if (!cond){ \
        std::cerr << "Assertion failed." << std::endl \
    }