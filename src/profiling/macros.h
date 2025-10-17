#pragma once
#include <iostream>

enum NVTX_COLOR {
    RED,
    GREEN,
    BLUE,
    ORANGE
};

//#ifdef ENABLE_NVTX
//    #include <nvtx3/nvtx3.hpp>
//    #define NVTX_RANGE(name) nvtx3::scoped_range _nvtx_range{name}
//#else
//    #define NVTX_RANGE(name) 
//#endif