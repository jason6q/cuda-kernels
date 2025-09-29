cmake_minimum_required(VERSION 3.30)

project(jq_kernels LANGUAGES CXX CUDA)

# Options
option(USE_ASAN "Enable AdressSanitizer for C/C++ and CUDA host code" OFF)

function(enable_asan target)
    if (NOT USE_ASAN)
    return()
    endif()

    # Instrument C and C++
    target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-fsanitize=address -fno-omit-frame-pointer>
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address -fno-omit-frame-pointer>)

    # Instrument CUDA host code (device code is unaffected; use compute-sanitizer for that)
    target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fsanitize=address,-fno-omit-frame-pointer>)

    # Ensure the linker gets ASan too
    target_link_options(${target} PRIVATE -fsanitize=address)
endfunction()

# C++ Settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA Settings
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# Add include files
set(INCLUDES_DIR
    includes/
)

# Add source files
file(GLOB TESTS "tests/*.cpp")
set(SOURCES
    src/matmul_naive.cu
    src/data_ptr.cpp
    src/tensor.cpp
)

# CMakeLists.txt
file(GLOB TEST_SOURCES tests/*.cpp)

# Build shared library
add_library(jq_kernels SHARED ${SOURCES})

# Specify where headers are located.
target_include_directories(jq_kernels PUBLIC ${INCLUDES_DIR})

# Find CUDA runtime
find_package(CUDAToolkit REQUIRED)
target_link_libraries(jq_kernels PRIVATE CUDA::cudart)

# Link Torch
find_package(Torch REQUIRED)
target_link_libraries(jq_kernels PRIVATE ${TORCH_LIBRARIES})
set_property(TARGET jq_kernels PROPERTY CXX_STANDARD 17)
set_property(TARGET jq_kernels PROPERTY CUDA_STANDARD 17)

enable_asan(jq_kernels)

# Build Test cases
file(GLOB TEST_SOURCES cu_tests/*.cpp)

foreach(test_src ${TEST_SOURCES})
    # Get the filename without extension
    get_filename_component(test_name ${test_src} NAME_WE)

    # Create an executable per test
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE jq_kernels)

    # Link libraries your tests need
    target_link_libraries(${test_name} PRIVATE CUDA::cudart)
    target_link_libraries(${test_name} PRIVATE ${TORCH_LIBRARIES})

    set_property(TARGET ${test_name} PROPERTY CXX_STANDARD 17)
    set_property(TARGET ${test_name} PROPERTY CUDA_STANDARD 17)

    enable_asan(${test_name})

    # Register with CTest
    #add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
