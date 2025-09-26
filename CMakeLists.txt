project(jq_kernels LANGUAGES CXX CUDA)

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
set(SOURCES
    src/matmul_naive.cu
)

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