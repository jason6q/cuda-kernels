cmake_minimum_required(VERSION 3.20)  # 3.30 fine too
project(jq_kernels LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find Torch and CUDA runtime from Conda
find_package(Torch REQUIRED)                   

# Build lib
set(INCLUDES_DIR 
  src/
) 
set(SOURCES
  src/
  src/core/data_ptr.cpp
  src/tensor/tensor.cpp
  src/tensor/native/matmul.cu
  src/tensor/native/arange.cu
  src/tensor/native/cuda/matmul_naive.cu
  src/tensor/native/cuda/arange.cu
)
add_library(jq_kernels SHARED ${SOURCES})
target_include_directories(jq_kernels PUBLIC ${INCLUDES_DIR})
target_link_libraries(jq_kernels PRIVATE ${TORCH_LIBRARIES} CUDA::cudart)
set_property(TARGET jq_kernels PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(jq_kernels PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON
)

# Build tests
file(GLOB TEST_SOURCES 
  src/core/tests/*.cpp
  src/tensor/tests/*.cpp
  src/tensor/native/tests/*.cpp
)
foreach(test_src ${TEST_SOURCES})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} PRIVATE jq_kernels ${TORCH_LIBRARIES} CUDA::cudart)
  target_compile_features(${test_name} PRIVATE cxx_std_17)
  set_target_properties(${test_name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
  )

endforeach()

# Torch libs live here:
get_filename_component(TORCH_LIBDIR "${TORCH_LIBRARIES}" DIRECTORY)
set(TORCH_NVIDIA_ROOT "${TORCH_LIBDIR}/../../nvidia")

# We need to set the RPATH_DIRS here because for whatever
# reason when building our binaries theres a mix of system versus
# conda libraries being pulled in.
# We want to make this as consistent as possible across the board
# and use the torch libs installed under conda.
set(RPATH_DIRS
  "$ENV{CONDA_PREFIX}/lib"
  "$ENV{CONDA_PREFIX}/lib64"
  "${TORCH_LIBDIR}"
  "${TORCH_NVIDIA_ROOT}/cublas/lib"
  "${TORCH_NVIDIA_ROOT}/cusparse/lib"
  "${TORCH_NVIDIA_ROOT}/cufft/lib"
  "${TORCH_NVIDIA_ROOT}/curand/lib"
  "${TORCH_NVIDIA_ROOT}/cudnn/lib"
  "${TORCH_NVIDIA_ROOT}/nccl/lib"
  "${TORCH_NVIDIA_ROOT}/cuda_cupti/lib"
)

# prune non-existent dirs (optional)
foreach(p IN LISTS RPATH_DIRS)
  if(NOT EXISTS "${p}")
    list(REMOVE_ITEM RPATH_DIRS "${p}")
  endif()
endforeach()

list(JOIN RPATH_DIRS ":" ALL_RPATHS)

# Global prefs
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)        # embed at build time
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Apply to all targets we made
set(_targets jq_kernels ${TEST_SOURCES})

# rebuild list of test targets by name
set(_targets jq_kernels)
foreach(test_src ${TEST_SOURCES})
  get_filename_component(_t ${test_src} NAME_WE)
  list(APPEND _targets ${_t})
endforeach()

foreach(t ${_targets})
  set_target_properties(${t} PROPERTIES
    BUILD_RPATH   "${ALL_RPATHS}"
    INSTALL_RPATH "${ALL_RPATHS}"
  )
endforeach()
