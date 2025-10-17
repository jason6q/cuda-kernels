#!/usr/bin/bash

mkdir -p build && cd build
cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_PREFIX_PATH=/home/jq/Packages/miniconda3/envs/cuda-kernels/lib/python3.11/site-packages/torch/ \
    -DCUDAToolkit_INCLUDE_DIRS=/usr/local/cuda/include \
    -DENABLE_NVTX=ON
    #-DUSE_ASAN=ON

make -j
