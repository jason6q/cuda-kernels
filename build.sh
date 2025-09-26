#!/usr/bin/bash

mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/home/jq/Packages/miniconda3/envs/cuda-kernels/lib/python3.11/site-packages/torch/
make -j