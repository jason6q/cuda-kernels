#include <iostream>
#include <memory>

#include <cstddef>
#include <cuda_runtime.h>
#include <cstring>

#include "data_ptr.h"
#include "tensor.h"

// TODO: Maybe create a custom deleter class instead.

namespace jq{
    DataPtr DataPtr::cpu(std::size_t bytes, bool zero, std::size_t align){
        // Allocate memory and ptr
        void* buf = ::operator new(bytes, std::align_val_t(align)); // Alignment required for CPU

        if(zero){
            buf = std::memset(buf, 0, bytes);
        }

        // Instantiate shared_ptr with CPU lambda deleter
        // Ownership of buf to shared_ptr
        auto s_ptr = std::shared_ptr<void>(buf, [align](void *q){
            ::operator delete(q, std::align_val_t(align));
        });


        return DataPtr{Device::CPU, std::move(s_ptr), bytes};
    }

    DataPtr DataPtr::cuda(std::size_t bytes, bool zero){
        // Allocate memory and ptr
        void* buf = nullptr;
        cudaMalloc(&buf, bytes); // Automatically aligned >256B

        // TODO: Grab correct stream
        cudaStream_t stream = cudaStreamDefault;

        if(zero){
            cudaMemsetAsync(buf, 0, bytes, static_cast<cudaStream_t>(stream));
        }

        // Instantiate shared_ptr with CUDA lambda deleter
        // Ownership of buf to shared_ptr
        auto s_ptr = std::shared_ptr<void>(buf, [=](void* q){
            cudaFree(q);
        });

        return DataPtr{Device::CUDA, std::move(s_ptr), bytes};
    }

    void DataPtr::to(Device device){
        // TODO: Create deleter funcs instead
        if(this->device_ == device) return;

        // TODO: Add cuda checks.
        if(this->device_ == Device::CUDA){
            // CUDA -> CPU
            if(device == Device::CPU){
                size_t align = this->align;
                void* cuda_buf = this->get();
                void* cpu_buf = ::operator new(this->size_, std::align_val_t(align)); // Alignment required for CPU


                cudaMemcpy(cpu_buf, cuda_buf, this->size_, cudaMemcpyDeviceToHost);
                this->ptr_.reset(cpu_buf, [align](void* q){::operator delete(q, std::align_val_t(align));});
            }
        }
        else if(this->device_ == Device::CPU){
            // CPU -> CUDA
            if(device == Device::CUDA){
                void* cpu_buf = this->get();
                void* cuda_buf = nullptr;

                cudaError_t err = cudaMalloc(&cuda_buf, this->size_);
                if (err != cudaSuccess){
                    std::cerr << cudaGetErrorString(err) << std::endl;
                }
                cudaMemcpy(cuda_buf, cpu_buf, this->size_, cudaMemcpyHostToDevice);
                this->ptr_.reset(cuda_buf, [](void* q){cudaFree(q);});
            }
        }
        this->device_ = device;
    }
}