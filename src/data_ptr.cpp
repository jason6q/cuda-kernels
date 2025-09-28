#include <memory>
#include <cstddef>

#include "data_ptr.h"
#include "tensor.h"

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
            ::operator delete(buf, std::align_val_t(align));
        });


        DataPtr{Device::CPU, std::move(s_ptr), bytes, align};
    }

    DataPtr DataPtr::cuda(std::size_t bytes, bool zero, void* stream){
        // Allocate memory and ptr
        void* buf;
        cudaMalloc(&buf, bytes); // Automatically aligned >256B

        if(zero){
            cudaMemsetAsync(buf, 0, bytes, static_cast<cudaStream_t> stream)
        }

        // Instantiate shared_ptr with CUDA lambda deleter
        // Ownership of buf to shared_ptr
        auto s_ptr = std::shared_ptr<void>(buf, [align](void *q){
            cudaFree(buf);
        });

        DataPtr{Device::CUDA, std::move(s_ptr), bytes, align};
    }
}