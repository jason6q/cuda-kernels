/**
 * Custom Data Pointer class to be used with a Tensor class.
 * Mainly just for managing a smart pointer with different
 * deallocation methods based off hardware.
 */
#pragma once

#include <memory>
#include <cstddef>

#include "device.h"

namespace jq{
    /*
        Contains smart pointer to memory buffer
        and manages allocation/deallocation for CPU
        and CUDA.

        Factory for CPU and CUDA for self-documenting
        rather than single constructor
    */
    class DataPtr{
        public:
            static DataPtr cpu(std::size_t bytes, bool zero, std::size_t align);
            static DataPtr cuda(std::size_t bytes, bool zero); // Type-erasure on stream, but will use cudaStream_t

            void* get(){ return ptr_.get(); }
            Device device(){ return device_; }
            std::size_t size(){ return size_; }

        private:
            DataPtr(Device d, std::shared_ptr<void> ptr, std::size_t bytes)
            : device_(d), ptr_(ptr), size_(bytes) {}
            // Use shared_ptr<void> for type-erasure. That way
            // we only instantiate one instance.
            // unique_ptr enforces explicit deleter with
            // different types.
            std::shared_ptr<void> ptr_;

            // Make sure to use size_t for 32-bit or 64-bit addressable
            // memory.
            std::size_t size_;

            Device device_;
    };
}