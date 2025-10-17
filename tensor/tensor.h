/**
 * 
 * Custom Tensor Library.
 * Keep this as minimal as possible. Using this mainly
 * for testing purposes.
 */
#pragma once
#include <vector>
#include <cstdint>
#include <optional>
#include <iostream>

#include "tensor/macros.h"
#include "core/data_ptr.h"
#include "core/device.h"

namespace jqTen{
    /**
     * Try to have this mimic the ATen Tensor minimally.
     **/
    class Tensor{
        public:
            Tensor(const std::vector<int32_t>& shape, 
                core::DType dtype = core::DType::FP32, 
                core::Device device = core::Device::CPU);

            // Move the underlying data to a device.
            void to(core::Device device);

            // Getters
            void* data() { return data_ptr_.get(); }
            const void* data() const {return data_ptr_.get(); }
            size_t nbytes() const {return nbytes_; }

            const std::vector<int32_t>& shape() const { return shape_; }
            core::DType dtype() const { return dtype_; }
            core::Device device() const { return device_; }

            // Other
            int32_t numel() const {
                int32_t numel = 1;
                for(int i = 0; i < shape_.size(); ++i){
                    numel *= shape_[i];
                }

                return numel;
            }

            void print() const {
                // TODO: Make this based off DType
                // Would it be safe to handle the raw pointer like this or should everything
                // be shared.
                //// Not really the most ideal print method right now.
                if(this->device_ == core::Device::CPU){
                    const float* buf = static_cast<const float*>(this->data());
                    for(int i = 0; i < this->numel(); ++i){
                        std::cout << buf[i] << " ";
                    }
                    std::cout << std::endl;
                }
                else if(this->device_ == core::Device::CUDA){
                    // Move to CPU I guess.
                    // TODO: Type logic
                    auto data_buf = std::make_unique<float[]>(this->numel());

                    JQ_ASSERT_CUDA_ERR_CHECK(cudaMemcpy(data_buf.get(), this->data(), this->nbytes_, cudaMemcpyDeviceToHost));
                    for(int i = 0; i < this->numel(); ++i){
                        std::cout << data_buf.get()[i] << " ";
                    }
                    std::cout << std::endl;
                }
            }

        private:
            // Order matters here since data_ptr_ depends on its above member
            // variables during default construction.
            std::vector<int32_t> shape_;
            core::DType dtype_;
            core::Device device_;
            size_t nbytes_;
            core::DataPtr data_ptr_;
    };

    Tensor empty(
        const std::vector<int32_t>& shape, 
        std::optional<core::DType> dtype = std::nullopt,
        std::optional<core::Device> device = std::nullopt
    );
    Tensor random_uniform(
        const std::vector<int32_t> &shape,
        std::optional<core::DType> dtype = std::nullopt,
        std::optional<int64_t> seed = std::nullopt,
        std::optional<core::Device> device = std::nullopt);
}