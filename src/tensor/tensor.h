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