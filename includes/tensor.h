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

#include "data_ptr.h"
#include "device.h"
namespace jq{
    /**
     * Try to have this mimic the ATen Tensor minimally.
     **/
    class Tensor{
        public:
            Tensor(const std::vector<int32_t>& shape, 
                DType dtype = DType::FP32, 
                Device device = Device::CPU);

            // Move the underlying data to a device.
            void to(Device device);

            // Getters
            const DataPtr& data_ptr() const { return data_ptr_; }
            const std::vector<int32_t>& shape() const { return shape_; }
            DType dtype() const { return dtype_; }
            Device device() const { return device_; }

        private:
            // Order matters here since data_ptr_ depends on its above member
            // variables during default construction.
            std::vector<int32_t> shape_;
            DType dtype_;
            Device device_;
            size_t nbytes_;
            DataPtr data_ptr_;
    };

    Tensor empty(
        const std::vector<int32_t>& shape, 
        std::optional<DType> dtype = std::nullopt,
        std::optional<Device> device = std::nullopt
    );
    Tensor random_uniform(
        const std::vector<int32_t> &shape,
        std::optional<DType> dtype = std::nullopt,
        std::optional<int64_t> seed = std::nullopt,
        std::optional<Device> device = std::nullopt);
}