#include <random>
#include <vector>
#include <cstdint>
#include <algorithm>

#include <cuda_runtime_api.h>

#include "core/data_ptr.h"
#include "core/device.h"
#include "tensor/tensor.h"
namespace jqTen{
    size_t compute_nbytes(const std::vector<int32_t>& shape, core::DType dtype){
        // Calculate bytes
        size_t size = 1;
        for(int32_t i = 0; i < shape.size(); ++i){
            size *= shape[i];
        }
        if(dtype == core::DType::FP32){
            size *= sizeof(float);
        }
        else{
            size *= sizeof(float);
        }

        return size;
    }

    Tensor::Tensor(
        const std::vector<int32_t>& shape, 
        core::DType dtype,
        core::Device device): 
        shape_(shape), dtype_(dtype), device_(device), nbytes_(compute_nbytes(shape_, dtype_)),
        data_ptr_(device == core::Device::CUDA ? core::DataPtr::cuda(nbytes_, true) : core::DataPtr::cpu(nbytes_, true, 64)) // 64 byte alignment
        {

    }

    void Tensor::to(core::Device device){
        this->data_ptr_.to(device);
        this->device_ = device;
    }

    Tensor empty(
        const std::vector<int32_t>& shape, 
        std::optional<core::DType> dtype,
        std::optional<core::Device> device
    ){
        core::DType dtype_ = dtype.value_or(core::DType::FP32);
        core::Device device_ = device.value_or(core::Device::CPU);

        return Tensor(shape, dtype_, device_);
    }

    Tensor random_uniform(
        const std::vector<int32_t>& shape, 
        std::optional<core::DType> dtype,
        std::optional<int64_t> seed,
        std::optional<core::Device> device
    ){
        core::Device device_ = device.value_or(core::Device::CPU);
        core::DType dtype_ = dtype.value_or(core::DType::FP32);
        int32_t seed_num = seed.value_or(42);

        std::mt19937 gen(seed_num) ;// Mersenne Twister appears to be highly uniform

        std::uniform_real_distribution<float> dist(0.0,1.0); // TODO: Handle different data types.

        // We'll init a CPU tensor than move it to CUDA.
        // TODO: RNG on CUDA variant?
        jqTen::Tensor tensor = empty(shape, dtype_, core::Device::CPU);
        auto buf_ptr = tensor.data();

        // Fill elements out
        int32_t num_elements = 1;
        for(int i = 0; i < shape.size(); ++i){
            num_elements *= shape[i];
        }
        // Todo handle types here. Is there a way to template this?
        float* buf = static_cast<float*>(buf_ptr);
        for(int i = 0; i < num_elements; ++i){
            buf[i] = dist(gen);
        }

        if(device_ == core::Device::CUDA) tensor.to(device_);

        return tensor;
    }
}