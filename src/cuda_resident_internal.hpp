#pragma once

#include <covoxslam/cuda_resident.hpp>
#include "ray_resident/types.h"
#include <cuda_runtime_api.h>
#include <array>

namespace covoxslam
{
struct cuda_resident_context::implementation
{
    mapping_config config;
    cuda_resident_options options;
    detail::cuda_resident_view view;
     detail::cuda_device_ray* rays{};
    cudaStream_t stream{};
    std::array<void*, 12> allocations{};
    std::size_t allocation_count{}, bytes{};
    bool failed{};

    operation_status check(int code)
    {
        if (code) {
            failed = true;
            return std::unexpected(operation_error{cudaGetErrorString(static_cast<cudaError_t>(code))});
        }
        return {};
    }

    template<class value_type>
    operation_status allocate(value_type*& pointer, std::size_t count)
    {

        if (allocation_count == allocations.size())
            return std::unexpected(operation_error{"resident allocation table is full"});
        auto& allocation = allocations[allocation_count];
        const auto result = check(cudaMalloc(&allocation, count * sizeof(value_type)));

        if (!result) return result;
        pointer = static_cast<value_type*>(allocation);
         ++allocation_count;
        return {};
    }

    ~implementation()
    {
        cudaSetDevice(options.device);
        if (stream) cudaStreamSynchronize(stream);

        for (auto allocation : allocations)

            if (allocation) cudaFree(allocation);
        if (stream) cudaStreamDestroy(stream);
    }
};
}
