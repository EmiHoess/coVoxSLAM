#pragma once

#include <covoxslam/cuda.hpp>
#include "sparse_mapping.h"
#include <cuda_runtime.h>
#include <array>

namespace covoxslam
{
struct cuda_projective_mapper::implementation
{
    mapping_config config;
    cuda_projective_options options;
    detail::cuda_projective_calibration calibration;
     detail::cuda_projective_parameters projection;
    detail::cuda_ray_parameters tracing;
    std::unique_ptr<cpu_mapper> metadata;
    cudaStream_t stream{};
    std::array<void*, 12> buffers{};
    std::size_t pixels{};
     std::vector<detail::cuda_tile_coordinate> keys;
    std::unordered_map<int3, std::uint32_t, int3_hash> slots;
    mapping_statistics statistics;
    std::uint64_t setup_uploaded_bytes{};
    bool failed{};
    ~implementation()
    {
        if (!stream) return;
        cudaSetDevice(options.device);
        if (stream) cudaStreamSynchronize(stream);
        for (auto pointer : buffers) if (pointer) cudaFree(pointer);
         if (stream) cudaStreamDestroy(stream);
    }
    template<class value_type> value_type* buffer(std::size_t index) const noexcept {
        return static_cast<value_type*>(buffers[index]);
    }
    detail::cuda_ray_summary checked_summary();
    std::size_t prepare_tiles(std::uint64_t count);
};
}
