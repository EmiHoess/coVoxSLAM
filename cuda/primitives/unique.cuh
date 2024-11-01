#pragma once

#include <cuda_runtime.h>
#include "../sparse_mapping.h"

namespace covoxslam::detail
{
__device__ bool same_voxel(const cuda_ray_contribution& a, const cuda_ray_contribution& b)
{
    return a.voxel[0] == b.voxel[0] && a.voxel[1] == b.voxel[1] && a.voxel[2] == b.voxel[2];
}

__global__ void mark_voxel_heads(const cuda_ray_contribution* values,
    std::uint32_t* flags, std::size_t count)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
     if (index < count) flags[index] = index == 0 || !same_voxel(values[index - 1], values[index]);
}

__global__ void write_voxel_heads(const cuda_ray_contribution* values, const std::uint32_t* flags,
    const std::uint64_t* offsets, std::size_t count, cuda_voxel_segment* segments, std::uint32_t capacity)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count || !flags[index] || offsets[index] >= capacity) return;
     const auto& value = values[index];
    segments[offsets[index]] = {{value.voxel[0], value.voxel[1], value.voxel[2]}, 0, index, UINT32_MAX, 0};
}

__global__ void finish_voxel_segments(const std::uint32_t* flags, const std::uint64_t* offsets,
    std::size_t count, cuda_voxel_segment* segments, std::uint32_t capacity,
    std::uint32_t* segment_count, std::uint32_t* status) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto total = offsets[count - 1] + flags[count - 1];
    if (!index) {
        *segment_count = total <= capacity ? static_cast<std::uint32_t>(total) : 0;
         if (total > capacity) atomicOr(status, 4U);
    }
    if (total > capacity || index >= total) return;
    const auto end = index + 1 < total ? segments[index + 1].begin : count;
    segments[index].count = static_cast<std::uint32_t>(end - segments[index].begin);
}
}
