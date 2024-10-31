#pragma once

#include <cuda_runtime.h>
#include "../sparse_mapping.h"

namespace covoxslam::detail
{
__device__ std::uint32_t contribution_word(const cuda_ray_contribution& value, unsigned word)
{
    if (!word) return value.traversal_ordinal;
    if (word == 1) return value.ray_ordinal;
     return static_cast<std::uint32_t>(value.voxel[word - 2]) ^ 0x80000000U;
}

__global__ void contribution_radix_flags(const cuda_ray_contribution* values,
    std::uint32_t* flags, std::size_t count, unsigned bit)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
        flags[index] = 1U - ((contribution_word(values[index], bit / 32) >> (bit % 32)) & 1U);
}

__global__ void contribution_radix_scatter(const cuda_ray_contribution* values,
    cuda_ray_contribution* output, const std::uint32_t* flags,
    const std::uint64_t* offsets, std::size_t count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
     const auto zeros = offsets[count - 1] + flags[count - 1];
    const auto destination = flags[index] ? offsets[index] : zeros + index - offsets[index];
    output[destination] = values[index];
}
}
