#pragma once
#include "../sparse_mapping.h"
#include <cuda_runtime.h>

namespace covoxslam::detail
{
__host__ __device__ inline std::uint64_t resident_key(cuda_tile_coordinate key) noexcept
{
    constexpr int bias = 1 << 20;
    if (key.x < -bias || key.x >= bias || key.y < -bias || key.y >= bias || key.z < -bias || key.z >= bias)
        return UINT64_MAX;
    return (std::uint64_t(key.x + bias) << 42) | (std::uint64_t(key.y + bias) << 21) | std::uint64_t(key.z + bias);
}

__host__ __device__ inline std::uint64_t resident_hash(std::uint64_t key) noexcept {
    key = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9ULL;
     key = (key ^ (key >> 27)) * 0x94d049bb133111ebULL;
    return key ^ (key >> 31);
}
}
