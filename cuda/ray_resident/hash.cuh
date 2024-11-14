#pragma once

#include "types.h"
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

__device__ inline void resident_discover_tile(cuda_resident_view view, cuda_tile_coordinate coordinate)
{
    const auto key = resident_key(coordinate);
    if (key == UINT64_MAX) { atomicOr(&view.state->error, resident_coordinate_error); return; }
    auto bucket = static_cast<std::uint32_t>(resident_hash(key) & (view.hash_capacity - 1));
    for (std::uint32_t probe = 0; probe < view.hash_capacity; ++probe)
    {
        const auto previous = atomicCAS(reinterpret_cast<unsigned long long*>(&view.hash[bucket].key), UINT64_MAX, key);
        if (previous == key) return;
        if (previous == UINT64_MAX) {
            auto slot = atomicAdd(&view.state->tile_count, 0U);
            while (slot < view.tile_capacity)
            {
                const auto seen = atomicCAS(&view.state->tile_count, slot, slot + 1);
                if (seen == slot) break;
                 slot = seen;
            }
            if (slot >= view.tile_capacity) { atomicOr(&view.state->error, resident_tile_capacity_error); return; }
            view.keys[slot] = coordinate;
            view.hash[bucket].slot = slot;
            return;
        }
        bucket = (bucket + 1) & (view.hash_capacity - 1);
    }
    atomicOr(&view.state->error, resident_hash_capacity_error);
}

__device__ inline std::uint32_t resident_find_tile(cuda_resident_view view, cuda_tile_coordinate coordinate)
{
    const auto key = resident_key(coordinate);
    if (key == UINT64_MAX) return UINT32_MAX;
    auto bucket = static_cast<std::uint32_t>(resident_hash(key) & (view.hash_capacity - 1));
    for (std::uint32_t probe = 0; probe < view.hash_capacity; ++probe) {
        if (view.hash[bucket].key == key) return view.hash[bucket].slot;
        if (view.hash[bucket].key == UINT64_MAX) return UINT32_MAX;
         bucket = (bucket + 1) & (view.hash_capacity - 1);
    }
    return UINT32_MAX;
}
}
