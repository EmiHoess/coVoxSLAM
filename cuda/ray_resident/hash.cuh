#pragma once

#include "types.h"
#include "../common/tile_hash.cuh"
#include <cuda_runtime.h>

namespace covoxslam::detail
{

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
