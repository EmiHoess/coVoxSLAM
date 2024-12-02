#pragma once

#include "../ray_resident_compact/types.h"
#include <cuda_runtime.h>

namespace covoxslam::detail
{
__device__ inline unsigned long long reserve_compact_group(unsigned long long* counter)
{
    const auto mask = __activemask();
    const auto lane = threadIdx.x & 31;
    const auto first = __ffs(mask) - 1;
    unsigned long long base{};

    if (lane == first) base = atomicAdd(counter, static_cast<unsigned long long>(__popc(mask)));
    base = __shfl_sync(mask, base, first);
    return base + __popc(mask & ((1U << lane) - 1));
}

__device__ inline void compact_batched_add(cuda_resident_view view, cuda_resident_reduction_view reduction,
    cuda_compact_view compact, unsigned voxel, double weighted, double weight)
{
    const auto previous = atomicAdd(view.counts + voxel, 1U);
    atomicAdd(view.cursors + voxel, 1U);

    if (previous == UINT32_MAX) { atomicOr(&view.state->error, resident_contribution_capacity_error); return; }
    if (!previous)
    {
        const auto offset = reserve_compact_group(reinterpret_cast<unsigned long long*>(&view.state->batch_touched));

        if (offset >= std::uint64_t(view.tile_capacity) * 512) {
            atomicOr(&view.state->error, resident_scatter_error);
            return;
        }
        compact.touched[offset] = voxel;
    }
    atomicAdd(reduction.weighted_distances + voxel, weighted);
     atomicAdd(reduction.weights + voxel, weight);
}
}
