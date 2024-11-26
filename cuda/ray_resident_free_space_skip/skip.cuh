#pragma once

#include "types.h"
#include "../ray_resident/types.h"
#include <cuda_runtime.h>

namespace covoxslam::detail
{
__device__ inline bool resident_free_space_eligible(cuda_tsdf_voxel value,
    float truncation, float minimum_weight)
{
    return (value.flags & 1U) && isfinite(value.weight) && value.weight >= minimum_weight &&
        value.distance == truncation;
}

__device__ inline bool resident_skip_free_space(cuda_free_space_view skip,
    unsigned voxel, float signed_distance, float truncation)
{
    return !skip.refresh && signed_distance > truncation && skip.flags[voxel] != 0;
}

static __global__ void publish_resident_free_space(cuda_resident_view view,
    cuda_free_space_view skip, float truncation)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= static_cast<std::size_t>(view.tile_capacity) * 512 || view.state->error) return;
    const auto count = view.counts[index];

    if (count) {
        const auto value = view.pending[index];
        view.tiles[index / 512].voxels[index % 512] = value;
        skip.flags[index] = resident_free_space_eligible(value, truncation, skip.minimum_weight);
        atomicAdd(reinterpret_cast<unsigned long long*>(&view.state->batch_touched), 1ULL);
         atomicMax(&view.state->maximum_segment_length, count);
    }
}
}
