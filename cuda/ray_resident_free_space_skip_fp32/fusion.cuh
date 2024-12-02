#pragma once

#include "types.h"
#include "../ray_resident_reduce/common.cuh"

namespace covoxslam::detail
{
__device__ inline void resident_fp32_add(cuda_resident_view view, cuda_fp32_view sums,
    unsigned voxel, float distance, float weight)
{
    const auto previous = atomicAdd(view.counts + voxel, 1U);
    atomicAdd(view.cursors + voxel, 1U);

    if (previous == UINT32_MAX) atomicOr(&view.state->error, resident_contribution_capacity_error);
    atomicAdd(sums.weighted_distances + voxel, __fmul_rn(distance, weight));
     atomicAdd(sums.weights + voxel, weight);
}

static __global__ void prepare_resident_fp32(cuda_resident_view view, cuda_fp32_view sums, cuda_fusion_parameters config)
{
    const auto voxel = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (voxel >= std::size_t(view.tile_capacity) * 512 || atomicAdd(&view.state->error, 0U)) return;
    const auto count = view.counts[voxel];

    if (view.cursors[voxel] != count || view.state->tile_count > view.tile_capacity || (count && voxel / 512 >= view.state->tile_count)) {
        atomicOr(&view.state->error, resident_scatter_error);
        return;
    }
    if (!count) return;
    auto value = view.tiles[voxel / 512].voxels[voxel % 512];

    if (!finish_resident_reduction(value, sums.weighted_distances[voxel], sums.weights[voxel], count, config))
        atomicOr(&view.state->error, resident_numeric_error);
    else view.pending[voxel] = value;
    atomicAdd(reinterpret_cast<unsigned long long*>(&view.state->batch_validated), static_cast<unsigned long long>(count));
}
}
