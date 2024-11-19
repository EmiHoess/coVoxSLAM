#pragma once

#include "../ray_resident_reduce/common.cuh"

namespace covoxslam::detail
{
__global__ void reduce_resident_indices(cuda_resident_view view, cuda_resident_reduction_view reduction,
    cuda_fusion_parameters config)
{
    const auto total = view.state->batch_contributions;

    if (atomicAdd(&view.state->error, 0U)) return;
    if (total > view.contribution_capacity || view.state->tile_count > view.tile_capacity) {
        atomicOr(&view.state->error, resident_contribution_capacity_error);
        return;
    }
    const auto stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;

    for (auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < total; index += stride)
    {
        const auto voxel = reduction.voxel_indices[index];

        if (voxel / 512 >= view.state->tile_count || !view.counts[voxel]) {
            atomicOr(&view.state->error, resident_scatter_error);
            continue;
        }
        const auto cursor = atomicAdd(view.cursors + voxel, 1U);
         if (cursor >= view.counts[voxel] || !add_resident_reduction(reduction, voxel, reduction.samples[index], config))
            atomicOr(&view.state->error, resident_numeric_error);
    }
}
}
