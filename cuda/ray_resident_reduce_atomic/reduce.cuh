#pragma once

#include "../ray_resident_reduce/common.cuh"

namespace covoxslam::detail
{
__global__ void prepare_resident_sums(cuda_resident_view view, cuda_resident_reduction_view reduction,
    cuda_fusion_parameters config)
{
    const auto voxel = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (voxel >= static_cast<std::size_t>(view.tile_capacity) * 512 || atomicAdd(&view.state->error, 0U)) return;
    const auto count = view.counts[voxel];
    if (!count) return;
    if (view.state->tile_count > view.tile_capacity || voxel / 512 >= view.state->tile_count || view.cursors[voxel] != count)
    {
        atomicOr(&view.state->error, resident_scatter_error);
        return;
    }
    auto result = view.tiles[voxel / 512].voxels[voxel % 512];
    if (!finish_resident_reduction(result, reduction.weighted_distances[voxel], reduction.weights[voxel], count, config))
        atomicOr(&view.state->error, resident_numeric_error);
    else view.pending[voxel] = result;
}
}
