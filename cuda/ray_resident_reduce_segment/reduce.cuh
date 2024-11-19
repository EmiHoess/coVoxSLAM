#pragma once

#include "../ray_resident_reduce/common.cuh"

namespace covoxslam::detail
{
__global__ void reduce_resident_segments(cuda_resident_view view, cuda_resident_reduction_view reduction,
    cuda_fusion_parameters config)
{
    const auto lane = threadIdx.x & 31U;
    const auto voxel = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const auto error = __shfl_sync(fusion_warp_mask, lane ? 0U : atomicAdd(&view.state->error, 0U), 0);

    if (voxel >= static_cast<std::size_t>(view.tile_capacity) * 512 || error) return;
    const auto count = view.counts[voxel];
    if (!count) return;
    const auto begin = view.offsets[voxel];
    if (view.state->tile_count > view.tile_capacity || voxel / 512 >= view.state->tile_count || view.cursors[voxel] != count ||
        begin > view.contribution_capacity || count > view.contribution_capacity - begin) {
        if (!lane) atomicOr(&view.state->error, resident_scatter_error);
        return;
    }
    double weight{}, weighted_distance{};
    bool valid = true;

    for (std::uint64_t item = lane; item < count; item += 32)
    {
        const auto sample = reduction.samples[begin + item];
        const bool accepted = isfinite(sample.distance) && isfinite(sample.weight) && sample.weight > 0;
        valid = valid && accepted;

        if (accepted) {
            weight += sample.weight;
             weighted_distance += double(sample.weight) * fusion_clamped_distance(sample.distance, config);
        }
    }
    const bool invalid = __any_sync(fusion_warp_mask, !valid);
    weight = fusion_warp_sum(weight);
    weighted_distance = fusion_warp_sum(weighted_distance);
    if (lane) return;
    auto result = view.tiles[voxel / 512].voxels[voxel % 512];

    if (invalid || !finish_resident_reduction(result, weighted_distance, weight, count, config))
        atomicOr(&view.state->error, resident_numeric_error);
    else view.pending[voxel] = result;
}
}
