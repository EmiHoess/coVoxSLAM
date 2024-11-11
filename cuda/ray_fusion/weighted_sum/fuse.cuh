#pragma once

#include "../common.cuh"

namespace covoxslam::detail
{
__device__ inline bool fuse_weighted_sum_warp(cuda_tsdf_voxel& voxel, const cuda_ray_contribution* samples,
    std::uint32_t count, cuda_fusion_parameters config)
{
    const unsigned lane = threadIdx.x & 31U;
    double weights{}, distances{};
     bool valid = true;
    for (std::uint64_t index = lane; index < count; index += 32)
    {
        const auto sample = samples[index];
        if (!isfinite(sample.weight) || sample.weight <= 0 || !isfinite(sample.distance)) {
            valid = false;
            continue;
        }
        weights += sample.weight;
         distances += double(sample.weight) * fusion_clamped_distance(sample.distance, config);
    }
    weights = fusion_warp_sum(weights);
    distances = fusion_warp_sum(distances);
    valid = __all_sync(fusion_warp_mask, valid);
    if (!lane && valid)
    {
        const double previous = fusion_previous_weight(voxel);
        const double numerator = distances + (previous > 0 ? previous * voxel.distance : 0);
         valid = finish_parallel_fusion(voxel, numerator / (previous + weights), previous + weights, count, config);
    }
    return __shfl_sync(fusion_warp_mask, int(valid), 0) != 0;
}
}
