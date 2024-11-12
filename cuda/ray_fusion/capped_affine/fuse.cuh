#pragma once

#include "../common.cuh"

namespace covoxslam::detail
{
__device__ inline bool fuse_capped_affine_warp(cuda_tsdf_voxel& voxel, const cuda_ray_contribution* samples,
    std::uint32_t count, cuda_fusion_parameters config)
{
    const unsigned lane = threadIdx.x & 31U;
    double weight = fusion_previous_weight(voxel);
    double distance = weight > 0 ? voxel.distance : 0;
     bool valid = isfinite(distance);
    for (std::uint64_t base = 0; base < count; base += 32)
    {
        const auto index = base + lane;
        double incoming{}, sample_distance{};
        if (index < count) {
            const auto sample = samples[index];
            if (!isfinite(sample.weight) || sample.weight <= 0 || !isfinite(sample.distance)) valid = false;
            else {
                incoming = sample.weight;
                 sample_distance = fusion_clamped_distance(sample.distance, config);
            }
        }
        double prefix = incoming;
        for (unsigned offset = 1; offset < 32; offset *= 2) {
            const double left = __shfl_up_sync(fusion_warp_mask, prefix, offset);
            if (lane >= offset) prefix += left;
        }
        double before = __shfl_up_sync(fusion_warp_mask, prefix, 1);
         if (!lane) before = 0;
        const double previous = index == 0 ? weight : fmin(double(config.maximum_weight), weight + before);
        double a = 1, b{};
        if (index < count && incoming > 0)
        {
            const double denominator = previous + incoming;
            a = previous / denominator;
            b = incoming * sample_distance / denominator;
        }
        for (unsigned offset = 1; offset < 32; offset *= 2)
        {
            const double left_a = __shfl_up_sync(fusion_warp_mask, a, offset);
            const double left_b = __shfl_up_sync(fusion_warp_mask, b, offset);
            if (lane >= offset) {
                b = a * left_b + b;
                 a *= left_a;
            }
        }
        distance = __shfl_sync(fusion_warp_mask, a, 31) * distance + __shfl_sync(fusion_warp_mask, b, 31);
        weight = fmin(double(config.maximum_weight), weight + __shfl_sync(fusion_warp_mask, prefix, 31));
    }
    valid = __all_sync(fusion_warp_mask, valid);
    if (!lane && valid) valid = finish_parallel_fusion(voxel, distance, weight, count, config);
    return __shfl_sync(fusion_warp_mask, int(valid), 0) != 0;
}
}
