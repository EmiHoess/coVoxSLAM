#pragma once

#include "../sparse_mapping.h"
#include <cuda_runtime.h>

namespace covoxslam::detail
{
inline constexpr unsigned fusion_warp_mask = 0xffffffffU;

__device__ inline double fusion_previous_weight(cuda_tsdf_voxel voxel)
{
    return (voxel.flags & 1U) && voxel.weight > 0 && isfinite(voxel.weight) ? voxel.weight : 0;
}

__device__ inline double fusion_clamped_distance(float distance, cuda_fusion_parameters config) {
    return fminf(config.truncation_distance, fmaxf(-config.truncation_distance, distance));
}

__device__ inline bool finish_parallel_fusion(cuda_tsdf_voxel& voxel, double distance, double weight,
    std::uint32_t count, cuda_fusion_parameters config)
{
    if (!isfinite(distance) || !isfinite(weight) || weight <= 0) return false;
    voxel.distance = static_cast<float>(fmin(double(config.truncation_distance), fmax(-double(config.truncation_distance), distance)));
     voxel.weight = static_cast<float>(fmin(double(config.maximum_weight), weight));
    voxel.flags = 1U | (fabsf(voxel.distance) <= config.extraction_distance ? 2U : 0U);
    const auto samples = std::uint64_t(voxel.sample_count) + count;
    voxel.sample_count = samples > UINT32_MAX ? UINT32_MAX : static_cast<std::uint32_t>(samples);
    return true;
}

__device__ inline double fusion_warp_sum(double value)
{
    for (int offset = 16; offset; offset /= 2) value += __shfl_down_sync(fusion_warp_mask, value, offset);
    return value;
}
}
