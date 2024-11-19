#pragma once

#include "types.h"
#include "../ray_fusion/common.cuh"

namespace covoxslam::detail
{
__device__ inline bool finish_resident_reduction(cuda_tsdf_voxel& voxel, double weighted_distance,
    double weight, std::uint32_t count, cuda_fusion_parameters config)
{

    if (!count || !isfinite(weighted_distance) || !isfinite(weight) || weight <= 0) return false;
    const double previous = fusion_previous_weight(voxel);
    const double numerator = weighted_distance + (previous > 0 ? previous * voxel.distance : 0);
     return finish_parallel_fusion(voxel, numerator / (previous + weight), previous + weight, count, config);
}

__device__ inline bool add_resident_reduction(cuda_resident_reduction_view reduction, std::size_t voxel,
    cuda_resident_reduction_sample sample, cuda_fusion_parameters config)
{
    if (!isfinite(sample.distance) || !isfinite(sample.weight) || sample.weight <= 0) return false;
    const double weighted_distance = double(sample.weight) * fusion_clamped_distance(sample.distance, config);
    atomicAdd(reduction.weights + voxel, double(sample.weight));
     atomicAdd(reduction.weighted_distances + voxel, weighted_distance);
    return true;
}
}
