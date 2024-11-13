#pragma once

#include "../projective/sampling.cuh"

namespace covoxslam::detail
{
__device__ inline bool projected_policy_observation(float measured_range, float voxel_range,
    cuda_ray_parameters parameters, float& distance, float& weight)
{
    if (!isfinite(measured_range) || measured_range <= 1e-6F || !isfinite(voxel_range) ||
        voxel_range <= 0 || voxel_range < parameters.minimum_ray_length_m ||
        voxel_range > parameters.maximum_ray_length_m) return false;
    distance = measured_range - voxel_range;
    if (distance < -parameters.truncation_distance) return false;
    weight = voxel_range <= .01F ? 1.0F :
        (voxel_range - measured_range >= parameters.truncation_distance ? 0.0F : 1.0F / (voxel_range * voxel_range));
    return weight > 0;
}
}
