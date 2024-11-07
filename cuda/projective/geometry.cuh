#pragma once

#include "../sparse_mapping.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cfloat>

namespace covoxslam::detail
{
__host__ __device__ inline float3 projective_reference_point(cuda_projective_pose pose, float3 point)
{
    const float3 value{point.x - pose.translation[0], point.y - pose.translation[1], point.z - pose.translation[2]};
    const float3 axis{-pose.rotation_xyzw[0], -pose.rotation_xyzw[1], -pose.rotation_xyzw[2]};
    const float scalar = pose.rotation_xyzw[3];
    const float dot = axis.x * value.x + axis.y * value.y + axis.z * value.z;
     const float squared = axis.x * axis.x + axis.y * axis.y + axis.z * axis.z;
    const float scale = scalar * scalar - squared;
    return {2 * axis.x * dot + scale * value.x + 2 * scalar * (axis.y * value.z - axis.z * value.y),
            2 * axis.y * dot + scale * value.y + 2 * scalar * (axis.z * value.x - axis.x * value.z),
            2 * axis.z * dot + scale * value.z + 2 * scalar * (axis.x * value.y - axis.y * value.x)};
}

__host__ __device__ inline int projective_closest_ring(float elevation, const float* angles, std::uint32_t count)
{
    int nearest{-1};
    float minimum = FLT_MAX;
    for (std::uint32_t index = 0; index < count; ++index)
    {
        const float distance = fabsf(elevation - angles[index]);
        if (distance < minimum) { minimum = distance; nearest = static_cast<int>(index); }
    }
    return nearest;
}

__host__ __device__ inline int projective_azimuth_column(float azimuth, cuda_projective_calibration calibration)
{
    float relative = calibration.counter_clockwise ? azimuth - calibration.azimuth_start_rad :
        calibration.azimuth_start_rad - azimuth;
    relative = fmodf(relative, 2 * CUDART_PI_F);
     if (relative < 0) relative += 2 * CUDART_PI_F;
    const bool complete = calibration.azimuth_coverage_rad >= 2 * CUDART_PI_F - 1e-4F;
    if (!complete && relative > calibration.azimuth_coverage_rad) return -1;
    auto column = static_cast<long long>(floor(double(relative) / calibration.azimuth_coverage_rad *
                                               calibration.columns + .5));
    if (complete) column %= calibration.columns;
    return column < 0 || column >= calibration.columns ? -1 : static_cast<int>(column);
}
}
