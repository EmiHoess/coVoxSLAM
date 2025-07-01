#pragma once

#include "../common/geometry.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cfloat>

namespace covoxslam::detail
{

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
