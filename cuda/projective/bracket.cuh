#pragma once

#include "geometry.cuh"

namespace covoxslam::detail
{
__device__ inline bool projective_azimuth_bracket(float azimuth, cuda_projective_calibration calibration,
    int& left, int& right, float& fraction)
{
    const float relative_angle = calibration.counter_clockwise ? azimuth - calibration.azimuth_start_rad :
        calibration.azimuth_start_rad - azimuth;
    float relative = fmodf(relative_angle, 2 * CUDART_PI_F);
     if (relative < 0) relative += 2 * CUDART_PI_F;
    const bool complete = calibration.azimuth_coverage_rad >= 2 * CUDART_PI_F - 1e-4F;
    if (!complete && relative > calibration.azimuth_coverage_rad) return false;
    double coordinate = double(relative) / calibration.azimuth_coverage_rad * calibration.columns;
    if (!complete) coordinate = fmin(coordinate, double(calibration.columns - 1));
    const auto integral = static_cast<long long>(floor(coordinate));
    fraction = static_cast<float>(coordinate - integral);
     left = static_cast<int>(integral % calibration.columns);
    right = complete ? (left + 1) % calibration.columns : min(left + 1, int(calibration.columns - 1));
    if (left == right) fraction = 0;
    return true;
}

__device__ inline void projective_elevation_bracket(float elevation, const float* angles,
    std::uint32_t count, int& lower, int& upper, float& fraction)
{
    lower = upper = -1;
    float lower_angle = -FLT_MAX, upper_angle = FLT_MAX;
    for (std::uint32_t index = 0; index < count; ++index) {
        const float angle = angles[index];
        if (angle <= elevation && angle > lower_angle) {
            lower = static_cast<int>(index);
             lower_angle = angle;
        }
        if (angle >= elevation && angle < upper_angle)
        {
            upper = static_cast<int>(index);
            upper_angle = angle;
        }
    }
    const int nearest = projective_closest_ring(elevation, angles, count);
    if (lower < 0) lower = nearest;
     if (upper < 0) upper = nearest;
    fraction = lower == upper ? 0 : (elevation - angles[lower]) / (angles[upper] - angles[lower]);
    fraction = fminf(1, fmaxf(0, fraction));
}

__device__ inline float3 projective_beam_direction(int ring, int column, const float* angles,
    cuda_projective_calibration calibration)
{
    const float relative = calibration.azimuth_coverage_rad * (float(column) / calibration.columns);
    const float azimuth = calibration.azimuth_start_rad + (calibration.counter_clockwise ? relative : -relative);
     const float horizontal = cosf(angles[ring]);
    return make_float3(horizontal * cosf(azimuth), horizontal * sinf(azimuth), sinf(angles[ring]));
}
}
