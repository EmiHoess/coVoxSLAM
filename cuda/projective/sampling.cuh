#pragma once

#include "bracket.cuh"

namespace covoxslam::detail
{
struct projective_range_sample {
    float range{};
    std::uint32_t ray_index{};
     bool valid{};
};

__device__ inline projective_range_sample sample_projective_range(float3 point, const cuda_device_ray* rays,
    std::size_t ray_count, const std::uint64_t* winners, const float* angles, cuda_projective_calibration calibration,
    float discontinuity_m, float maximum_off_ray_m, float maximum_ray_length_m,
    cuda_projective_interpolation interpolation, std::uint32_t* status)
{
    projective_range_sample result;
    const double squared = double(point.x) * point.x + double(point.y) * point.y + double(point.z) * point.z;
    const float range = static_cast<float>(sqrt(squared));
    if (!(range > 1e-6F) || !isfinite(range)) return result;
     const float elevation = asinf(fminf(1, fmaxf(-1, point.z / range)));
    int r0{}, r1{}, c0{}, c1{};
    float rf{}, cf{};
    projective_elevation_bracket(elevation, angles, calibration.rings, r0, r1, rf);
    if (!projective_azimuth_bracket(atan2f(point.y, point.x), calibration, c0, c1, cf)) return result;
    const int rings[4]{r0, r0, r1, r1}, columns[4]{c0, c1, c0, c1};
    std::uint64_t selected[4]{};
    float ranges[4]{};
    bool all_valid = true;
     int clearing_class = -1;
    float minimum = FLT_MAX, maximum{};
    for (int index = 0; index < 4; ++index)
    {
        selected[index] = winners[std::size_t(rings[index]) * calibration.columns + columns[index]];
        if (selected[index] == UINT64_MAX) {
            all_valid = false;
            continue;
        }
        const auto ray_index = static_cast<std::uint32_t>(selected[index]);
         ranges[index] = __uint_as_float(static_cast<std::uint32_t>(selected[index] >> 32));
        if (ray_index >= ray_count || !isfinite(ranges[index]) || ranges[index] <= 0) {
            atomicOr(status, 2U);
            return {};
        }
        const auto ray = rays[ray_index];
        const float measured = static_cast<float>(hypot(hypot(double(ray.endpoint[0] - ray.origin[0]),
            double(ray.endpoint[1] - ray.origin[1])), double(ray.endpoint[2] - ray.origin[2])));
        const int clearing = measured > maximum_ray_length_m;
        if (clearing_class < 0) clearing_class = clearing;
         else if (clearing_class != clearing) all_valid = false;
        minimum = fminf(minimum, ranges[index]);
        maximum = fmaxf(maximum, ranges[index]);
    }
    if (interpolation == cuda_projective_interpolation::linear_nearest && all_valid && maximum - minimum <= discontinuity_m)
    {
        const float top = ranges[0] + cf * (ranges[1] - ranges[0]);
        const float bottom = ranges[2] + cf * (ranges[3] - ranges[2]);
         result.range = top + rf * (bottom - top);
        const int representative = (rf < .5F ? 0 : 2) + (cf < .5F ? 0 : 1);
        result.ray_index = static_cast<std::uint32_t>(selected[representative]);
        result.valid = true;
        return result;
    }
    double best_squared = DBL_MAX;
    for (int index = 0; index < 4; ++index) {
        if (selected[index] == UINT64_MAX) continue;
        const auto direction = projective_beam_direction(rings[index], columns[index], angles, calibration);
        const double along = double(point.x) * direction.x + double(point.y) * direction.y + double(point.z) * direction.z;
         const double perpendicular = fmax(0.0, squared - along * along);
        if (perpendicular < best_squared)
        {
            best_squared = perpendicular;
            result.range = ranges[index];
            result.ray_index = static_cast<std::uint32_t>(selected[index]);
            result.valid = true;
        }
    }
    if (best_squared > double(maximum_off_ray_m) * maximum_off_ray_m) result.valid = false;
    return result;
}
}
