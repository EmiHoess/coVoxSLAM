#pragma once

#include "policy.cuh"

namespace covoxslam::detail
{
__device__ inline projective_range_sample sample_projected_policy(float3 point, std::size_t ray_count,
    const std::uint64_t* winners, const float* angles, cuda_projective_calibration calibration,
    float voxel_size, std::uint32_t* status)
{
    const float range = static_cast<float>(hypot(hypot(double(point.x), double(point.y)), double(point.z)));
    if (!isfinite(range) || range <= 1e-6F) return {};
    const float elevation = asinf(fminf(1, fmaxf(-1, point.z / range)));
    int r0{}, r1{}, c0{}, c1{};
    float rf{}, cf{};
    projective_elevation_bracket(elevation, angles, calibration.rings, r0, r1, rf);
    bool linear_rows = r0 != r1;
    if (!linear_rows && calibration.rings > 1) {
        float spacing = CUDART_INF_F, higher_angle = CUDART_INF_F;
        int higher = -1;
        for (std::uint32_t ring = 0; ring < calibration.rings; ++ring) {
            if (int(ring) == r0) continue;
            spacing = fminf(spacing, fabsf(angles[ring] - angles[r0]));
            if (angles[ring] > angles[r0] && angles[ring] < higher_angle) {
                higher = static_cast<int>(ring);
                higher_angle = angles[ring];
            }
        }
        if (fabsf(elevation - angles[r0]) > .5F * spacing) return {};
        if (elevation == angles[r0] && higher >= 0) { r1 = higher; rf = 0; linear_rows = true; }
    }
    if (!projective_azimuth_bracket(atan2f(point.y, point.x), calibration, c0, c1, cf)) return {};
    const int nearest_ring = rf < .5F ? r0 : r1, nearest_column = cf < .5F ? c0 : c1;
    const auto nearest = winners[std::size_t(nearest_ring) * calibration.columns + nearest_column];
    if (linear_rows && c1 == c0 + 1) {
        const int rings[4]{r0, r0, r1, r1}, columns[4]{c0, c1, c0, c1};
        float depths[4]{};
        bool valid = true;
        for (int index = 0; index < 4; ++index) {
            const auto value = winners[std::size_t(rings[index]) * calibration.columns + columns[index]];
            if (value == UINT64_MAX) { valid = false; continue; }
            depths[index] = __uint_as_float(static_cast<std::uint32_t>(value >> 32));
            if (static_cast<std::uint32_t>(value) >= ray_count || !isfinite(depths[index]) || depths[index] <= 1e-6F) {
                atomicOr(status, 2U);
                return {};
            }
        }
        if (valid) {
            const float top = depths[0] + cf * (depths[1] - depths[0]);
            const float bottom = depths[2] + cf * (depths[3] - depths[2]);
            const float interpolated = top + rf * (bottom - top);
            for (int index = 0; index < 4; ++index)
                valid = valid && fabsf(depths[index] - interpolated) <= 2 * voxel_size;
            if (valid) return {interpolated, static_cast<std::uint32_t>(nearest), true};
        }
    }
    if (nearest == UINT64_MAX) return {};
    const float depth = __uint_as_float(static_cast<std::uint32_t>(nearest >> 32));
    if (static_cast<std::uint32_t>(nearest) >= ray_count || !isfinite(depth) || depth <= 1e-6F) {
        atomicOr(status, 2U);
        return {};
    }
    const auto direction = projective_beam_direction(nearest_ring, nearest_column, angles, calibration);
    const double along = double(point.x) * direction.x + double(point.y) * direction.y + double(point.z) * direction.z;
    const double x = point.x - along * direction.x, y = point.y - along * direction.y, z = point.z - along * direction.z;
    if (x * x + y * y + z * z > .25 * double(voxel_size) * voxel_size) return {};
    return {depth, static_cast<std::uint32_t>(nearest), true};
}
}
