#pragma once
#include <covoxslam/sensor_grid_layout_types.h>
#include <covoxslam/sensor_grid_precision.h>
#include <cuda_runtime.h>

namespace covoxslam::sensor_grid
{
__device__ inline bool prepared_contribution(prepared_ray ray, float entry, float projected,
    config settings, float& weight, float& clipped)
{
    if (!(ray.flags & 1U)) return false;
    const float distance = ray.range - projected;

    if (!(entry <= ray.end) || !(projected >= 0) || (!(ray.flags & 2U) && distance < -settings.truncation)) return false;
    weight = 1;
    if (settings.weight_dropoff && distance < -settings.voxel_size)
        weight = fmaxf((settings.truncation + distance) / (settings.truncation - settings.voxel_size), 0.f);

    if (fabsf(distance) < settings.truncation) weight *= settings.sparsity_compensation;
    if (!(weight > FLT_EPSILON)) return false;
    clipped = fmaxf(-settings.truncation, fminf(distance, settings.truncation));
    return true;
}
__device__ inline voxel finish_prepared(gather_accumulator weighted, gather_accumulator weights,
    std::uint32_t admitted, std::uint32_t tested, config settings)
{
    voxel result{0, 0, admitted, tested};

    if (admitted) {
        result.distance = static_cast<float>(weighted / weights);
        if (isfinite(result.distance))
            result.distance = fmaxf(-settings.truncation, fminf(result.distance, settings.truncation));
        if constexpr (accumulation_bits == 32) result.weight = fminf(weights, settings.maximum_weight);
        else result.weight = static_cast<float>(fmin(weights, double(settings.maximum_weight)));
    }
    return result;
}
}
