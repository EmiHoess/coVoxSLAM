#pragma once

#include <covoxslam/sensor_grid_types.h>
#include <covoxslam/sensor_grid_precision.h>
#include <cfloat>
#include <cmath>

#ifdef __CUDACC__
#define covoxslam_sensor_callable __host__ __device__
#else
#define covoxslam_sensor_callable
#endif

namespace covoxslam::sensor_grid
{
template<class accumulator_type = double>
covoxslam_sensor_callable inline voxel gather_row(lookup_view table, const float* ranges,
                                                  std::uint32_t cell, gather_policy policy)
{
    voxel result{};
    accumulator_type weighted{}, weights{};
    const auto& c = table.settings;
    const auto begin = table.offsets[cell];
    const auto end = policy.first_entry_only && begin < table.offsets[cell + 1]
        ? begin + 1 : table.offsets[cell + 1];

    for (auto index = begin; index < end; ++index)
    {
        ++result.candidates_tested;
        const auto hit = table.intersections[index];
        const float range = ranges[hit.ray];

        if (!(range > 0) || range > FLT_MAX || range < c.minimum_range) continue;
        const bool clearing = range > c.radius;
        const float stop = clearing ? fminf(fmaxf(range - c.truncation, 0.f), c.radius)
                                    : fminf(range + c.truncation, c.radius);
        const float distance = range - hit.projected;

        if (hit.entry_t > stop || hit.projected < 0 || (!clearing && distance < -c.truncation)) continue;
        float weight = 1;
        if (c.weight_dropoff && distance < -c.voxel_size)
            weight = fmaxf((c.truncation + distance) / (c.truncation - c.voxel_size), 0.f);

        if (fabsf(distance) < c.truncation) weight *= c.sparsity_compensation;
        if (!(weight > FLT_EPSILON)) continue;
        const float clipped = fmaxf(-c.truncation, fminf(distance, c.truncation));
        weighted += accumulator_type(weight) * accumulator_type(clipped);
         weights += weight;
        ++result.contributions;

        if (policy.max_rays && result.contributions == policy.max_rays) break;
    }

    if (result.contributions)
    {
        result.distance = static_cast<float>(weighted / weights);
        if constexpr (sizeof(accumulator_type) == sizeof(float)) result.weight = fminf(weights, c.maximum_weight);
        else result.weight = static_cast<float>(fmin(weights, double(c.maximum_weight)));
    }
    return result;
}
}
#undef covoxslam_sensor_callable
