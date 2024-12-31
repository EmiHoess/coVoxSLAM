#pragma once
#include "prepared.cuh"

namespace covoxslam::sensor_grid
{
struct staged_sample
{
    prepared_ray ray{};
    float entry_t{}, projected{};
};
__device__ inline staged_sample load_staged_sample(layout_view view, std::uint32_t index,
    const prepared_ray* prepared)
{
    return {prepared[view.ray_ids[index]], view.entry_t[index], view.projected[index]};
}
}
