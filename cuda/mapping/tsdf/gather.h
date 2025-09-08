#pragma once
#include <covoxslam/sensor_grid_layout_types.h>

#include <covoxslam/sensor/detail/types.hpp>

namespace covoxslam::sensor_grid
{
int cuda_sensor_combined(layout_view view, const float* ranges, compact_ray* prepared, voxel* output,
    config settings, std::uint32_t ray_count, gather_policy policy, void* stream);
}
