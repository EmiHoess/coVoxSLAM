#pragma once
#include <covoxslam/sensor_grid_layout_types.h>

#include <covoxslam/sensor_grid_types.h>

namespace covoxslam::sensor_grid
{
int cuda_sensor_prepared(layout_view view, const float* ranges, prepared_ray* prepared, voxel* output,
    config settings, std::uint32_t ray_count, gather_backend backend, gather_policy policy, void* stream,
    void* profile_events = nullptr);
int cuda_sensor_gather(lookup_view table, const float* ranges,
    voxel* output, gather_policy policy, void* stream);
}
