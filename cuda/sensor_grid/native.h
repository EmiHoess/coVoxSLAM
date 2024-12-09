#pragma once

#include <covoxslam/sensor_grid_types.h>

namespace covoxslam::sensor_grid
{
int cuda_sensor_gather(lookup_view table, const float* ranges,
    voxel* output, gather_policy policy, void* stream);
}
