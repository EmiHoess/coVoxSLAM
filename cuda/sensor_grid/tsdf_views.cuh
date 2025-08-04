#pragma once
#include <covoxslam/storage/device_view.hpp>
#include "../common/mapping_types.h"

namespace covoxslam::sensor_grid
{
using sensor_tsdf_view = storage::device_map_view<storage::schema::tsdf>;
__device__ inline detail::cuda_tsdf_voxel load_sensor_voxel(
    sensor_tsdf_view tiles, unsigned slot, unsigned local) {
    const auto value = tiles[slot].voxels[local];
     return {value.distance, value.weight, value.sample_count, value.flags};
}
__device__ inline void store_sensor_voxel(sensor_tsdf_view tiles,
    unsigned slot, unsigned local, detail::cuda_tsdf_voxel value)
{
    auto target = tiles[slot].voxels[local];
    target.distance = value.distance;
     target.weight = value.weight;
    target.sample_count = value.sample_count;
    target.flags = value.flags;
}
}
