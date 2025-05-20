#pragma once
#include "mapping.h"
#include <cuda_runtime.h>

namespace covoxslam::store
{
using band_tsdf_view = storage::device_map_view<storage::schema::tsdf, true>;
using band_work_view = storage::device_map_view<storage::schema::projective_esdf>;
using band_output_view = storage::device_map_view<storage::schema::esdf>;

__device__ inline detail::cuda_site_esdf_voxel band_load(band_work_view values,
    std::size_t tile, unsigned voxel)
{
    const auto value = values[tile].voxels[voxel];
    return {value.squared_distance_voxels, {value.parent_x, value.parent_y, value.parent_z}, value.flags};
}
__device__ inline void band_store(band_work_view values, std::size_t tile,
    unsigned voxel, const detail::cuda_site_esdf_voxel& value)
{
    storage::scatter_record<storage::schema::projective_esdf>(values[tile].voxels[voxel],
        {value.squared_distance_voxels, value.parent[0], value.parent[1], value.parent[2], value.flags});
}
inline bool band_descriptor_valid(storage::device_descriptor value, storage::schema schema)
{
    return value.record_schema == schema && value.storage_layout == storage::layout::pool_soa &&
        value.generation && value.size_tiles <= value.capacity_tiles &&
        value.size_tiles <= UINT32_MAX / 512 && value.tile_stride == 2048 && value.voxel_stride == 4;
}
}
