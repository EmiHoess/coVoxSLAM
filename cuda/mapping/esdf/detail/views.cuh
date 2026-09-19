#pragma once
#include "driver.cuh"
#include <covoxslam/voxel/esdf_encodings.hpp>
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
    const bool shape = value.record_schema == schema && value.storage_layout == storage::layout::pool_soa &&
        value.generation && value.size_tiles <= value.capacity_tiles &&
        value.capacity_tiles <= UINT32_MAX / tile_voxels && value.tile_stride == 2048 && value.voxel_stride == 4;
    if (!shape) return false;
    const unsigned fields = schema == storage::schema::tsdf ? 4 : 5;

    for (unsigned i = 0; i < fields; ++i)
    {
        const auto address = reinterpret_cast<std::uintptr_t>(value.fields[i]);
        if (!address || address % 4 || value.capacity_tiles > (UINTPTR_MAX - address) / 2048) return false;

        for (unsigned j = 0; j < i; ++j) {
            const auto other = reinterpret_cast<std::uintptr_t>(value.fields[j]);
            const auto bytes = value.capacity_tiles * 2048;
            if (address < other + bytes && other < address + bytes) return false;
        }
    }
    return true;
}
inline bool band_disjoint(storage::device_descriptor a, storage::device_descriptor b)
{
    const unsigned fields = a.record_schema == storage::schema::tsdf ? 4 : 5;

    for (unsigned i = 0; i < fields; ++i)

        for (unsigned j = 0; j < 5; ++j) {
            const auto x = reinterpret_cast<std::uintptr_t>(a.fields[i]);
             const auto y = reinterpret_cast<std::uintptr_t>(b.fields[j]);
            if (x < y + b.capacity_tiles * 2048 && y < x + a.capacity_tiles * 2048) return false;
        }
    return true;
}
inline bool band_index_valid(device_index_view index, storage::device_descriptor payload, unsigned field)
{
    int device{-1};
    return index.entries && index.capacity && !(index.capacity & (index.capacity - 1)) &&
        index.allocation_generation && index.field == field && index.device == payload.device &&
        cudaGetDevice(&device) == cudaSuccess && device == index.device;
}
}
