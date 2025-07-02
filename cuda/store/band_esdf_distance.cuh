#pragma once
#include "band_esdf_views.cuh"
#include "../common/esdf_geometry.cuh"

namespace covoxslam::store
{
__device__ inline bool band_mutable(const detail::cuda_site_esdf_voxel& value)
{
    return (value.flags & detail::cuda_esdf_observed) && !(value.flags & detail::cuda_esdf_site);
}
__device__ inline void assign_band_site(detail::cuda_site_esdf_voxel& value)
{
    if (value.flags & detail::cuda_esdf_site) {
        value.squared_distance_voxels = 0;
        value.flags |= detail::cuda_esdf_parent_assigned;
    }
}
__device__ inline bool relax_band_value(detail::cuda_site_esdf_voxel& value,
    band_work_view fields, std::size_t tile, unsigned local, const int* position,
    float maximum_squared)
{
    if (!band_mutable(value)) return false;
    return detail::relax_site_distance(value, band_load(fields, tile, local),
        position[0], position[1], position[2], maximum_squared);
}
}
