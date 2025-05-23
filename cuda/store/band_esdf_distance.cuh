#pragma once
#include "band_esdf_views.cuh"
#include "../esdf_site/distance.cuh"

namespace covoxslam::store
{
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
    return detail::relax_site_distance(value, band_load(fields, tile, local),
        position[0], position[1], position[2], maximum_squared);
}
}
