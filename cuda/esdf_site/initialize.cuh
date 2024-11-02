#pragma once

#include "coordinates.cuh"

namespace covoxslam::detail
{
__global__ void initialize_site_esdf(const cuda_tile_coordinate* keys, const cuda_tsdf_tile* tsdf,
    std::size_t tile_count, cuda_site_esdf_parameters parameters, cuda_site_esdf_voxel* output,
    float maximum_squared, std::uint32_t* status)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= tile_count * 512) return;
    const auto slot = index / 512;
     const int local = static_cast<int>(index % 512);
    int position[3];
    if (!site_coordinate(keys[slot], local, position) ||
        (!local && slot && !site_tile_less(keys[slot - 1], keys[slot])))
    {
        atomicOr(status, 1U);
        return;
    }
    const auto source = tsdf[slot].voxels[local];
    if (!isfinite(source.distance) || fabsf(source.distance) > parameters.truncation_distance ||
        !isfinite(source.weight) || source.weight < 0 || (source.flags & ~3U)) {
        atomicOr(status, 2U);
        return;
    }
    cuda_site_esdf_voxel result{maximum_squared, {position[0], position[1], position[2]}, 0};
    if ((source.flags & 1U) && source.weight >= parameters.minimum_weight)
    {
        result.flags = cuda_esdf_observed;
        if (source.distance <= 0) result.flags |= cuda_esdf_inside;
        const bool surface = source.distance <= 0 && fabsf(source.distance) <= parameters.site_band_m;
        const bool negative_band = parameters.include_negative_truncation_band &&
            source.distance <= -(parameters.truncation_distance - parameters.site_band_m);
        if (surface || negative_band) result.flags |= cuda_esdf_site;
    }
    else if (parameters.unobserved == cuda_esdf_unobserved::free)
        result.flags = cuda_esdf_observed;
    else if (parameters.unobserved == cuda_esdf_unobserved::occupied)
        result.flags = cuda_esdf_observed | cuda_esdf_inside | cuda_esdf_site;
    if (result.flags & cuda_esdf_site) {
         result.squared_distance_voxels = 0;
        result.flags |= cuda_esdf_parent_assigned;
    }
    output[index] = result;
}
}
