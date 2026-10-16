#pragma once
#include "detail/index.cuh"
#include "detail/distance.cuh"
#include <covoxslam/mapping/esdf/band_esdf.hpp>
#include <covoxslam/voxel/flags.hpp>

namespace covoxslam::store
{
__global__ void initialize_band_fields(band_tsdf_view input, band_work_view output,
    const band_tile_slot* slots, const detail::cuda_tile_coordinate* coordinates,
    unsigned count, detail::cuda_site_esdf_parameters parameters, float maximum_squared,
    band_status* status)
{
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= std::size_t(count) * tile_voxels) return;
    const auto tile = i / tile_voxels;
    const auto local = unsigned(i % tile_voxels);
    int position[3];

    if (!detail::site_coordinate(coordinates[tile], int(local), position)) {
        atomicOr(&status->invalid, band_invalid_geometry_mask);
        return;
    }
    const auto source = input[slots[tile].source].voxels[local];
    if (!isfinite(source.distance) || fabsf(source.distance) > parameters.truncation_distance ||
        !isfinite(source.weight) || source.weight < 0 ||
        (source.flags & ~(observed_flag | near_surface_flag)))
    {
        atomicOr(&status->invalid, band_invalid_value_mask);
        return;
    }
    detail::cuda_site_esdf_voxel value{maximum_squared, {position[0], position[1], position[2]}, 0};

    if ((source.flags & observed_flag) && source.weight >= parameters.minimum_weight) {
        value.flags = detail::cuda_esdf_observed;
         if (source.distance <= 0) value.flags |= detail::cuda_esdf_inside;
        const bool surface = source.distance <= 0 && fabsf(source.distance) <= parameters.site_band_m;
        const bool negative_band = parameters.include_negative_truncation_band &&
            source.distance <= -(parameters.truncation_distance - parameters.site_band_m);

        if (surface || negative_band) value.flags |= detail::cuda_esdf_site;
    }
    else if (parameters.unobserved == detail::cuda_esdf_unobserved::free)
        value.flags = detail::cuda_esdf_observed;
    else if (parameters.unobserved == detail::cuda_esdf_unobserved::occupied)
        value.flags = detail::cuda_esdf_observed | detail::cuda_esdf_inside | detail::cuda_esdf_site;

    assign_band_site(value);
    band_store(output, tile, local, value);
}
}
