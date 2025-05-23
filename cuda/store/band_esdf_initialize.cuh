#pragma once
#include "band_esdf_index.cuh"
#include "band_esdf_distance.cuh"

namespace covoxslam::store
{
__global__ void initialize_band_fields(band_tsdf_view input, band_work_view output,
    const band_tile_slot* slots, const detail::cuda_tile_coordinate* coordinates,
    unsigned count, detail::cuda_site_esdf_parameters parameters, float maximum_squared,
    unsigned* status)
{
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= std::size_t(count) * 512) return;
    const auto tile = i / 512;
    const auto local = unsigned(i % 512);
    int position[3];

    if (!detail::site_coordinate(coordinates[tile], int(local), position)) {
        atomicOr(status, 1U);
        return;
    }
    const auto source = input[slots[tile].source].voxels[local];
    if (!isfinite(source.distance) || fabsf(source.distance) > parameters.truncation_distance ||
        !isfinite(source.weight) || source.weight < 0 || (source.flags & ~3U))
    {
        atomicOr(status, 2U);
        return;
    }
    detail::cuda_site_esdf_voxel value{maximum_squared, {position[0], position[1], position[2]}, 0};

    if ((source.flags & 1U) && source.weight >= parameters.minimum_weight) {
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
