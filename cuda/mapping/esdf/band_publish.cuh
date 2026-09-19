#pragma once
#include "detail/index.cuh"
#include <covoxslam/voxel/esdf_encodings.hpp>
#include <covoxslam/voxel/flags.hpp>

namespace covoxslam::store
{
__global__ void clear_band_output(band_output_view output)
{
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < output.size() * tile_voxels)
        storage::scatter_record<storage::schema::esdf>(
            output[i / tile_voxels].voxels[i % tile_voxels], {});
}
__global__ void publish_band_fields(band_work_view input, band_output_view output,
    const band_tile_slot* slots, const detail::cuda_tile_coordinate* keys,
    unsigned count, float voxel_size, float maximum, unsigned* status)
{
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= std::size_t(count) * tile_voxels) return;
    const auto tile = i / tile_voxels;
    const unsigned local = unsigned(i % tile_voxels);
    const auto value = band_load(input, tile, local);
    const float extent = maximum / voxel_size;
    const bool clipped = value.squared_distance_voxels >= extent * extent;
    const bool assigned = (value.flags & detail::cuda_esdf_parent_assigned) != 0;
    const bool site = (value.flags & detail::cuda_esdf_site) != 0;
    storage::esdf_value result{};
    const float distance = clipped ? maximum : fminf(maximum, voxel_size * sqrtf(value.squared_distance_voxels));
    result.signed_distance_m = copysignf(distance, value.flags & detail::cuda_esdf_inside ? -1.f : 1.f);
    result.flags = value.flags & (observed_flag | esdf_site_flag);
    constexpr auto valid_flags = detail::cuda_esdf_observed | detail::cuda_esdf_inside |
        detail::cuda_esdf_site | detail::cuda_esdf_parent_assigned;
    bool valid = isfinite(value.squared_distance_voxels) && value.squared_distance_voxels >= 0 &&
        !(value.flags & ~valid_flags) &&
        ((value.flags & detail::cuda_esdf_observed) || !(value.flags & ~detail::cuda_esdf_observed)) &&
        (assigned || clipped) &&
        (!site || (assigned && (value.flags & detail::cuda_esdf_inside) && value.squared_distance_voxels == 0));

    if (assigned && !clipped) {
        int position[3]{};
        valid = valid && detail::site_coordinate(keys[tile], int(local), position);
        std::int32_t delta[3]{};

        for (unsigned axis = 0; axis < 3; ++axis)
        {
            const auto wide = std::int64_t(value.parent[axis]) - position[axis];
            if (wide < INT_MIN || wide > INT_MAX) valid = false;
            else delta[axis] = int(wide);
        }
        result.parent_dx = delta[0]; result.parent_dy = delta[1]; result.parent_dz = delta[2];
        valid = valid && (!site || !(delta[0] || delta[1] || delta[2]));
    }
    if (fabsf(result.signed_distance_m) >= maximum)
    {
        result.signed_distance_m = copysignf(maximum, result.signed_distance_m);
        result.parent_dx = result.parent_dy = result.parent_dz = 0;
    }
    if (!valid) { atomicOr(status, 2U); return; }
    storage::scatter_record<storage::schema::esdf>(output[slots[tile].target].voxels[local], result);
}
}
