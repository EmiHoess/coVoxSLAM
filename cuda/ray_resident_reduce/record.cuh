#pragma once

#include "common.cuh"
#include "../ray_resident/hash.cuh"

namespace covoxslam::detail
{
template<unsigned stage>
__device__ bool record_resident_reduction(cuda_resident_view view, cuda_resident_reduction_view reduction,
    const int* voxel, float distance, float weight, std::uint32_t ray, std::uint32_t step, float truncation)
{
    const cuda_tile_coordinate tile{voxel[0] / 8 - (voxel[0] % 8 < 0),
        voxel[1] / 8 - (voxel[1] % 8 < 0), voxel[2] / 8 - (voxel[2] % 8 < 0)};
    const auto slot = resident_find_tile(view, tile);

    if (slot >= view.tile_capacity) { atomicOr(&view.state->error, resident_scatter_error); return false; }
    const auto local = (voxel[0] - tile.x * 8) + 8 * (voxel[1] - tile.y * 8) + 64 * (voxel[2] - tile.z * 8);
    const auto index = static_cast<std::uint32_t>(slot * 512 + local);
    if constexpr (stage == 4 || stage == 7)
    {
        if (atomicAdd(view.counts + index, 1U) == UINT32_MAX) {
            atomicOr(&view.state->error, resident_contribution_capacity_error);
            return false;
        }
    }

    if constexpr (stage == 5)
    {
        const auto offset = reduction.ray_offsets[ray];

        if (step >= reduction.ray_counts[ray] || offset > view.contribution_capacity || step >= view.contribution_capacity - offset) {
            atomicOr(&view.state->error, resident_contribution_capacity_error);
            return false;
        }
        reduction.samples[offset + step] = {distance, weight};
         reduction.voxel_indices[offset + step] = index;
    }
    if constexpr (stage == 6) {
        const auto cursor = atomicAdd(view.cursors + index, 1U);
        const auto offset = view.offsets[index];

        if (cursor >= view.counts[index] || offset > view.contribution_capacity || cursor >= view.contribution_capacity - offset) {
            atomicOr(&view.state->error, resident_contribution_capacity_error);
            return false;
        }
        reduction.samples[offset + cursor] = {distance, weight};
    }
    if constexpr (stage == 7)
    {
        atomicAdd(view.cursors + index, 1U);

        if (!add_resident_reduction(reduction, index, {distance, weight}, {truncation, 1.f, 0.f})) {
            atomicOr(&view.state->error, resident_numeric_error);
            return false;
        }
    }
    return true;
}
}
