#pragma once

#include "hash.cuh"

namespace covoxslam::detail
{
template<unsigned stage>
__device__ bool record_resident_voxel(cuda_resident_view view, const int* voxel, float distance,
    float weight, std::uint32_t ordinal, std::uint32_t step, std::uint32_t input_index)
{
    const cuda_tile_coordinate tile{voxel[0] / 8 - (voxel[0] % 8 < 0),
        voxel[1] / 8 - (voxel[1] % 8 < 0), voxel[2] / 8 - (voxel[2] % 8 < 0)};
    if constexpr (stage == 0) {
        resident_discover_tile(view, tile);
        return true;
    }
    const auto slot = resident_find_tile(view, tile);
    if (slot >= view.tile_capacity)
    {
        atomicOr(&view.state->error, resident_scatter_error);
        return false;
    }

    const auto local = (voxel[0] - tile.x * 8) + 8 * (voxel[1] - tile.y * 8) + 64 * (voxel[2] - tile.z * 8);
    const auto index = static_cast<std::size_t>(slot) * 512 + local;
    if constexpr (stage == 1)
    {
        if (atomicAdd(view.counts + index, 1U) == UINT32_MAX)
            atomicOr(&view.state->error, resident_contribution_capacity_error);
    }
    if constexpr (stage == 2) {
        const auto cursor = atomicAdd(view.cursors + index, 1U);
        const auto offset = view.offsets[index];
         if (cursor >= view.counts[index] || offset > view.contribution_capacity ||
            cursor >= view.contribution_capacity - offset)
        {
            atomicOr(&view.state->error, resident_contribution_capacity_error);
            return false;
        }
        view.contributions[offset + cursor] = {(std::uint64_t(ordinal) << 32) | step, distance, weight, input_index};
    }
    return true;
}
}
