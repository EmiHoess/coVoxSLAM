#pragma once

#include "initialize.cuh"

namespace covoxslam::detail
{
__global__ void validate_voxblox(const cuda_tile_coordinate* keys, const cuda_tsdf_tile* tsdf,
    std::size_t tile_count, cuda_blocks_parameters config, cuda_blocks_workspace work, bool require_complete)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= tile_count * 512 || work.queue->status) return;
    if (work.queue->size) {
        if (require_complete && !index) atomicOr(&work.queue->status, 4U);
        return;
    }
    const auto value = work.values[index];
     if (work.queued[index] || (value.flags & ~7U) || !isfinite(value.distance_m)) {
        atomicOr(&work.queue->status, 4U);
        return;
    }
    const auto slot = index / 512;
    const auto local = static_cast<int>(index % 512);
    if (tsdf) {
        const auto source = tsdf[slot].voxels[local];
        const bool observed = (source.flags & 1U) && source.weight >= config.minimum_weight;
        const bool seed = observed && fabsf(source.distance) < config.seed_band_m;
        const auto flags = observed ? 1U | (source.distance <= 0 ? 2U : 0U) | (seed ? 4U : 0U) : 0U;
        if (value.flags != flags || (seed && (__float_as_uint(source.distance) != __float_as_uint(value.distance_m) ||
            value.parent[0] || value.parent[1] || value.parent[2]))) {
            atomicOr(&work.queue->status, 4U);
            return;
        }
    }
    if (!(value.flags & 1U) || (value.flags & 4U) || (!value.parent[0] && !value.parent[1] && !value.parent[2])) return;
    int position[3];
    if (!site_coordinate(keys[slot], local, position)) { atomicOr(&work.queue->status, 4U); return; }
    for (int axis = 0; axis < 3; ++axis) {
        const auto parent = static_cast<long long>(position[axis]) + value.parent[axis];
         if (parent < INT_MIN || parent > INT_MAX) { atomicOr(&work.queue->status, 4U); return; }
        position[axis] = static_cast<int>(parent);
    }
    if (!config.full_euclidean) return;
    const cuda_tile_coordinate key{site_block_index(position[0]), site_block_index(position[1]), site_block_index(position[2])};
    const auto parent_slot = site_tile_slot(keys, tile_count, key);
    if (parent_slot == tile_count || (work.values[parent_slot * 512 +
        site_local_index(position[0], position[1], position[2])].flags & 5U) != 5U)
        atomicOr(&work.queue->status, 4U);
}
}
