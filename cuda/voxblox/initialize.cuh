#pragma once

#include "../esdf_site/coordinates.cuh"
#include "queue.cuh"

namespace covoxslam::detail
{
__global__ void initialize_voxblox(const cuda_tile_coordinate* keys, const cuda_tsdf_tile* tsdf,
    std::size_t tile_count, cuda_blocks_parameters config, cuda_blocks_workspace work)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= tile_count * 512) return;
    const auto slot = index / 512;
    const auto local = static_cast<int>(index % 512);
     int position[3];
    if ((slot && !site_tile_less(keys[slot - 1], keys[slot])) || !site_coordinate(keys[slot], local, position)) {
        atomicOr(&work.queue->status, 1U);
        return;
    }
    const auto source = tsdf[slot].voxels[local];
    if ((source.flags & 1U) && (!isfinite(source.distance) || !isfinite(source.weight) || source.weight < 0)) {
        atomicOr(&work.queue->status, 1U);
        return;
    }
    cuda_voxblox_voxel value;
    if ((source.flags & 1U) && source.weight >= config.minimum_weight)
    {
        value.flags = 1U | (source.distance <= 0 ? 2U : 0U);
        value.distance_m = static_cast<float>(voxblox_sign(source.distance)) * config.maximum_distance_m;
        if (fabsf(source.distance) < config.seed_band_m) {
            value.distance_m = source.distance;
            value.flags |= 4U;
             atomicAdd(reinterpret_cast<unsigned long long*>(&work.queue->seeds), 1ULL);
        }
        atomicAdd(reinterpret_cast<unsigned long long*>(&work.queue->eligible), 1ULL);
    }
    work.values[index] = value;
}
}
