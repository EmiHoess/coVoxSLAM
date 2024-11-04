#pragma once

#include "coordinates.cuh"
#include "distance.cuh"

namespace covoxslam::detail
{
__global__ void relax_site_neighbors(const cuda_tile_coordinate* keys, std::size_t tile_count,
    cuda_site_esdf_voxel* output, float maximum_squared, int parity, std::uint32_t* changed)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= tile_count * 512) return;
    const auto slot = index / 512;
    const int local = static_cast<int>(index % 512);
     if (((local + (local >> 3) + (local >> 6)) & 1) != parity) return;
    int position[3];
    if (!site_coordinate(keys[slot], local, position)) return;
    auto value = output[index];
    bool modified{};
    for (int axis = 0; axis < 3; ++axis)
    {
        for (int sign = -1; sign <= 1; sign += 2) {
            int neighbor[3]{position[0], position[1], position[2]};
            const auto component = static_cast<long long>(neighbor[axis]) + sign;
            if (component < INT_MIN || component > INT_MAX) continue;
            neighbor[axis] = static_cast<int>(component);
            const cuda_tile_coordinate block{site_block_index(neighbor[0]),
                site_block_index(neighbor[1]), site_block_index(neighbor[2])};
             const auto adjacent = site_tile_slot(keys, tile_count, block);
            if (adjacent == tile_count) continue;
            const int offset = site_local_index(neighbor[0], neighbor[1], neighbor[2]);
            modified |= relax_site_distance(value, output[adjacent * 512 + offset],
                position[0], position[1], position[2], maximum_squared);
        }
    }
    if (modified) {
        output[index] = value;
        atomicExch(changed, 1U);
    }
}
}
