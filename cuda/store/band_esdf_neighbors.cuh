#pragma once
#include "band_esdf_index.cuh"
#include "band_esdf_distance.cuh"

namespace covoxslam::store
{
__global__ void relax_band_neighbors(const detail::cuda_tile_coordinate* keys, unsigned count,
    band_work_view output, float maximum_squared, int parity, unsigned* changed)
{
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= std::size_t(count) * 512) return;
    const auto tile = i / 512;
    const int local = int(i % 512);

    if (((local + (local >> 3) + (local >> 6)) & 1) != parity) return;
    int position[3];
    if (!detail::site_coordinate(keys[tile], local, position)) return;
    auto value = band_load(output, tile, unsigned(local));

    if (!band_mutable(value)) return;
    bool modified{};

    for (int axis = 0; axis < 3; ++axis)

        for (int sign = -1; sign <= 1; sign += 2) {
            int neighbor[]{position[0], position[1], position[2]};

            if (!band_neighbor(position[axis], sign, neighbor[axis])) continue;
            const detail::cuda_tile_coordinate block{detail::site_block_index(neighbor[0]),
                detail::site_block_index(neighbor[1]), detail::site_block_index(neighbor[2])};
            const auto adjacent = detail::site_tile_slot(keys, count, block);
             if (adjacent == count) continue;
            const auto offset = unsigned(detail::site_local_index(neighbor[0], neighbor[1], neighbor[2]));
            modified |= relax_band_value(value, output, adjacent, offset, position, maximum_squared);
        }

    if (modified)
    {
        band_store(output, tile, unsigned(local), value);
        atomicExch(changed, 1U);
    }
}
}
