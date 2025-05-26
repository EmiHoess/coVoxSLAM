#pragma once
#include "band_esdf_index.cuh"
#include "band_esdf_distance.cuh"

namespace covoxslam::store
{
__global__ void sweep_band_axis(const detail::cuda_tile_coordinate* keys, unsigned count,
    band_work_view output, int axis, float maximum_squared, unsigned* changed)
{
    const auto line = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (line >= std::size_t(count) * 64) return;
    const auto tile = line / 64;
     const int a = int(line % 8), b = int(line / 8 % 8);
    bool modified{};

    for (int pass = 0; pass < 2; ++pass) {

        for (int step = 1; step < 8; ++step)
        {
            const int current = pass ? 7 - step : step;
            const int previous = current + (pass ? 1 : -1);
            int cell[]{a, b, current}, source[]{a, b, previous};

            if (axis == 0) {
                cell[0] = current; cell[1] = a; cell[2] = b;
                source[0] = previous; source[1] = a; source[2] = b;
            }
            else if (axis == 1) {
                cell[1] = current; cell[2] = b;
                source[1] = previous; source[2] = b;
            }
            const auto local = unsigned(cell[0] + 8 * (cell[1] + 8 * cell[2]));
            const auto neighbor = unsigned(source[0] + 8 * (source[1] + 8 * source[2]));
            int position[3];

            if (!detail::site_coordinate(keys[tile], int(local), position)) return;
            auto value = band_load(output, tile, local);

            if (!band_mutable(value)) continue;

            if (relax_band_value(value, output, tile, neighbor, position, maximum_squared))
            {
                band_store(output, tile, local, value);
                 modified = true;
            }
        }
    }
    if (modified) atomicExch(changed, 1U);
}
}
