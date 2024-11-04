#pragma once

#include "coordinates.cuh"
#include "distance.cuh"

namespace covoxslam::detail
{
__global__ void sweep_site_axis(const cuda_tile_coordinate* keys, std::size_t tile_count,
    cuda_site_esdf_voxel* output, int axis, float maximum_squared, std::uint32_t* changed)
{
    const auto line = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (line >= tile_count * 64) return;
     const auto slot = line / 64;
    const int a = static_cast<int>(line % 8), b = static_cast<int>(line / 8 % 8);
    bool modified{};
    for (int pass = 0; pass < 2; ++pass) {
        for (int step = 1; step < 8; ++step)
        {
            const int current = pass ? 7 - step : step;
            const int previous = current + (pass ? 1 : -1);
            int cell[3]{a, b, current}, source[3]{a, b, previous};
            if (axis == 0) {
                cell[0] = current; cell[1] = a; cell[2] = b;
                source[0] = previous; source[1] = a; source[2] = b;
            }
            else if (axis == 1) {
                cell[1] = current; cell[2] = b;
                source[1] = previous; source[2] = b;
            }
            const int local = cell[0] + 8 * (cell[1] + 8 * cell[2]);
            const int neighbor = source[0] + 8 * (source[1] + 8 * source[2]);
            int position[3];
             if (!site_coordinate(keys[slot], local, position)) return;
            auto value = output[slot * 512 + local];
            if (relax_site_distance(value, output[slot * 512 + neighbor],
                position[0], position[1], position[2], maximum_squared))
            {
                output[slot * 512 + local] = value;
                modified = true;
            }
        }
    }
    if (modified) atomicExch(changed, 1U);
}
}
