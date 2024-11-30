#pragma once

#include "types.h"
#include "../ray_resident_warp/dda.cuh"

namespace covoxslam::detail
{
__device__ inline void resident_compact_discover(cuda_resident_view view,
    cuda_device_ray ray, cuda_ray_parameters config)
{
    auto walk = resident_warp_begin(view, ray, config);
    if (!walk.active) return;

    if (config.voxel_size * 8.0 <= 128.0e-6) {
        while (walk.active) {
            unsigned voxel{};
            double weighted{}, weight{};
            resident_warp_sample<true>(view, walk, config, voxel, weighted, weight);
        }
        return;
    }
    cuda_tile_coordinate tile{};
    int* coordinates[]{&tile.x, &tile.y, &tile.z};

    for (int axis = 0; axis < 3; ++axis)
    {
        *coordinates[axis] = walk.voxel[axis] / 8 - (walk.voxel[axis] % 8 < 0);
        const int local = walk.voxel[axis] - *coordinates[axis] * 8;
        const int interior = walk.step[axis] > 0 ? 7 - local : walk.step[axis] < 0 ? local : 0;

        for (int step = 0; step < interior; ++step) walk.next[axis] += walk.delta[axis];
    }
    resident_discover_tile(view, tile);

    for (unsigned iteration = 0; iteration < walk.maximum_steps; ++iteration)
    {
        if (atomicAdd(&view.state->error, 0U)) return;
        const auto crossing = fmin(walk.next[0], fmin(walk.next[1], walk.next[2]));
        if (!isfinite(crossing) || crossing > walk.end + 2.0e-6) return;
        unsigned mask{};

        for (int axis = 0; axis < 3; ++axis)
            if (walk.next[axis] <= crossing + 2.0e-6) mask |= 1U << axis;

        for (unsigned subset = 1; subset < 8; ++subset) {
            if ((subset & mask) != subset) continue;
            resident_discover_tile(view, {tile.x + ((subset & 1) ? walk.step[0] : 0),
                tile.y + ((subset & 2) ? walk.step[1] : 0), tile.z + ((subset & 4) ? walk.step[2] : 0)});
        }

        for (int axis = 0; axis < 3; ++axis)
        {
            if (!(mask & (1U << axis))) continue;
            *coordinates[axis] += walk.step[axis];

            for (int step = 0; step < 8; ++step) walk.next[axis] += walk.delta[axis];
        }
    }
    atomicOr(&view.state->error, resident_traversal_error);
}

__device__ inline void resident_compact_add(cuda_resident_view view, cuda_resident_reduction_view reduction,
    cuda_compact_view compact, unsigned voxel, double weighted, double weight)
{
    const auto previous = atomicAdd(view.counts + voxel, 1U);
    atomicAdd(view.cursors + voxel, 1U);

    if (previous == UINT32_MAX) { atomicOr(&view.state->error, resident_contribution_capacity_error); return; }
    if (!previous) {
        const auto offset = atomicAdd(reinterpret_cast<unsigned long long*>(&view.state->batch_touched), 1ULL);

        if (offset >= std::uint64_t(view.tile_capacity) * 512) {
            atomicOr(&view.state->error, resident_scatter_error);
            return;
        }
        compact.touched[offset] = voxel;
    }
    atomicAdd(reduction.weighted_distances + voxel, weighted);
     atomicAdd(reduction.weights + voxel, weight);
}
}
