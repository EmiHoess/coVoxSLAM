#pragma once

#include "dda.cuh"
#include "types.h"

namespace covoxslam::detail
{
__global__ void begin_resident_warp(cuda_resident_view view)
{
    if (view.state->error) return;

    if (!view.state->batch_open || view.state->batch_task_phase || view.state->batch_contributions ||
        view.state->batch_reduction != static_cast<unsigned>(cuda_resident_reduction::direct_atomic)) {
        view.state->error = resident_scatter_error;
        return;
    }
    view.state->batch_task_phase = 6;
}

template<unsigned entries>
__global__ void walk_resident_warp(cuda_resident_view view, cuda_resident_reduction_view reduction,
    const cuda_device_ray* rays, unsigned count, cuda_ray_parameters config, bool scattered)
{
    __shared__ resident_warp_cache_entry caches[4][entries ? entries : 1];
    const unsigned lane = threadIdx.x & 31;
    const auto warp = blockIdx.x * 4 + threadIdx.x / 32;
    auto* cache = caches[threadIdx.x / 32];
    resident_warp_counters counters{};
    unsigned integrated{}, rejected{};
    unsigned long long contributions{};

    if constexpr (entries) {
        for (unsigned index = lane; index < entries; index += 32) cache[index].count = 0;
        __syncwarp(resident_warp_mask);
    }

    for (std::uint64_t packet = warp; packet < (std::uint64_t(count) + 31) / 32; packet += gridDim.x * 4)
    {
        unsigned error{};
        if (!lane) error = atomicAdd(&view.state->error, 0U);

        if (__shfl_sync(resident_warp_mask, error, 0)) return;
        const auto index = resident_warp_ray_index(count, packet, lane, scattered);
        resident_warp_dda walk{};

        if (index < count) walk = resident_warp_begin(view, rays[index], config);

        while (__any_sync(resident_warp_mask, walk.active)) {
            unsigned voxel{};
            double distance{}, weight{};
            const bool accepted = resident_warp_sample(view, walk, config, voxel, distance, weight);
            resident_warp_aggregate<entries>(view, reduction, cache, accepted, voxel, distance, weight, counters);
        }
        if (index < count) {
            integrated += walk.contributions != 0;
             rejected += walk.contributions == 0;
            contributions += walk.contributions;
        }
    }
    if constexpr (entries)
    {
        __syncwarp(resident_warp_mask);

        for (unsigned index = lane; index < entries; index += 32) {
            const auto entry = cache[index];

            if (entry.count) {
                resident_warp_flush(view, reduction, entry.voxel, entry.weighted_distance,
                    entry.weight, entry.count, counters);
                ++counters.cache_flushes;
            }
        }
    }

    for (unsigned delta = 16; delta; delta /= 2) {
        integrated += __shfl_down_sync(resident_warp_mask, integrated, delta);
        rejected += __shfl_down_sync(resident_warp_mask, rejected, delta);
        contributions += __shfl_down_sync(resident_warp_mask, contributions, delta);
        counters.flush_groups += __shfl_down_sync(resident_warp_mask, counters.flush_groups, delta);
        counters.peer_groups += __shfl_down_sync(resident_warp_mask, counters.peer_groups, delta);
         counters.cache_hits += __shfl_down_sync(resident_warp_mask, counters.cache_hits, delta);
        counters.cache_flushes += __shfl_down_sync(resident_warp_mask, counters.cache_flushes, delta);
    }
    if (!lane) {
        atomicAdd(&view.state->batch_integrated, integrated);
        atomicAdd(&view.state->batch_rejected, rejected);
        atomicAdd(reinterpret_cast<unsigned long long*>(&view.state->batch_contributions), contributions);
        atomicAdd(&view.state->batch_warp_counts[0], counters.flush_groups);
        atomicAdd(&view.state->batch_warp_counts[1], counters.peer_groups);
        atomicAdd(&view.state->batch_warp_counts[2], counters.cache_hits);
        atomicAdd(&view.state->batch_warp_counts[3], counters.cache_flushes);
    }
}
}
