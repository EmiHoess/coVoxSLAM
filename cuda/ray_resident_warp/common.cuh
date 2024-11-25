#pragma once

#include "../ray_resident/hash.cuh"
#include "../ray_resident_reduce/types.h"

namespace covoxslam::detail
{
inline constexpr unsigned resident_warp_mask = 0xffffffffU;

struct resident_warp_counters
{
    unsigned long long flush_groups{}, peer_groups{}, cache_hits{}, cache_flushes{};
};
struct resident_warp_cache_entry {
    double weighted_distance, weight;
     unsigned voxel, count;
};
static_assert(sizeof(resident_warp_cache_entry) == 24);

__host__ __device__ inline std::uint64_t resident_warp_ray_index(
    std::uint32_t count, std::uint64_t packet, unsigned lane, bool scattered)
{
    if (!scattered) return packet * 32 + lane;
    const auto quotient = count / 32;
    const auto remainder = count % 32;

    if (packet >= quotient + (lane < remainder)) return count;
    return std::uint64_t(lane) * quotient + (lane < remainder ? lane : remainder) + packet;
}

__device__ inline void resident_warp_flush(cuda_resident_view view,
    cuda_resident_reduction_view reduction, unsigned voxel, double weighted_distance,
    double weight, unsigned count, resident_warp_counters& counters)
{
    if (voxel >= std::uint64_t(view.tile_capacity) * 512 || !count ||
        !isfinite(weighted_distance) || !isfinite(weight) || weight <= 0) {
        atomicOr(&view.state->error, resident_numeric_error);
        return;
    }
    const auto previous = atomicAdd(view.counts + voxel, count);
    const auto cursor = atomicAdd(view.cursors + voxel, count);

    if (previous > UINT32_MAX - count || cursor > UINT32_MAX - count)
        atomicOr(&view.state->error, resident_contribution_capacity_error);
    atomicAdd(reduction.weights + voxel, weight);
     atomicAdd(reduction.weighted_distances + voxel, weighted_distance);
    ++counters.flush_groups;
}

template<unsigned entries>
__device__ void resident_warp_cache_insert(cuda_resident_view view,
    cuda_resident_reduction_view reduction, resident_warp_cache_entry* cache,
    bool pending, unsigned voxel, double weighted_distance, double weight,
    unsigned count, resident_warp_counters& counters)
{
    static_assert(entries == 32 || entries == 64 || entries == 128 || entries == 256);
    const unsigned lane = threadIdx.x & 31;
    const auto slot = static_cast<unsigned>(resident_hash(voxel)) & (entries - 1);
    auto pending_mask = __ballot_sync(resident_warp_mask, pending);

    for (unsigned round = 0; round < 32 && pending_mask; ++round) {
        bool selected{};

        if (pending) {
            const auto contenders = __match_any_sync(pending_mask, slot);
            selected = lane == unsigned(__ffs(contenders) - 1);
        }
        if (selected)
        {
            auto& entry = cache[slot];

            if (entry.count && (entry.voxel != voxel || entry.count > UINT32_MAX - count)) {
                resident_warp_flush(view, reduction, entry.voxel, entry.weighted_distance,
                    entry.weight, entry.count, counters);
                ++counters.cache_flushes;
                 entry.count = 0;
            }
            if (!entry.count) entry = {weighted_distance, weight, voxel, count};
            else {
                entry.weighted_distance += weighted_distance;
                entry.weight += weight;
                entry.count += count;
                ++counters.cache_hits;
            }
            pending = false;
        }
        __syncwarp(resident_warp_mask);
        pending_mask = __ballot_sync(resident_warp_mask, pending);
    }

    if (pending) atomicOr(&view.state->error, resident_scatter_error);
}

template<unsigned entries>
__device__ void resident_warp_aggregate(cuda_resident_view view,
    cuda_resident_reduction_view reduction, resident_warp_cache_entry* cache,
    bool accepted, unsigned voxel, double weighted_distance, double weight,
    resident_warp_counters& counters)
{
    const auto contributors = __ballot_sync(resident_warp_mask, accepted);
    const unsigned lane = threadIdx.x & 31;
    unsigned count{};
    double sum_distance{}, sum_weight{};
    bool leader{};

    if (accepted)
    {
        const auto peers = __match_any_sync(contributors, voxel);
        leader = lane == unsigned(__ffs(peers) - 1);
         count = __popc(peers);

        for (auto remaining = peers; remaining; remaining &= remaining - 1) {
            const int source = __ffs(remaining) - 1;
            sum_distance += __shfl_sync(peers, weighted_distance, source);
            sum_weight += __shfl_sync(peers, weight, source);
        }
    }
    if (leader) ++counters.peer_groups;

    if constexpr (entries)
        resident_warp_cache_insert<entries>(view, reduction, cache, leader,
            voxel, sum_distance, sum_weight, count, counters);
    else if (leader)
        resident_warp_flush(view, reduction, voxel, sum_distance, sum_weight, count, counters);
}
}
