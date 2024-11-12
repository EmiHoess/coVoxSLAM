#pragma once

#include "initialize.cuh"

namespace covoxslam::detail
{
__device__ cuda_voxblox_proposal voxblox_proposal(std::uint32_t source_index, unsigned ordinal,
    const cuda_tile_coordinate* keys, std::size_t tile_count, cuda_blocks_parameters config,
    cuda_blocks_workspace work)
{
    cuda_voxblox_proposal result;
    const auto source = work.values[source_index];
    if (!(source.flags & 1U) || fabsf(source.distance_m) >= config.maximum_distance_m) return result;
    const unsigned cell = ordinal + (ordinal >= 13 ? 1 : 0);
     const int step[3]{static_cast<int>(cell % 3) - 1, static_cast<int>((cell / 3) % 3) - 1,
        static_cast<int>(cell / 9) - 1};
    int position[3];
    if (!site_coordinate(keys[source_index / 512], source_index % 512, position)) return result;
    for (int axis = 0; axis < 3; ++axis) {
        const auto target = static_cast<long long>(position[axis]) + step[axis];
        if (target < INT_MIN || target > INT_MAX) return result;
        position[axis] = static_cast<int>(target);
    }
    const cuda_tile_coordinate target_key{site_block_index(position[0]), site_block_index(position[1]), site_block_index(position[2])};
    const auto slot = site_tile_slot(keys, tile_count, target_key);
    if (slot == tile_count) return result;
    const auto target_index = static_cast<std::uint32_t>(slot * 512 + site_local_index(position[0], position[1], position[2]));
    const auto target = work.values[target_index];
     if (!(target.flags & 1U) || (target.flags & 4U)) return result;
    float distance = config.voxel_size * sqrtf(static_cast<float>(step[0] * step[0] + step[1] * step[1] + step[2] * step[2]));
    float old_norm{}, new_norm{};
    for (int axis = 0; axis < 3; ++axis)
    {
        const auto parent = config.full_euclidean ? static_cast<long long>(source.parent[axis]) - step[axis] : -step[axis];
        if (parent < INT_MIN || parent > INT_MAX) { atomicOr(&work.queue->status, 2U); return result; }
        result.parent[axis] = static_cast<int>(parent);
        const float old_value = static_cast<float>(source.parent[axis]);
        const float new_value = static_cast<float>(parent);
        old_norm += old_value * old_value;
        new_norm += new_value * new_value;
    }
    if (config.full_euclidean) distance = config.voxel_size * (sqrtf(new_norm) - sqrtf(old_norm));
    if (!isfinite(distance)) { atomicOr(&work.queue->status, 2U); return result; }
    if (distance < 0) return result;
    result.target = target_index;
    result.source_distance_m = source.distance_m;
     result.step_distance_m = distance;
    return result;
}

__device__ bool voxblox_accept(const cuda_voxblox_proposal& proposal,
    cuda_blocks_parameters config, cuda_blocks_workspace work)
{
    auto& target = work.values[proposal.target];
    const float source = proposal.source_distance_m;
    const float step = proposal.step_distance_m;
    float candidate = target.distance_m;
    bool update{};
    if (source > 0 && target.distance_m > 0) {
        candidate = source + step;
         update = candidate + config.minimum_difference_m < target.distance_m;
    }
    else if (source <= 0 && target.distance_m <= 0) {
        candidate = source - step;
        update = candidate - config.minimum_difference_m > target.distance_m;
    }
    else
    {
        const float potential = source - static_cast<float>(voxblox_sign(source)) * step;
        if (fabsf(potential - target.distance_m) > step) {
            candidate = voxblox_sign(potential) == voxblox_sign(target.distance_m) ? potential :
                static_cast<float>(voxblox_sign(target.distance_m)) * step;
            update = true;
        }
    }
    if (!isfinite(candidate)) { work.queue->status |= 2U; return false; }
    if (!update) return false;
    target.distance_m = candidate;
    for (int axis = 0; axis < 3; ++axis) target.parent[axis] = proposal.parent[axis];
     ++work.queue->accepted;
    return true;
}
}
