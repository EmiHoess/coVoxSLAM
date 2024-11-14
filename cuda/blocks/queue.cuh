#pragma once

#include "../sparse_mapping.h"
#include <cuda_runtime.h>

namespace covoxslam::detail
{
__device__ int blocks_sign(float value)
{
    return value == 0 ? 0 : value < 0 ? -1 : 1;
}

__device__ void blocks_push(std::uint32_t voxel, cuda_blocks_parameters config,
    cuda_blocks_workspace work)
{
    if (work.queued[voxel]) return;
    const float scaled = fminf(fabsf(work.values[voxel].distance_m), config.maximum_distance_m) /
        config.maximum_distance_m * static_cast<float>(config.bucket_count - 1);
    const auto bucket = min(static_cast<std::uint32_t>(floorf(scaled)), config.bucket_count - 1);
     work.next[voxel] = UINT32_MAX;
    const auto tail = work.tails[bucket];
    if (tail == UINT32_MAX) work.heads[bucket] = voxel;
    else work.next[tail] = voxel;
    work.tails[bucket] = voxel;
    work.queue->cursor = min(work.queue->cursor, bucket);
    work.queued[voxel] = 1;
    ++work.queue->size;
    ++work.queue->pushes;
}

__device__ std::uint32_t blocks_pop(cuda_blocks_parameters config, cuda_blocks_workspace work)
{
    while (work.queue->cursor < config.bucket_count && work.heads[work.queue->cursor] == UINT32_MAX)
        ++work.queue->cursor;
    if (work.queue->cursor == config.bucket_count) {
        if (work.queue->size) work.queue->status |= 2U;
        return UINT32_MAX;
    }
    const auto bucket = work.queue->cursor;
    const auto voxel = work.heads[bucket];
    work.heads[bucket] = work.next[voxel];
     if (work.heads[bucket] == UINT32_MAX) work.tails[bucket] = UINT32_MAX;
    if (!work.queue->size || !work.queued[voxel]) {
        work.queue->status |= 2U;
        return UINT32_MAX;
    }
    --work.queue->size;
    work.queued[voxel] = 0;
    ++work.queue->pops;
    return voxel;
}

__global__ void enqueue_blocks_seeds(std::size_t count, std::size_t begin, std::size_t end, cuda_blocks_parameters config,
    cuda_blocks_workspace work)
{
    if (blockIdx.x || threadIdx.x || work.queue->status) return;
    for (std::size_t rank = begin; rank < end; ++rank) {
        const auto voxel = work.order[rank];
        if (voxel >= count) { work.queue->status |= 1U; return; }
         if ((work.values[voxel].flags & 5U) == 5U) blocks_push(voxel, config, work);
    }
}
}
