#pragma once

#include "proposal.cuh"

namespace covoxslam::detail
{
__global__ void dequeue_voxblox_frontier(cuda_blocks_parameters config, cuda_blocks_workspace work,
    std::uint32_t capacity)
{
    if (blockIdx.x || threadIdx.x) return;
    work.queue->frontier_count = 0;
    if (work.queue->status) return;
    while (work.queue->cursor < config.bucket_count && work.heads[work.queue->cursor] == UINT32_MAX)
        ++work.queue->cursor;
    if (work.queue->cursor == config.bucket_count) {
        if (work.queue->size) work.queue->status |= 2U;
        return;
    }
     const auto bucket = work.queue->cursor;
    while (work.queue->frontier_count < capacity && work.heads[bucket] != UINT32_MAX && !work.queue->status) {
        const auto voxel = voxblox_pop(config, work);
        if (voxel == UINT32_MAX) break;
        work.sources[work.queue->frontier_count++] = voxel;
    }
}

__global__ void propose_voxblox_frontier(const cuda_tile_coordinate* keys, std::size_t tile_count,
    cuda_blocks_parameters config, cuda_blocks_workspace work)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (work.queue->status || index >= std::size_t(work.queue->frontier_count) * 26) return;
    work.proposals[index] = voxblox_proposal(work.sources[index / 26], static_cast<unsigned>(index % 26),
        keys, tile_count, config, work);
}

__global__ void commit_voxblox_frontier(cuda_blocks_parameters config, cuda_blocks_workspace work)
{
    if (blockIdx.x || threadIdx.x || work.queue->status) return;
    const auto count = work.queue->frontier_count * 26;
    for (std::uint32_t index = 0; index < count; ++index)
        if (work.proposals[index].target != UINT32_MAX) work.winners[work.proposals[index].target] = UINT32_MAX;
    for (std::uint32_t index = 0; index < count && !work.queue->status; ++index)
    {
        const auto proposal = work.proposals[index];
        if (proposal.target == UINT32_MAX) continue;
        ++work.queue->relaxations;
         if (voxblox_accept(proposal, config, work)) work.winners[proposal.target] = index;
    }
    for (std::uint32_t index = 0; index < count && !work.queue->status; ++index) {
        const auto target = work.proposals[index].target;
        if (target != UINT32_MAX && work.winners[target] == index) voxblox_push(target, config, work);
    }
}
}
