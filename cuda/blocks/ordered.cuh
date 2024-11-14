#pragma once

#include "proposal.cuh"

namespace covoxslam::detail
{
__global__ void propagate_voxblox_ordered(const cuda_tile_coordinate* keys, std::size_t tile_count,
    cuda_blocks_parameters config, cuda_blocks_workspace work, std::uint32_t maximum_pops)
{
    if (blockIdx.x || threadIdx.x || work.queue->status) return;
    for (std::uint32_t pop = 0; pop < maximum_pops && work.queue->size && !work.queue->status; ++pop)
    {
        const auto source = blocks_pop(config, work);
        if (source == UINT32_MAX) break;
        for (unsigned neighbor = 0; neighbor < 26 && !work.queue->status; ++neighbor) {
            const auto proposal = voxblox_proposal(source, neighbor, keys, tile_count, config, work);
            if (proposal.target == UINT32_MAX) continue;
             ++work.queue->relaxations;
            if (voxblox_accept(proposal, config, work))
                blocks_push(proposal.target, config, work);
        }
    }
}
}
