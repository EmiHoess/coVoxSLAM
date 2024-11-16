#pragma once

#include "../ray_resident/fusion.cuh"

namespace covoxslam::detail
{
__global__ void prepare_resident_arrival(cuda_resident_view view, cuda_fusion_parameters config)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= static_cast<std::size_t>(view.tile_capacity) * 512 || view.state->error) return;
    const auto count = view.counts[index];
    if (!count) return;
     const auto* values = view.contributions + view.offsets[index];
    auto voxel = view.tiles[index / 512].voxels[index % 512];

    if (!fuse_resident_samples(voxel, values, count, config))
    {
        atomicOr(&view.state->error, resident_numeric_error);
        return;
    }
    view.pending[index] = voxel;
}
}
