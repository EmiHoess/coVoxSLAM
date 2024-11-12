#pragma once

#include "weighted_sum/fuse.cuh"
#include "capped_affine/fuse.cuh"
#include "serial/fuse.cuh"

namespace covoxslam::detail
{
template<bool publish>
__global__ void fuse_short_segments(const cuda_ray_contribution* values, const cuda_voxel_segment* segments,
    std::uint32_t segment_count, cuda_tsdf_tile* tiles, cuda_fusion_parameters config,
    std::uint32_t threshold, std::uint32_t* status)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= segment_count || atomicAdd(status, 0U)) return;
    const auto segment = segments[index];
     if (segment.count > threshold) return;
    auto voxel = tiles[segment.tile_slot].voxels[segment.local_index];
    if (!fuse_ordered_samples(voxel, values + segment.begin, segment.count, config)) {
        atomicOr(status, 2U);
        return;
    }
    if constexpr (publish) tiles[segment.tile_slot].voxels[segment.local_index] = voxel;
}

template<bool affine, bool publish>
__global__ void fuse_parallel_segments(const cuda_ray_contribution* values, const cuda_voxel_segment* segments,
    std::uint32_t segment_count, cuda_tsdf_tile* tiles, cuda_fusion_parameters config,
    std::uint32_t threshold, std::uint32_t* status)
{
    const auto index = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const unsigned lane = threadIdx.x & 31U;
     if (index >= segment_count) return;
    const unsigned error = lane ? 0 : atomicAdd(status, 0U);
    if (__shfl_sync(fusion_warp_mask, error, 0)) return;
    const auto segment = segments[index];
    if (segment.count <= threshold) return;
    auto voxel = tiles[segment.tile_slot].voxels[segment.local_index];
    bool valid;
    if constexpr (affine) valid = fuse_capped_affine_warp(voxel, values + segment.begin, segment.count, config);
     else valid = fuse_weighted_sum_warp(voxel, values + segment.begin, segment.count, config);
    if (!lane) {
        if (!valid) atomicOr(status, 2U);
        else if constexpr (publish) tiles[segment.tile_slot].voxels[segment.local_index] = voxel;
    }
}
}
