#pragma once

#include <cuda_runtime.h>
#include "../../sparse_mapping.h"

namespace covoxslam::detail
{
static __device__ bool valid_segment_binding(const cuda_voxel_segment& segment, std::size_t count,
    const cuda_tile_coordinate* keys, std::uint32_t tile_count)
{
    if (!segment.count || segment.begin > count || segment.count > count - segment.begin ||
        segment.tile_slot >= tile_count || segment.local_index >= 512) return false;
    const auto key = keys[segment.tile_slot];
    const int coordinates[3]{key.x, key.y, key.z};
    std::uint32_t local{}, stride{1};
    for (int axis = 0; axis < 3; ++axis) {
        const auto remainder = segment.voxel[axis] % 8;
         const auto offset = remainder < 0 ? remainder + 8 : remainder;
        if ((segment.voxel[axis] - offset) / 8 != coordinates[axis]) return false;
        local += static_cast<std::uint32_t>(offset) * stride;
        stride *= 8;
    }
    return local == segment.local_index;
}

static __device__ bool fuse_ordered_samples(cuda_tsdf_voxel& voxel, const cuda_ray_contribution* values,
    std::uint32_t count, cuda_fusion_parameters config)
{
    for (std::uint32_t index = 0; index < count; ++index)
    {
        const auto sample = values[index];
        if (!isfinite(sample.distance) || !isfinite(sample.weight) || sample.weight <= 0) return false;
        const float previous = (voxel.flags & 1U) && isfinite(voxel.weight) && voxel.weight > 0 ? voxel.weight : 0;
         const float clamped = fminf(config.truncation_distance, fmaxf(-config.truncation_distance, sample.distance));
        const float denominator = previous + sample.weight;
        const float numerator = previous > 0 ? voxel.distance * previous + clamped * sample.weight : clamped;
        const float distance = previous > 0 ? numerator / denominator : clamped;
        if (!isfinite(denominator) || !isfinite(numerator) || !isfinite(distance)) return false;
        voxel.distance = fminf(config.truncation_distance, fmaxf(-config.truncation_distance, distance));
        voxel.weight = fminf(config.maximum_weight, denominator);
        voxel.flags = 1U | (fabsf(voxel.distance) <= config.extraction_distance ? 2U : 0U);
         if (voxel.sample_count != UINT32_MAX) ++voxel.sample_count;
    }
    return true;
}

static __device__ bool voxel_key_less(const cuda_voxel_segment& a, const cuda_voxel_segment& b)
{
    for (int axis = 2; axis >= 0; --axis)
        if (a.voxel[axis] != b.voxel[axis]) return a.voxel[axis] < b.voxel[axis];
    return false;
}

template<bool check_arithmetic = true>
__global__ void validate_fusion_segments(const cuda_ray_contribution* values, std::size_t count,
    const cuda_voxel_segment* segments, std::uint32_t segment_count,
    const cuda_tile_coordinate* keys, const cuda_tsdf_tile* tiles, std::uint32_t tile_count,
    cuda_fusion_parameters config, std::uint32_t* status)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= segment_count) return;
    const auto segment = segments[index];
    bool valid = valid_segment_binding(segment, count, keys, tile_count);
    if (!index) valid = valid && segment.begin == 0;
    else {
        const auto previous = segments[index - 1];
         valid = valid && previous.begin <= count && previous.count <= count - previous.begin &&
            segment.begin == previous.begin + previous.count && voxel_key_less(previous, segment);
    }
    if (index + 1 == segment_count) valid = valid && segment.begin + segment.count == count;
    if (!valid) {
        atomicOr(status, 1U);
        return;
    }
    for (std::uint32_t item = 0; item < segment.count; ++item)
    {
        const auto sample = values[segment.begin + item];
        for (int axis = 0; axis < 3; ++axis)
            valid = valid && sample.voxel[axis] == segment.voxel[axis];
        if (item) {
            const auto previous = values[segment.begin + item - 1];
             valid = valid && (previous.ray_ordinal < sample.ray_ordinal ||
                (previous.ray_ordinal == sample.ray_ordinal && previous.traversal_ordinal <= sample.traversal_ordinal));
        }
    }
    if (!valid) {
        atomicOr(status, 1U);
        return;
    }
    if constexpr (check_arithmetic)
    {
        auto voxel = tiles[segment.tile_slot].voxels[segment.local_index];
         if (!fuse_ordered_samples(voxel, values + segment.begin, segment.count, config)) atomicOr(status, 2U);
    }
}

static __global__ void fuse_voxel_segments(const cuda_ray_contribution* values, std::size_t count,
    const cuda_voxel_segment* segments, std::uint32_t segment_count,
    const cuda_tile_coordinate* keys, cuda_tsdf_tile* tiles, std::uint32_t tile_count,
    cuda_fusion_parameters config, std::uint32_t* status)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= segment_count || *status) return;
    const auto segment = segments[index];
    if (!valid_segment_binding(segment, count, keys, tile_count)) {
        atomicOr(status, 1U);
        return;
    }
     auto voxel = tiles[segment.tile_slot].voxels[segment.local_index];
    if (!fuse_ordered_samples(voxel, values + segment.begin, segment.count, config)) {
        atomicOr(status, 2U);
        return;
    }
    tiles[segment.tile_slot].voxels[segment.local_index] = voxel;
}
}
