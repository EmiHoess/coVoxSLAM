#pragma once

#include "types.h"
#include <cuda_runtime.h>

namespace covoxslam::detail
{
__device__ inline bool resident_order_less(cuda_resident_contribution a, cuda_resident_contribution b)
{
    return a.order < b.order || (a.order == b.order && a.input_index < b.input_index);
}

__device__ inline bool fuse_resident_samples(cuda_tsdf_voxel& voxel, const cuda_resident_contribution* values,
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

__global__ void validate_resident_segments(cuda_resident_view view)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto capacity = static_cast<std::size_t>(view.tile_capacity) * 512;

    if (index >= capacity || atomicAdd(&view.state->error, 0U)) return;
    const auto count = view.counts[index];
    const auto begin = view.offsets[index];
    const auto expected = index ? view.offsets[index - 1] + view.counts[index - 1] : 0;
    const bool invalid = view.state->tile_count > view.tile_capacity ||
        (count && index / 512 >= view.state->tile_count) || view.cursors[index] != count ||
        begin != expected || begin > view.contribution_capacity || count > view.contribution_capacity - begin;

    if (invalid) {
        atomicOr(&view.state->error, resident_scatter_error);
    }
}

__global__ void prepare_resident_fusion(cuda_resident_view view, cuda_fusion_parameters config)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= static_cast<std::size_t>(view.tile_capacity) * 512 || view.state->error) return;
    const auto count = view.counts[index];
    if (!count) return;
    auto* values = view.contributions + view.offsets[index];

    for (std::uint32_t item = 1; item < count; ++item)
    {
        const auto sample = values[item];
         auto cursor = item;
        while (cursor && resident_order_less(sample, values[cursor - 1])) {
            values[cursor] = values[cursor - 1];
            --cursor;
        }
        values[cursor] = sample;
    }
    auto voxel = view.tiles[index / 512].voxels[index % 512];

    if (!fuse_resident_samples(voxel, values, count, config))
    {
        atomicOr(&view.state->error, resident_numeric_error);
        return;
    }
    view.pending[index] = voxel;
}

__global__ void publish_resident_fusion(cuda_resident_view view)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= static_cast<std::size_t>(view.tile_capacity) * 512 || view.state->error) return;
    const auto count = view.counts[index];

    if (count) {
        view.tiles[index / 512].voxels[index % 512] = view.pending[index];
        atomicAdd(reinterpret_cast<unsigned long long*>(&view.state->batch_touched), 1ULL);
         atomicMax(&view.state->maximum_segment_length, count);
    }
}

__global__ void finish_resident_statistics(cuda_resident_view view)
{

    if (view.state->error) return;
    auto& state = *view.state;
    const auto last = static_cast<std::size_t>(view.tile_capacity) * 512 - 1;
    state.batch_contributions = view.offsets[last] + view.counts[last];
    state.contributions += state.batch_contributions;
    state.touched += state.batch_touched;
     state.integrated += state.batch_integrated;
    state.rejected += state.batch_rejected;
    ++state.batches;
}
}
