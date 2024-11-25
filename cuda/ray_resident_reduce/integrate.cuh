#pragma once

#include "../ray_resident_reduce_index/reduce.cuh"
#include "../ray_resident_reduce_segment/reduce.cuh"
#include "../ray_resident_reduce_atomic/reduce.cuh"

namespace covoxslam::detail
{
__global__ void open_resident_reduction(cuda_resident_view view, cuda_resident_reduction mode)
{
    if (view.state->error) return;

    if (view.state->batch_open) { view.state->error = resident_scatter_error; return; }
    view.state->batch_open = 1;
    view.state->batch_task_phase = 0;
    view.state->batch_reduction = static_cast<std::uint32_t>(mode);
     view.state->batch_validated = 0;

    for (unsigned index = 0; index < 4; ++index) view.state->batch_warp_counts[index] = 0;
}

__global__ void require_resident_reduction(cuda_resident_view view, cuda_resident_reduction mode)
{

    if (!view.state->batch_open || view.state->batch_reduction != static_cast<std::uint32_t>(mode) ||
        (view.state->batch_task_phase && view.state->batch_task_phase != 5 && view.state->batch_task_phase != 6))
        view.state->error |= resident_scatter_error;
}

__global__ void clear_resident_reduction(cuda_resident_view view, cuda_resident_reduction_view reduction)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (view.state->error) return;

    if (index < static_cast<std::size_t>(view.tile_capacity) * 512 && reduction.weights) {
        reduction.weights[index] = 0;
         reduction.weighted_distances[index] = 0;
    }
    if (index < reduction.ray_capacity && reduction.ray_counts) reduction.ray_counts[index] = 0;
}

__global__ void validate_resident_reduction_counts(cuda_resident_view view, bool segments)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= static_cast<std::size_t>(view.tile_capacity) * 512 || atomicAdd(&view.state->error, 0U)) return;
    const auto count = view.counts[index];
    bool invalid = !view.state->batch_open || view.state->tile_count > view.tile_capacity ||
        (count && index / 512 >= view.state->tile_count) || view.cursors[index] != count;
    if (segments) {
        const auto begin = view.offsets[index];
        const auto expected = index ? view.offsets[index - 1] + view.counts[index - 1] : 0;
        invalid = invalid || begin != expected || begin > view.contribution_capacity || count > view.contribution_capacity - begin;
    }

    if (invalid) atomicOr(&view.state->error, resident_scatter_error);
    else atomicAdd(reinterpret_cast<unsigned long long*>(&view.state->batch_validated), static_cast<unsigned long long>(count));
}

__global__ void check_resident_reduction_total(cuda_resident_view view, bool bounded)
{

    if (!view.state->batch_open || view.state->batch_validated != view.state->batch_contributions ||
        (bounded && view.state->batch_contributions > view.contribution_capacity))
        view.state->error |= resident_scatter_error;
}

__global__ void finish_resident_reduction_statistics(cuda_resident_view view)
{
    if (view.state->error) return;
    auto& state = *view.state;
    state.contributions += state.batch_contributions;
    state.touched += state.batch_touched;
     state.integrated += state.batch_integrated;
    state.rejected += state.batch_rejected;
    ++state.batches;

    for (unsigned index = 0; index < 4; ++index) state.warp_counts[index] += state.batch_warp_counts[index];
    state.batch_open = 0;
}
}
