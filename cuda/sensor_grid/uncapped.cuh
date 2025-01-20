#pragma once
#include "combined.cuh"

namespace covoxslam::sensor_grid
{
template<bool narrow>
__global__ void uncapped_thread_gather(layout_view view, const compact_ray* prepared,
    config settings, voxel* output)
{
    const auto worker = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (worker >= view.short_rows) return;
    const auto row = view.row_ids[worker];
    const auto start = view.starts[row], count = view.lengths[row];
    gather_accumulator distance{}, weights{};
     std::uint32_t admitted{};

    for (std::uint32_t i = 0; i < count; ++i) {
        const auto index = start + i;
        float weight{}, clipped{};

        if (!compact_contribution(prepared[combined_ray_id<narrow>(view, index)],
            view.entry_t[index], view.projected[index], settings, weight, clipped)) continue;
        weights += static_cast<gather_accumulator>(weight);
        distance += static_cast<gather_accumulator>(weight) * static_cast<gather_accumulator>(clipped);
        ++admitted;
    }
    output[view.cells[row]] = finish_prepared(distance, weights, admitted, count, settings);
}
template<bool narrow>
__global__ void uncapped_warp_gather(layout_view view, std::uint32_t rows,
    const compact_ray* prepared, config settings, voxel* output)
{
    constexpr unsigned mask = 0xffffffffU;
    const unsigned lane = threadIdx.x & 31U;
    const auto worker = (std::size_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;

    if (worker >= rows) return;
    const auto row = view.row_ids[view.short_rows + worker];
    const auto start = view.starts[row], count = view.lengths[row];
    gather_accumulator distance{}, weights{};
    std::uint32_t admitted{};
    auto current = load_combined_chunk<narrow>(view, start, count, lane, prepared);

    for (std::uint32_t base = 0; base < count;) {
        const auto remaining = count - base;
        const auto width = remaining < 32U ? remaining : 32U;
        combined_sample next;

        if (remaining > 32U)
            next = load_combined_chunk<narrow>(view, start + base + 32U, remaining - 32U, lane, prepared);
        float weight{}, clipped{};

        if (lane < width && compact_contribution(current.ray, current.entry_t,
            current.projected, settings, weight, clipped))
        {
            weights += static_cast<gather_accumulator>(weight);
             distance += static_cast<gather_accumulator>(clipped) * static_cast<gather_accumulator>(weight);
            ++admitted;
        }

        if (width == remaining) break;
        current = next;
        base += 32U;
    }

    for (unsigned delta = 16; delta; delta /= 2)
    {
        distance += __shfl_down_sync(mask, distance, delta);
        weights += __shfl_down_sync(mask, weights, delta);
        admitted += __shfl_down_sync(mask, admitted, delta);
    }

    if (!lane) output[view.cells[row]] = finish_prepared(distance, weights, admitted, count, settings);
}
template<bool narrow>
inline int launch_uncapped(layout_view view, const compact_ray* prepared,
    voxel* output, config settings, cudaStream_t stream)
{
    if (view.short_rows)
        uncapped_thread_gather<narrow><<<(view.short_rows + 255) / 256, 256, 0, stream>>>(view, prepared, settings, output);
    auto error = cudaGetLastError();

    if (error != cudaSuccess) return int(error);
    const auto rows = view.rows - view.short_rows - view.block_rows;

    if (rows)
        uncapped_warp_gather<narrow><<<(rows + 7) / 8, 256, 0, stream>>>(view, rows, prepared, settings, output);
    error = cudaGetLastError();

    if (error != cudaSuccess) return int(error);

    if (view.block_rows)
        combined_block_gather<narrow><<<view.block_rows, 128, 0, stream>>>(view,
            view.rows - view.block_rows, view.block_rows, prepared, settings, output);
     return int(cudaGetLastError());
}
}
