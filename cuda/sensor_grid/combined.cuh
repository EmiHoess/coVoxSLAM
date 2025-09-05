#pragma once
#include <covoxslam/sensor_grid_combined.h>

namespace covoxslam::sensor_grid
{
template <bool narrow>
__device__ inline combined_sample load_combined_sample(layout_view view, std::uint32_t index,
    const compact_ray* prepared)
{
    return {prepared[compact_ray_id<narrow>(view, index)], view.entry_t[index], view.projected[index]};
}
template <bool narrow>
__device__ inline combined_sample load_combined_chunk(layout_view view, std::uint32_t start,
    std::uint32_t remaining, unsigned lane, const compact_ray* prepared)
{
    if (lane >= remaining) return {};
    return load_combined_sample<narrow>(view, start + lane, prepared);
}
template <bool narrow>
__global__ void combined_pipelined_gather(layout_view view, std::uint32_t offset, std::uint32_t rows,
    const compact_ray* prepared, config settings, std::uint32_t max_rays, voxel* output) {
    constexpr unsigned mask = 0xffffffffU;
    const unsigned lane = threadIdx.x & 31U;
    const std::size_t worker = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;

    if (worker >= rows) return;
    const std::size_t row = view.row_ids[offset + worker];
    const auto start = view.starts[row], count = view.lengths[row];
    gather_accumulator sum_distance = 0, sum_weight = 0;
    std::uint32_t admitted = 0, tested = 0;
    auto current = load_combined_chunk<narrow>(view, start, count, lane, prepared);

    for (std::uint32_t base = 0; base < count;) {
        const unsigned remaining = count - base;
        const unsigned width = remaining < 32U ? remaining : 32U;
        combined_sample next;


        if (remaining > 32U)
            next = load_combined_chunk<narrow>(view, start + base + 32U, remaining - 32U, lane, prepared);

        float weight = 0, clipped = 0;
        const bool valid = lane < width && compact_contribution(current.ray,
            current.entry_t, current.projected, settings, weight, clipped);
        const auto valid_mask = __ballot_sync(mask, valid);
        const unsigned prefix = lane == 0 ? 0U : (mask >> (32U - lane));
        const unsigned rank = __popc(valid_mask & prefix);
        const unsigned available = __popc(valid_mask);
        const bool limited = max_rays != 0 && available >= max_rays - admitted;
        const unsigned take = limited ? max_rays - admitted : available;
        const bool selected = valid && rank < take;
        const auto selected_mask = __ballot_sync(mask, selected);

        if (selected) {

            sum_weight += static_cast<gather_accumulator>(weight);
            sum_distance += static_cast<gather_accumulator>(clipped) * static_cast<gather_accumulator>(weight);
        }
        admitted += take;
        tested = base + (limited ? 32U - __clz(selected_mask) : width);

        if (limited) break;

        if (width == remaining) break;
        current = next;
        base += 32U;
    }

    for (unsigned delta = 16; delta != 0; delta /= 2) {
        sum_distance += __shfl_down_sync(mask, sum_distance, delta);
        sum_weight += __shfl_down_sync(mask, sum_weight, delta);
    }

    if (lane == 0)
        output[view.cells[row]] = finish_prepared(sum_distance, sum_weight, admitted, tested, settings);
}

template <bool narrow, unsigned block_threads = 128>
__global__ __launch_bounds__(block_threads)
void combined_block_gather(layout_view view, std::uint32_t plan_offset, std::uint32_t rows,
    const compact_ray* prepared, config settings, voxel* output) {
    constexpr unsigned mask = 0xffffffffU;
    constexpr unsigned warp_count = block_threads / 32U;
    const std::size_t worker = blockIdx.x;

    if (worker >= rows) return;
    const auto row = view.row_ids[plan_offset + worker];
    const auto start = view.starts[row];
    const auto count = view.lengths[row];
    const unsigned lane = threadIdx.x & 31U;
    const unsigned warp = threadIdx.x / 32U;

    gather_accumulator sum_distance = 0, sum_weight = 0;
    std::uint32_t admitted = 0;

    for (std::uint32_t offset = threadIdx.x; offset < count;) {
        const auto index = start + offset;
        float weight = 0.0F, clipped = 0.0F;

        if (compact_contribution(prepared[compact_ray_id<narrow>(view, index)], view.entry_t[index],
                                  view.projected[index], settings, weight, clipped)) {
            sum_weight += static_cast<gather_accumulator>(weight);
            sum_distance += static_cast<gather_accumulator>(clipped) * static_cast<gather_accumulator>(weight);
            ++admitted;
        }

        if (count - offset <= block_threads) break;
        offset += block_threads;
    }


    #pragma unroll

    for (unsigned delta = 16; delta != 0; delta /= 2) {
        sum_distance += __shfl_down_sync(mask, sum_distance, delta);
        sum_weight += __shfl_down_sync(mask, sum_weight, delta);
        admitted += __shfl_down_sync(mask, admitted, delta);
    }
    __shared__ gather_accumulator warp_distances[warp_count];
    __shared__ gather_accumulator warp_weights[warp_count];
    __shared__ std::uint32_t warp_admitted[warp_count];

    if (lane == 0) {
        warp_distances[warp] = sum_distance;
        warp_weights[warp] = sum_weight;
        warp_admitted[warp] = admitted;
    }
    __syncthreads();

    if (warp == 0) {
        sum_distance = lane < warp_count ? warp_distances[lane] : gather_accumulator{0};
        sum_weight = lane < warp_count ? warp_weights[lane] : gather_accumulator{0};
        admitted = lane < warp_count ? warp_admitted[lane] : 0U;
        #pragma unroll

        for (unsigned delta = 16; delta != 0; delta /= 2) {
            sum_distance += __shfl_down_sync(mask, sum_distance, delta);
            sum_weight += __shfl_down_sync(mask, sum_weight, delta);
            admitted += __shfl_down_sync(mask, admitted, delta);
        }

        if (lane == 0)
            output[view.cells[row]] = finish_prepared(sum_distance, sum_weight, admitted, count, settings);
    }
}

template <bool narrow>
inline int launch_combined(layout_view view, const compact_ray* prepared, voxel* output,
    config settings, gather_policy policy, cudaStream_t stream)
{
    if (policy.first_entry_only) {
        compact_thread_gather<true, narrow><<<(view.rows + 255) / 256, 256, 0, stream>>>(view, 0, view.rows, prepared, settings, policy.max_rays, output);
        return int(cudaGetLastError());
    }

    if (view.short_rows)
        compact_thread_gather<false, narrow><<<(view.short_rows + 255) / 256, 256, 0, stream>>>(view, 0, view.short_rows, prepared, settings, policy.max_rays, output);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) return int(error);
    const auto count = view.rows - view.short_rows - (policy.max_rays ? 0 : view.block_rows);

    if (count) {
        if (!policy.max_rays)
            combined_pipelined_gather<narrow><<<(count + 7) / 8, 256, 0, stream>>>(view, view.short_rows, count, prepared, settings, 0, output);
        else if (policy.max_rays == 1)
            compact_warp_gather<true, narrow><<<(count + 7) / 8, 256, 0, stream>>>(view, view.short_rows, count, prepared, settings, 1, output);
        else
            compact_warp_gather<false, narrow><<<(count + 7) / 8, 256, 0, stream>>>(view, view.short_rows, count, prepared, settings, policy.max_rays, output);
    }
    error = cudaGetLastError();

    if (error != cudaSuccess) return int(error);

    if (!policy.max_rays && view.block_rows)
        combined_block_gather<narrow><<<view.block_rows, 128, 0, stream>>>(view, view.rows - view.block_rows, view.block_rows, prepared, settings, output);
     return int(cudaGetLastError());
}
}
