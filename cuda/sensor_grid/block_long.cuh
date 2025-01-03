#pragma once
#include "prepared.cuh"

namespace covoxslam::sensor_grid
{
template <unsigned block_threads = 128>
__global__ __launch_bounds__(block_threads)
void block_long_gather(layout_view view, std::uint32_t plan_offset, std::uint32_t rows,
    const prepared_ray* prepared, config settings, voxel* output) {
    static_assert(block_threads == 128 || block_threads == 256,
        "long-row gather requires 128 or 256 threads per block");
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

        if (prepared_contribution(prepared[view.ray_ids[index]], view.entry_t[index],
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


}
