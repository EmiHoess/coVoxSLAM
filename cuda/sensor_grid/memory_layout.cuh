#pragma once
#include "prepared.cuh"

namespace covoxslam::sensor_grid
{


template <bool first_entry_only, bool narrow>
__global__ void layout_thread_gather(layout_view view, std::uint32_t offset, std::uint32_t rows,
    const prepared_ray* prepared, config settings, std::uint32_t max_rays, voxel* output)
{
    const std::size_t worker = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (worker >= rows) return;
    const std::size_t row = first_entry_only ? worker : view.row_ids[offset + worker];
    const auto start = view.ranges[row].start;
    const auto end = first_entry_only ? start + 1U : start + view.ranges[row].count;
    gather_accumulator sum_distance = 0, sum_weight = 0;
    std::uint32_t admitted = 0, tested = 0;
    for (auto step = start; step < end; ++step) {
        const auto index = start + (step - start) * view.short_stride;
        ++tested;
        float weight = 0, clipped = 0;
        if (!prepared_contribution(prepared[narrow ? view.ray_ids16[index] : view.ray_ids[index]], view.entry_t[index],
                                   view.projected[index], settings, weight, clipped)) continue;
        sum_weight += static_cast<gather_accumulator>(weight);
        sum_distance += static_cast<gather_accumulator>(weight) * static_cast<gather_accumulator>(clipped);
        ++admitted;
        if (max_rays && admitted == max_rays) break;
    }
    output[view.cells[row]] = finish_prepared(sum_distance, sum_weight, admitted, tested, settings);
}

template <bool cap_one, bool narrow>
__global__ void layout_warp_gather(layout_view view, std::uint32_t offset, std::uint32_t rows,
    const prepared_ray* prepared, config settings, std::uint32_t max_rays, voxel* output)
{
    constexpr unsigned mask = 0xffffffffU;
    const unsigned lane = threadIdx.x & 31U;
    const std::size_t worker = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (worker >= rows) return;
    const auto row = view.row_ids[offset + worker];
    const auto start = view.ranges[row].start, count = view.ranges[row].count;
    gather_accumulator sum_distance = 0, sum_weight = 0;
    std::uint32_t admitted = 0, tested = 0;
    for (std::uint32_t base = 0; base < count;) {
        const unsigned remaining = count - base;
        const unsigned width = remaining < 32U ? remaining : 32U;
        float weight = 0, clipped = 0;
        bool valid = false;
        if (lane < width) {
            const auto index = start + base + lane;
            valid = prepared_contribution(prepared[narrow ? view.ray_ids16[index] : view.ray_ids[index]], view.entry_t[index],
                                          view.projected[index], settings, weight, clipped);
        }
        const auto valid_mask = __ballot_sync(mask, valid);
        if constexpr (cap_one) {
            if (valid_mask) {
                const unsigned first = static_cast<unsigned>(__ffs(valid_mask) - 1);
                const auto selected_weight = static_cast<gather_accumulator>(__shfl_sync(mask, weight, first));
                const auto selected_distance = static_cast<gather_accumulator>(__shfl_sync(mask, clipped, first));
                if (lane == 0)
                    output[view.cells[row]] = finish_prepared(selected_weight * selected_distance,
                        selected_weight, 1, base + first + 1, settings);
                return;
            }
            tested = base + width;
        } else {
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
        }
        if (width == remaining) break;
        base += 32;
    }
    if constexpr (!cap_one) {
        for (unsigned delta = 16; delta != 0; delta /= 2) {
            sum_distance += __shfl_down_sync(mask, sum_distance, delta);
            sum_weight += __shfl_down_sync(mask, sum_weight, delta);
}

template<bool narrow>
int launch_memory_layout(layout_view view, const prepared_ray* prepared, voxel* output,
    config settings, gather_policy policy, cudaStream_t stream)
{
    if (policy.first_entry_only)
        layout_thread_gather<true, narrow><<<(view.rows + 255) / 256, 256, 0, stream>>>(view, 0, view.rows, prepared, settings, policy.max_rays, output);
    else {
        if (view.short_rows) layout_thread_gather<false, narrow><<<(view.short_rows + 255) / 256, 256, 0, stream>>>(view, 0, view.short_rows, prepared, settings, policy.max_rays, output);
        const auto count = view.rows - view.short_rows;
        if (count) {
            if (policy.max_rays == 1) layout_warp_gather<true, narrow><<<(count + 7) / 8, 256, 0, stream>>>(view, view.short_rows, count, prepared, settings, policy.max_rays, output);
            else layout_warp_gather<false, narrow><<<(count + 7) / 8, 256, 0, stream>>>(view, view.short_rows, count, prepared, settings, policy.max_rays, output);
        }
    }
    return int(cudaGetLastError());
}
}
    if (lane == 0)
        output[view.cells[row]] = finish_prepared(sum_distance, sum_weight, admitted, tested, settings);
}

}
