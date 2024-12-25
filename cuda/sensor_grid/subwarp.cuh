#pragma once
#include "prepared.cuh"

namespace covoxslam::sensor_grid
{
enum class subwarp_selection { all, cap_one, capped, first_entry };

template <subwarp_selection selection>
__global__ void subwarp_thread_gather_kernel(layout_view view, std::uint32_t offset, std::uint32_t rows,
    const prepared_ray* prepared, settings settings, std::uint32_t max_rays, voxel* output) {
    const std::size_t worker = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (worker >= rows) return;
    const auto row = selection == subwarp_selection::first_entry ? worker : view.row_ids[offset + worker];
    const auto start = view.starts[row];
    const auto end = selection == subwarp_selection::first_entry ? start + 1U : start + view.lengths[row];
    gather_accumulator sum_distance = 0, sum_weight = 0;
    std::uint32_t admitted = 0, tested = 0;

    for (auto index = start; index < end; ++index) {
        ++tested;
        float weight = 0, clipped = 0;

        if (!prepared_contribution(prepared[view.ray_ids[index]], view.entry_t[index],
                                   view.projected[index], settings, weight, clipped)) continue;
        sum_weight += static_cast<gather_accumulator>(weight);
        sum_distance += static_cast<gather_accumulator>(clipped) * static_cast<gather_accumulator>(weight);
        ++admitted;
        if constexpr (selection == subwarp_selection::cap_one) break;
        if constexpr (selection == subwarp_selection::capped)

            if (max_rays && admitted == max_rays) break;
    }
    output[view.cells[row]] = finish_prepared(sum_distance, sum_weight, admitted, tested, settings);
}

template <unsigned width_limit, subwarp_selection selection>
__global__ void subwarp_group_gather_kernel(layout_view view, std::uint32_t offset, std::uint32_t rows,
    const prepared_ray* prepared, settings settings, std::uint32_t max_rays, voxel* output) {
    static_assert(width_limit == 8 || width_limit == 16 || width_limit == 32);
    static_assert(selection != subwarp_selection::first_entry);
    const unsigned lane = threadIdx.x % width_limit;
    const unsigned group_start = (threadIdx.x & 31U) - lane;
    constexpr unsigned local_mask = 0xffffffffU >> (32U - width_limit);
    const unsigned mask = local_mask << group_start;
    const std::size_t worker = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / width_limit;

    if (worker >= rows) return;
    const auto row = view.row_ids[offset + worker], start = view.starts[row], count = view.lengths[row];
    gather_accumulator sum_distance = 0, sum_weight = 0;
    std::uint32_t admitted = 0, tested = 0;

    for (std::uint32_t base = 0; base < count;) {
        const unsigned remaining = count - base;
        const unsigned width = remaining < width_limit ? remaining : width_limit;
        float weight = 0, clipped = 0;
        bool valid = false;

        if (lane < width) {
            const auto index = start + base + lane;
            valid = prepared_contribution(prepared[view.ray_ids[index]], view.entry_t[index],
                                          view.projected[index], settings, weight, clipped);
        }

        const unsigned valid_mask = (__ballot_sync(mask, valid) >> group_start) & local_mask;
        if constexpr (selection == subwarp_selection::all) {

            if (valid) {
                sum_weight += static_cast<gather_accumulator>(weight);
                sum_distance += static_cast<gather_accumulator>(clipped) * static_cast<gather_accumulator>(weight);
            }
            admitted += __popc(valid_mask);
        } else if constexpr (selection == subwarp_selection::cap_one) {

            if (valid_mask) {
                const unsigned first = static_cast<unsigned>(__ffs(valid_mask) - 1);
                const auto w = static_cast<gather_accumulator>(__shfl_sync(mask, weight, first, width_limit));
                const auto d = static_cast<gather_accumulator>(__shfl_sync(mask, clipped, first, width_limit));

                if (lane == 0)
                    output[view.cells[row]] = finish_prepared(w * d, w, 1, base + first + 1, settings);
                return;
            }
            tested = base + width;
        } else {
            const unsigned prefix = lane == 0 ? 0U : (0xffffffffU >> (32U - lane));
            const unsigned rank = __popc(valid_mask & prefix);
            const unsigned available = __popc(valid_mask);
            const bool limited = max_rays != 0 && available >= max_rays - admitted;
            const unsigned take = limited ? max_rays - admitted : available;
            const bool selected = valid && rank < take;
            const unsigned selected_mask = (__ballot_sync(mask, selected) >> group_start) & local_mask;

            if (selected) {
                sum_weight += static_cast<gather_accumulator>(weight);
                sum_distance += static_cast<gather_accumulator>(clipped) * static_cast<gather_accumulator>(weight);
            }
            admitted += take;
            tested = base + (limited ? 32U - __clz(selected_mask) : width);

            if (limited) break;
        }

        if (width == remaining) break;
        base += width_limit;
    }
    if constexpr (selection != subwarp_selection::cap_one)

        for (unsigned delta = width_limit / 2; delta != 0; delta /= 2) {
            sum_distance += __shfl_down_sync(mask, sum_distance, delta, width_limit);
            sum_weight += __shfl_down_sync(mask, sum_weight, delta, width_limit);
        }
    if constexpr (selection == subwarp_selection::all) tested = count;

    if (lane == 0)
        output[view.cells[row]] = finish_prepared(sum_distance, sum_weight, admitted, tested, settings);
}


template<subwarp_selection selection>
int launch_subwarp(layout_view view, const prepared_ray* prepared, voxel* output,
    config settings, gather_policy policy, cudaStream_t stream)
{
    const auto selected = policy.max_rays ? 1U : 0U;
    std::uint32_t offset = selected * view.rows;

    for (unsigned bin = 0; bin < 4; ++bin) {
        const auto count = view.bin_ends[selected][bin] - offset;
        if (count) {
            if (bin == 0) subwarp_thread_gather_kernel<selection><<<(count + 255) / 256, 256, 0, stream>>>(view, offset, count, prepared, settings, policy.max_rays, output);
            else if (bin == 1) subwarp_group_gather_kernel<8, selection><<<(count + 31) / 32, 256, 0, stream>>>(view, offset, count, prepared, settings, policy.max_rays, output);
            else if (bin == 2) subwarp_group_gather_kernel<16, selection><<<(count + 15) / 16, 256, 0, stream>>>(view, offset, count, prepared, settings, policy.max_rays, output);
            else subwarp_group_gather_kernel<32, selection><<<(count + 7) / 8, 256, 0, stream>>>(view, offset, count, prepared, settings, policy.max_rays, output);
            const auto error = cudaGetLastError();
            if (error != cudaSuccess) return int(error);
        }
        offset += count;
    }
    return 0;
}
}
