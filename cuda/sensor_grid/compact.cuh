#pragma once
#include "prepared.cuh"

namespace covoxslam::sensor_grid
{
__global__ void prepare_compact(const float* ranges, std::uint32_t count, config settings, compact_ray* output)
{
    const auto ray = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ray >= count) return;
    const float range = ranges[ray];
    compact_ray result{};

    if (isfinite(range) && range > 0 && range >= settings.minimum_range) {
        result.range = range;
        result.end = range > settings.radius ? fminf(fmaxf(range - settings.truncation, 0.f), settings.radius)
            : fminf(range + settings.truncation, settings.radius);
    }
    output[ray] = result;
}
__device__ inline bool compact_contribution(compact_ray ray, float entry, float projected,
    config settings, float& weight, float& clipped)
{
    const prepared_ray expanded{ray.range, ray.end, ray.range > 0 ? (ray.range > settings.radius ? 3U : 1U) : 0U};
    return prepared_contribution(expanded, entry, projected, settings, weight, clipped);
}



template <bool first_entry_only>
__global__ void compact_thread_gather(layout_view view, std::uint32_t offset, std::uint32_t rows,
    const compact_ray* prepared, config settings, std::uint32_t max_rays, voxel* output)
{
    const std::size_t worker = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (worker >= rows) return;
    const std::size_t row = first_entry_only ? worker : view.row_ids[offset + worker];
    const auto start = view.starts[row];
    const auto end = first_entry_only ? start + 1U : start + view.lengths[row];
    gather_accumulator sum_distance = 0, sum_weight = 0;
    std::uint32_t admitted = 0, tested = 0;
    for (auto index = start; index < end; ++index) {
        ++tested;
        float weight = 0, clipped = 0;
        if (!compact_contribution(prepared[view.ray_ids[index]], view.entry_t[index],
                                   view.projected[index], settings, weight, clipped)) continue;
        sum_weight += static_cast<gather_accumulator>(weight);
        sum_distance += static_cast<gather_accumulator>(weight) * static_cast<gather_accumulator>(clipped);
        ++admitted;
        if (max_rays && admitted == max_rays) break;
    }
    output[view.cells[row]] = finish_prepared(sum_distance, sum_weight, admitted, tested, settings);
}

template <bool cap_one>
__global__ void compact_warp_gather(layout_view view, std::uint32_t offset, std::uint32_t rows,
    const compact_ray* prepared, config settings, std::uint32_t max_rays, voxel* output)
{
    constexpr unsigned mask = 0xffffffffU;
    const unsigned lane = threadIdx.x & 31U;
    const std::size_t worker = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    if (worker >= rows) return;
    const auto row = view.row_ids[offset + worker];
    const auto start = view.starts[row], count = view.lengths[row];
    gather_accumulator sum_distance = 0, sum_weight = 0;
    std::uint32_t admitted = 0, tested = 0;
    for (std::uint32_t base = 0; base < count;) {
        const unsigned remaining = count - base;
        const unsigned width = remaining < 32U ? remaining : 32U;
        float weight = 0, clipped = 0;
        bool valid = false;
        if (lane < width) {
            const auto index = start + base + lane;
            valid = compact_contribution(prepared[view.ray_ids[index]], view.entry_t[index],
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
    }
    if (lane == 0)
        output[view.cells[row]] = finish_prepared(sum_distance, sum_weight, admitted, tested, settings);
}

}
