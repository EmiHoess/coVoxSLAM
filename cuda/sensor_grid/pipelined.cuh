#pragma once
#include "prepared.cuh"

namespace covoxslam::sensor_grid
{
struct staged_sample
{
    prepared_ray ray{};
    float entry_t{}, projected{};
};
__device__ inline staged_sample load_staged_sample(layout_view view, std::uint32_t index,
    const prepared_ray* prepared)
{
    return {prepared[view.ray_ids[index]], view.entry_t[index], view.projected[index]};
}
__device__ inline staged_sample load_staged_chunk(layout_view view, std::uint32_t start,
    std::uint32_t remaining, unsigned lane, const prepared_ray* prepared)
{
    if (lane >= remaining) return {};
    return load_staged_sample(view, start + lane, prepared);
}
__global__ void pipelined_warp_gather(layout_view view, std::uint32_t offset, std::uint32_t rows,
    const prepared_ray* prepared, config settings, std::uint32_t max_rays, voxel* output) {
    constexpr unsigned mask = 0xffffffffU;
    const unsigned lane = threadIdx.x & 31U;
    const std::size_t worker = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;

    if (worker >= rows) return;
    const std::size_t row = view.row_ids[offset + worker];
    const auto start = view.starts[row], count = view.lengths[row];
    gather_accumulator sum_distance = 0, sum_weight = 0;
    std::uint32_t admitted = 0, tested = 0;
    auto current = load_staged_chunk(view, start, count, lane, prepared);

    for (std::uint32_t base = 0; base < count;) {
        const unsigned remaining = count - base;
        const unsigned width = remaining < 32U ? remaining : 32U;
        staged_sample next;


        if (remaining > 32U)
            next = load_staged_chunk(view, start + base + 32U, remaining - 32U, lane, prepared);

        float weight = 0, clipped = 0;
        const bool valid = lane < width && prepared_contribution(current.ray,
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
}
