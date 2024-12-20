#pragma once
#include <covoxslam/sensor_grid_layout_types.h>
#include <covoxslam/sensor_grid_precision.h>
#include <cuda_runtime.h>

namespace covoxslam::sensor_grid
{
__global__ void prepare_rays(const float* ranges, std::uint32_t count, config settings, prepared_ray* output)
{
    const auto index = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float range = ranges[index];
    prepared_ray value;

    if (isfinite(range) && range > 0 && range >= settings.minimum_range) {
        const bool clearing = range > settings.radius;
        value.range = range;
        value.end = clearing ? fminf(fmaxf(range - settings.truncation, 0.f), settings.radius)
            : fminf(range + settings.truncation, settings.radius);
        value.flags = clearing ? 3 : 1;
    }
    output[index] = value;
}
__device__ inline bool prepared_contribution(prepared_ray ray, float entry, float projected,
    config settings, float& weight, float& clipped)
{
    if (!(ray.flags & 1U)) return false;
    const float distance = ray.range - projected;

    if (!(entry <= ray.end) || !(projected >= 0) || (!(ray.flags & 2U) && distance < -settings.truncation)) return false;
    weight = 1;
    if (settings.weight_dropoff && distance < -settings.voxel_size)
        weight = fmaxf((settings.truncation + distance) / (settings.truncation - settings.voxel_size), 0.f);

    if (fabsf(distance) < settings.truncation) weight *= settings.sparsity_compensation;
    if (!(weight > FLT_EPSILON)) return false;
    clipped = fmaxf(-settings.truncation, fminf(distance, settings.truncation));
    return true;
}
__device__ inline voxel finish_prepared(gather_accumulator weighted, gather_accumulator weights,
    std::uint32_t admitted, std::uint32_t tested, config settings)
{
    voxel result{0, 0, admitted, tested};

    if (admitted) {
        result.distance = static_cast<float>(weighted / weights);
        if constexpr (accumulation_bits == 32) result.weight = fminf(weights, settings.maximum_weight);
        else result.weight = static_cast<float>(fmin(weights, double(settings.maximum_weight)));
    }
    return result;
}
template<unsigned width>
__global__ void prepared_gather(layout_view view, const prepared_ray* prepared, config settings,
    gather_policy policy, voxel* output)
{
    static_assert(width == 1 || width == 32);
    const auto worker = (std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) / width;
    const unsigned lane = threadIdx.x % width;

    if (worker >= view.rows) return;
    const auto start = view.starts[worker];
    const auto count = policy.first_entry_only ? 1U : view.lengths[worker];
    gather_accumulator weighted{}, weights{};
    std::uint32_t admitted{}, tested{};

    for (std::uint32_t base = 0; base < count;)
    {
        const auto remaining = count - base, chunk = remaining < width ? remaining : width;
        float weight{}, clipped{};
        bool valid{};

        if (lane < chunk) {
            const auto index = start + (base + lane) * view.stride;
            valid = prepared_contribution(prepared[view.ray_ids[index]], view.entry_t[index], view.projected[index], settings, weight, clipped);
        }
        unsigned available = valid ? 1U : 0U, rank{};
        if constexpr (width == 32) {
            const unsigned mask = __ballot_sync(0xffffffffU, valid);
            available = __popc(mask);
            rank = __popc(mask & (lane ? (0xffffffffU >> (32 - lane)) : 0U));
        }
        const bool limited = policy.max_rays && available >= policy.max_rays - admitted;
        const unsigned take = limited ? policy.max_rays - admitted : available;
        const bool selected = valid && rank < take;
        unsigned selected_mask = selected ? 1 : 0;
        if constexpr (width == 32) selected_mask = __ballot_sync(0xffffffffU, selected);

        if (selected) {
            weighted += gather_accumulator(weight) * gather_accumulator(clipped);
             weights += weight;
        }
        admitted += take;
        tested = base + (limited ? 32U - __clz(selected_mask) : chunk);
        if (limited || chunk == remaining) break;
        base += width;
    }
    if constexpr (width == 32)
        for (unsigned delta = 16; delta; delta /= 2) {
            weighted += __shfl_down_sync(0xffffffffU, weighted, delta);
            weights += __shfl_down_sync(0xffffffffU, weights, delta);
        }

    if (!lane) output[view.cells[worker]] = finish_prepared(weighted, weights, admitted, tested, settings);
}
}
