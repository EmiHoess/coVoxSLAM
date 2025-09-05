#include <covoxslam/sensor_grid_layout_types.h>
#include <covoxslam/sensor_grid_precision.h>
#include <cuda_runtime.h>

namespace covoxslam::sensor_grid
{
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
        if (isfinite(result.distance))
            result.distance = fmaxf(-settings.truncation, fminf(result.distance, settings.truncation));
        if constexpr (accumulation_bits == 32) result.weight = fminf(weights, settings.maximum_weight);
        else result.weight = static_cast<float>(fmin(weights, double(settings.maximum_weight)));
    }
    return result;
}
}

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
    if (!(ray.range > 0) || !isfinite(ray.range) || !isfinite(ray.end)) return false;
    const prepared_ray expanded{ray.range, ray.end, ray.range > 0 ? (ray.range > settings.radius ? 3U : 1U) : 0U};
    return prepared_contribution(expanded, entry, projected, settings, weight, clipped);
}



template <bool narrow>
__device__ inline std::uint32_t compact_ray_id(layout_view view, std::uint32_t index)
{
    if constexpr (narrow) return view.ray_ids16[index];
    return view.ray_ids[index];
}
template <bool first_entry_only, bool narrow = false>
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
        if (!compact_contribution(prepared[compact_ray_id<narrow>(view, index)], view.entry_t[index],
                                   view.projected[index], settings, weight, clipped)) continue;
        sum_weight += static_cast<gather_accumulator>(weight);
        sum_distance += static_cast<gather_accumulator>(weight) * static_cast<gather_accumulator>(clipped);
        ++admitted;
        if (max_rays && admitted == max_rays) break;
    }
    output[view.cells[row]] = finish_prepared(sum_distance, sum_weight, admitted, tested, settings);
}

template <bool cap_one, bool narrow = false>
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
            valid = compact_contribution(prepared[compact_ray_id<narrow>(view, index)], view.entry_t[index],
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

#include "../../sensor_grid/native.h"

namespace covoxslam::sensor_grid
{
int cuda_sensor_combined(layout_view view, const float* ranges, compact_ray* prepared, voxel* output,
    config settings, std::uint32_t ray_count, gather_policy policy, void* opaque_stream)
{
    if (!view.rows || view.rows > 16777216 || view.rows > view.entries || !ray_count || ray_count > 1048576 ||
        !view.starts || !view.lengths || !view.cells || !view.row_ids ||
        (!view.ray_ids && !view.ray_ids16) || (view.ray_ids16 && ray_count > 65536) ||
        !view.entry_t || !view.projected || !ranges || !prepared || !output ||
        view.stride != 1 || view.short_rows > view.rows || view.block_rows > view.rows - view.short_rows ||
        !valid_gather_policy(policy) || !safe_accumulation(settings, ray_count) ||
        !(settings.voxel_size > 0) || !(settings.truncation > settings.voxel_size) ||
        !(settings.radius > 0) || !std::isfinite(settings.radius)) return int(cudaErrorInvalidValue);
    const auto stream = static_cast<cudaStream_t>(opaque_stream);
    prepare_compact<<<(ray_count + 255) / 256, 256, 0, stream>>>(ranges, ray_count, settings, prepared);
    const auto error = cudaGetLastError();

    if (error != cudaSuccess) return int(error);
    return view.ray_ids16 ? launch_combined<true>(view, prepared, output, settings, policy, stream)
        : launch_combined<false>(view, prepared, output, settings, policy, stream);
}
}
