#include "sparse_mapping.h"
#include "ray_resident/record.cuh"
#include "ray_resident_reduce/record.cuh"
#include "ray_resident_step_tasks/tasks.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>
#include <climits>
#include <cmath>
#include <initializer_list>
#include <limits>

namespace covoxslam::detail
{
namespace
{
template<bool emit, unsigned resident_stage = 3>
__device__ std::uint32_t walk_ray(const cuda_device_ray& ray, cuda_ray_parameters config,
    cuda_ray_contribution* output, std::uint64_t offset, std::uint32_t expected,
    std::uint64_t capacity, std::uint32_t* status, cuda_resident_view resident = {}, std::uint32_t input_index = 0,
    cuda_resident_reduction_view reduction = {}, cuda_resident_task_view tasks = {})
{
    float direction[3];
    for (int axis = 0; axis < 3; ++axis) {
        if (!isfinite(ray.origin[axis]) || !isfinite(ray.endpoint[axis])) return 0;
        direction[axis] = ray.endpoint[axis] - ray.origin[axis];
    }
    const float distance = static_cast<float>(hypot(hypot(double(direction[0]), double(direction[1])), double(direction[2])));
    if (!isfinite(distance) || distance < config.minimum_ray_length_m ||
        !isfinite(ray.weight) || ray.weight <= 0) return 0;
    const bool clearing = distance > config.maximum_ray_length_m;
    if (clearing && (!config.allow_clear || !config.voxel_carving)) return 0;
    const double surface_start = fmax(0.0, double(distance) - config.truncation_distance);
    const double start = config.voxel_carving ? 0.0 : surface_start;
    const double end = clearing ? fmin(surface_start, double(config.maximum_ray_length_m))
                               : double(distance) + config.truncation_distance;
    if (end <= start) return 0;
     int voxel[3], step[3];
    double next[3], delta[3];
    for (int axis = 0; axis < 3; ++axis)
    {
        direction[axis] /= distance;
        const double coordinate = floor((double(ray.origin[axis]) + direction[axis] * start) / config.voxel_size);
        if (!isfinite(coordinate) || coordinate < INT_MIN || coordinate > INT_MAX) {
            atomicOr(status, resident_stage != 3 ? unsigned(resident_traversal_error) : 1U);
            return 0;
        }
        voxel[axis] = static_cast<int>(coordinate);
        step[axis] = direction[axis] > 0 ? 1 : direction[axis] < 0 ? -1 : 0;
        if (!step[axis]) {
            next[axis] = delta[axis] = CUDART_INF;
            continue;
        }
        const double boundary = config.voxel_size * (coordinate + (step[axis] > 0 ? 1 : 0));
         next[axis] = fmax(start, (boundary - ray.origin[axis]) / direction[axis]);
        delta[axis] = double(config.voxel_size) / fabs(double(direction[axis]));
    }
    if constexpr (resident_stage == 8)
        tasks.prepared[input_index] = {{direction[0], direction[1], direction[2]}, distance, start, end};
    const auto maximum_steps = static_cast<std::uint32_t>(ceil((end - start) / config.voxel_size) * 3) + 16;
    std::uint32_t count{};
    double entered{start};
    bool complete{};
    for (std::uint32_t iteration = 0; iteration < maximum_steps; ++iteration)
    {
        const double boundary = fmin(next[0], fmin(next[1], next[2]));
        float projected{};
        if constexpr (resident_stage < 8) {
        for (int axis = 0; axis < 3; ++axis) {
            const float center = static_cast<float>((double(voxel[axis]) + 0.5) * config.voxel_size);
            if (!isfinite(center)) {
                atomicOr(status, resident_stage != 3 ? unsigned(resident_traversal_error) : 1U);
                return count;
            }
             projected += (center - ray.origin[axis]) * direction[axis];
        }
        if (!isfinite(projected)) {
            atomicOr(status, resident_stage != 3 ? unsigned(resident_traversal_error) : 1U);
            return count;
        }
        }
        if (boundary > entered && (resident_stage >= 8 || (projected >= start && projected <= end)))
        {
            if constexpr (resident_stage == 9) {
                if (!record_resident_task(resident, tasks, voxel, input_index, count)) return count;
            }
            else if constexpr (resident_stage == 8) {
                if (count >= tasks.capacity) {
                    atomicOr(status, resident_contribution_capacity_error);
                    return count;
                }
            }
            else if constexpr (resident_stage >= 4) {
                if (!record_resident_reduction<resident_stage>(resident, reduction, voxel,
                    distance - projected, ray.weight, input_index, count, config.truncation_distance)) return count;
            }
            else if constexpr (resident_stage < 3) {
                if (!record_resident_voxel<resident_stage>(resident, voxel,
                    fminf(config.truncation_distance, fmaxf(-config.truncation_distance, distance - projected)),
                    ray.weight, ray.ordinal, count, input_index)) return count;
            }
            else if constexpr (emit) {
                if (count >= expected || offset > capacity || count >= capacity - offset)
                {
                    atomicOr(status, 4U);
                    return count;
                }
                output[offset + count] = {{voxel[0], voxel[1], voxel[2]},
                    fminf(config.truncation_distance, fmaxf(-config.truncation_distance, distance - projected)),
                    ray.weight, ray.ordinal, count};
            }
            ++count;
        }
        if (!isfinite(boundary) || boundary >= end) {
            complete = true;
            break;
        }
        for (int axis = 0; axis < 3; ++axis)
        {
            if (next[axis] > boundary + 1.0e-6) continue;
            const auto candidate = static_cast<long long>(voxel[axis]) + step[axis];
             if (candidate < INT_MIN || candidate > INT_MAX)
            {
                atomicOr(status, resident_stage != 3 ? unsigned(resident_traversal_error) : 1U);
                return count;
            }
            voxel[axis] = static_cast<int>(candidate);
            next[axis] += delta[axis];
        }
        entered = boundary;
    }
    if (!complete) atomicOr(status, resident_stage != 3 ? unsigned(resident_traversal_error) : 2U);
    if constexpr (emit) if (count != expected) atomicOr(status, 4U);
    if constexpr (resident_stage == 9)
        if (count != tasks.counts[input_index]) atomicOr(status, resident_scatter_error);
    return count;
}

__global__ void set_resident_task_phase(cuda_resident_view view, cuda_resident_task_view tasks,
    std::uint32_t rays, unsigned phase)
{
    if (view.state->error) return;

    if (!view.state->batch_open || view.state->batch_reduction != static_cast<unsigned>(cuda_resident_reduction::direct_atomic)) {
        view.state->error = resident_scatter_error;
        return;
    }
    if (view.state->batch_task_phase + 1 != phase) { view.state->error = resident_scatter_error; return; }
    view.state->batch_task_phase = phase;
    if (phase == 1) { *tasks.state = {0, 1, rays}; return; }

    if (tasks.state->phase + 1 != phase || tasks.state->rays != rays) {
        view.state->error = resident_scatter_error;
        return;
    }
    if (phase == 2) {
        const auto count = rays ? tasks.offsets[rays - 1] + tasks.counts[rays - 1] : 0;

        if (count > tasks.capacity) { view.state->error = resident_contribution_capacity_error; return; }
        tasks.state->count = count;
    }
    tasks.state->phase = phase;
}

template<unsigned stage>
__global__ void walk_resident_tasks(cuda_resident_view view, cuda_resident_task_view tasks,
    const cuda_device_ray* rays, std::uint32_t count, cuda_ray_parameters config)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= count) return;
    if constexpr (stage == 8) { tasks.counts[index] = tasks.accepted[index] = 0; tasks.prepared[index] = {}; }
    if (atomicAdd(&view.state->error, 0U)) return;

    if (tasks.state->phase != stage - 7 || tasks.state->rays != count) {
        atomicOr(&view.state->error, resident_scatter_error);
        return;
    }
    const auto visits = walk_ray<false, stage>(rays[index], config, nullptr, 0, 0, 0,
        &view.state->error, view, static_cast<std::uint32_t>(index), {}, tasks);
    if constexpr (stage == 8) tasks.counts[index] = visits;
}

template<unsigned stage>
__global__ void walk_resident_rays(cuda_resident_view view, const cuda_device_ray* rays,
    std::size_t size, cuda_ray_parameters parameters)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size || atomicAdd(&view.state->error, 0U))
        return;
    const auto count = walk_ray<false, stage>(rays[index], parameters, nullptr, 0, 0, 0,
        &view.state->error, view, static_cast<std::uint32_t>(index));
    if constexpr (stage == 1)
        atomicAdd(count ? &view.state->batch_integrated : &view.state->batch_rejected, 1U);
}

template<unsigned stage>
__global__ void walk_reduction_rays(cuda_resident_view view, cuda_resident_reduction_view reduction,
    const cuda_device_ray* rays, std::size_t size, cuda_ray_parameters parameters, cuda_resident_reduction mode)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= size || atomicAdd(&view.state->error, 0U)) return;
    if (!view.state->batch_open || view.state->batch_reduction != static_cast<std::uint32_t>(mode) || view.state->batch_task_phase) {
        atomicOr(&view.state->error, resident_scatter_error);
        return;
    }
    const auto count = walk_ray<false, stage>(rays[index], parameters, nullptr, 0, 0, 0,
        &view.state->error, view, static_cast<std::uint32_t>(index), reduction);
    if constexpr (stage == 4 || stage == 7)
    {
        if (reduction.ray_counts) reduction.ray_counts[index] = count;
        atomicAdd(count ? &view.state->batch_integrated : &view.state->batch_rejected, 1U);
         atomicAdd(reinterpret_cast<unsigned long long*>(&view.state->batch_contributions), static_cast<unsigned long long>(count));
    }
}

__global__ void count_rays(const cuda_device_ray* rays, std::size_t size, cuda_ray_parameters config,
                            std::uint32_t* counts, std::uint32_t* status)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size) counts[index] = walk_ray<false>(rays[index], config, nullptr, 0, 0, 0, status);
}

__global__ void emit_rays(const cuda_device_ray* rays, std::size_t size, cuda_ray_parameters config,
    const std::uint32_t* counts, const std::uint64_t* offsets, cuda_ray_contribution* output,
    std::uint64_t capacity, std::uint32_t* status)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
     if (index < size) walk_ray<true>(rays[index], config, output, offsets[index], counts[index], capacity, status);
}

bool valid_ray_parameters(cuda_ray_parameters value)
{
    for (const float item : {value.voxel_size, value.truncation_distance,
                             value.minimum_ray_length_m, value.maximum_ray_length_m})
        if (!std::isfinite(item) || item <= 0) return false;
    return value.minimum_ray_length_m <= value.maximum_ray_length_m &&
        double(value.maximum_ray_length_m) + value.truncation_distance <= std::numeric_limits<float>::max() &&
        (double(value.maximum_ray_length_m) + value.truncation_distance) / value.voxel_size < 10000000;
}
}

int cuda_expand_resident_tasks(cuda_resident_view view, cuda_resident_task_view tasks,
    const cuda_device_ray* rays, std::size_t count, cuda_ray_parameters config, void* stream) noexcept
{
    if (!valid_resident_map(view) || !valid_resident_tasks(tasks) || !view.block_sums ||
        !valid_ray_parameters(config) || count > tasks.ray_capacity || (count && !rays))
        return static_cast<int>(cudaErrorInvalidValue);
    const auto queue = static_cast<cudaStream_t>(stream);
    const auto blocks = count ? static_cast<unsigned>(1 + (count - 1) / 256) : 0;
    const auto size = static_cast<std::uint32_t>(count);
    set_resident_task_phase<<<1, 1, 0, queue>>>(view, tasks, size, 1);

    if (count) walk_resident_tasks<8><<<blocks, 256, 0, queue>>>(view, tasks, rays, size, config);
    auto status = static_cast<int>(cudaGetLastError());
    if (status) return status;
    status = cuda_scan_counts(tasks.counts, tasks.offsets, view.block_sums, count, stream);
    if (status) return status;
    set_resident_task_phase<<<1, 1, 0, queue>>>(view, tasks, size, 2);

    if (count) walk_resident_tasks<9><<<blocks, 256, 0, queue>>>(view, tasks, rays, size, config);
    set_resident_task_phase<<<1, 1, 0, queue>>>(view, tasks, size, 3);
    return static_cast<int>(cudaGetLastError());
}

int cuda_walk_resident_rays(cuda_resident_view view, const cuda_device_ray* rays, std::size_t count,
    cuda_ray_parameters parameters, unsigned stage, void* stream) noexcept
{
    if (!valid_resident_view(view) || !valid_ray_parameters(parameters) || count > UINT32_MAX ||
        (count && !rays) || stage > 2) return static_cast<int>(cudaErrorInvalidValue);
    if (!count) return static_cast<int>(cudaSuccess);
    const auto blocks = static_cast<unsigned>(1 + (count - 1) / 256);
     const auto queue = static_cast<cudaStream_t>(stream);
    switch (stage)
    {
    case 0: walk_resident_rays<0><<<blocks, 256, 0, queue>>>(view, rays, count, parameters); break;
    case 1: walk_resident_rays<1><<<blocks, 256, 0, queue>>>(view, rays, count, parameters); break;
    case 2: walk_resident_rays<2><<<blocks, 256, 0, queue>>>(view, rays, count, parameters); break;
    }
    return static_cast<int>(cudaGetLastError());
}

int cuda_walk_resident_reduction(cuda_resident_view view, cuda_resident_reduction_view reduction,
    const cuda_device_ray* rays, std::size_t count, cuda_ray_parameters parameters,
    cuda_resident_reduction mode, unsigned stage, void* stream) noexcept
{
    reduction = resident_reduction_storage(reduction, mode);

    if (!valid_resident_reduction_view(view, reduction, mode) || !valid_ray_parameters(parameters) ||
        count > UINT32_MAX || (count && !rays) || stage > 2 ||
        (reduction.ray_counts && count > reduction.ray_capacity) ||
        (mode == cuda_resident_reduction::direct_atomic && stage == 2)) return static_cast<int>(cudaErrorInvalidValue);
    if (!count) return static_cast<int>(cudaSuccess);
    const auto blocks = static_cast<unsigned>(1 + (count - 1) / 256);
    const auto queue = static_cast<cudaStream_t>(stream);

    if (!stage) walk_reduction_rays<0><<<blocks, 256, 0, queue>>>(view, reduction, rays, count, parameters, mode);
    else if (mode == cuda_resident_reduction::direct_atomic)
        walk_reduction_rays<7><<<blocks, 256, 0, queue>>>(view, reduction, rays, count, parameters, mode);
    else if (stage == 1) walk_reduction_rays<4><<<blocks, 256, 0, queue>>>(view, reduction, rays, count, parameters, mode);
    else if (mode == cuda_resident_reduction::index)
        walk_reduction_rays<5><<<blocks, 256, 0, queue>>>(view, reduction, rays, count, parameters, mode);
    else walk_reduction_rays<6><<<blocks, 256, 0, queue>>>(view, reduction, rays, count, parameters, mode);
    return static_cast<int>(cudaGetLastError());
}

int cuda_count_ray_contributions(const cuda_device_ray* rays, std::size_t count,
    cuda_ray_parameters parameters, std::uint32_t* counts, std::uint32_t* status, void* stream) noexcept
{
    if (!valid_ray_parameters(parameters) || count > UINT32_MAX || (count && (!rays || !counts || !status)))
        return static_cast<int>(cudaErrorInvalidValue);
    if (!count) return static_cast<int>(cudaSuccess);
    const auto blocks = static_cast<unsigned>(1 + (count - 1) / 256);
    count_rays<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(rays, count, parameters, counts, status);
    return static_cast<int>(cudaGetLastError());
}

int cuda_emit_ray_contributions(const cuda_device_ray* rays, std::size_t count,
    cuda_ray_parameters parameters, const std::uint32_t* counts, const std::uint64_t* offsets,
    cuda_ray_contribution* output, std::uint64_t capacity, std::uint32_t* status, void* stream) noexcept
{
    if (!valid_ray_parameters(parameters) || count > UINT32_MAX ||
        (count && (!rays || !counts || !offsets || !status || (capacity && !output))))
        return static_cast<int>(cudaErrorInvalidValue);
    if (!count) return static_cast<int>(cudaSuccess);
     const auto blocks = static_cast<unsigned>(1 + (count - 1) / 256);
    emit_rays<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        rays, count, parameters, counts, offsets, output, capacity, status);
    return static_cast<int>(cudaGetLastError());
}
}

namespace covoxslam::detail
{
namespace
{
__global__ void summarize_rays(const std::uint32_t* counts, const std::uint64_t* offsets,
                               std::size_t size, cuda_ray_summary* summary)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size && counts[index]) atomicAdd(&summary->contributing_rays, 1U);
     if (index + 1 == size) summary->contributions = offsets[index] + counts[index];
}
}

int cuda_summarize_ray_contributions(const std::uint32_t* counts, const std::uint64_t* offsets,
    std::size_t count, cuda_ray_summary* summary, void* stream) noexcept
{
    if (count > UINT32_MAX || (count && (!counts || !offsets || !summary)))
        return static_cast<int>(cudaErrorInvalidValue);
    if (!count) return static_cast<int>(cudaSuccess);
    const auto blocks = static_cast<unsigned>(1 + (count - 1) / 256);
    summarize_rays<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(counts, offsets, count, summary);
    return static_cast<int>(cudaGetLastError());
}
}
