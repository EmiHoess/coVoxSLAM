#pragma once

#include "tasks.cuh"
#include "../ray_resident_reduce/record.cuh"

namespace covoxslam::detail
{
__global__ void check_resident_task_phase(cuda_resident_view view, cuda_resident_task_view tasks,
    unsigned phase, std::uint32_t rays, bool advance)
{
    if (view.state->error) return;

    if (!view.state->batch_open || view.state->batch_reduction != static_cast<unsigned>(cuda_resident_reduction::direct_atomic) ||
        view.state->batch_task_phase != phase || tasks.state->phase != phase ||
        tasks.state->count > tasks.capacity || tasks.state->rays != rays) {
        view.state->error = resident_scatter_error;
        return;
    }
    if (advance) tasks.state->phase = view.state->batch_task_phase = phase + 1;
}

__global__ void evaluate_resident_tasks(cuda_resident_view view, cuda_resident_task_view tasks,
    const cuda_device_ray* rays, cuda_ray_parameters config)
{
    if (atomicAdd(&view.state->error, 0U)) return;

    for (std::uint64_t index = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        index < tasks.state->count; index += std::uint64_t(gridDim.x) * blockDim.x)
    {
        auto& task = tasks.tasks[index];
        const auto ray = task.payload[0];

        if (ray >= tasks.state->rays) { atomicOr(&view.state->error, resident_scatter_error); return; }
        float distance{};
        bool invalid{};

        if (evaluate_resident_task(rays[ray], tasks.prepared[ray], task.voxel, config, distance, invalid)) {
            resident_discover_tile(view, {task.voxel[0] / 8 - (task.voxel[0] % 8 < 0),
                task.voxel[1] / 8 - (task.voxel[1] % 8 < 0), task.voxel[2] / 8 - (task.voxel[2] % 8 < 0)});
            atomicAdd(tasks.accepted + ray, 1U);
            task.payload[0] = __float_as_uint(distance);
             task.payload[1] = __float_as_uint(rays[ray].weight);
        }
        else task.payload[0] = task.payload[1] = 0;

        if (invalid) atomicOr(&view.state->error, resident_numeric_error);
    }
}

__global__ void accumulate_resident_tasks(cuda_resident_view view, cuda_resident_reduction_view reduction,
    cuda_resident_task_view tasks, float truncation)
{
    if (atomicAdd(&view.state->error, 0U)) return;

    for (std::uint64_t index = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        index < tasks.state->count; index += std::uint64_t(gridDim.x) * blockDim.x)
    {
        const auto task = tasks.tasks[index];
        const float weight = __uint_as_float(task.payload[1]);

        if (weight == 0) continue;
        if (!record_resident_reduction<7>(view, reduction, task.voxel,
            __uint_as_float(task.payload[0]), weight, 0, 0, truncation)) return;
    }
}

__global__ void count_resident_task_rays(cuda_resident_view view, cuda_resident_task_view tasks)
{
    const auto ray = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (ray >= tasks.state->rays || atomicAdd(&view.state->error, 0U)) return;
    const auto accepted = tasks.accepted[ray];
    atomicAdd(accepted ? &view.state->batch_integrated : &view.state->batch_rejected, 1U);
    atomicAdd(reinterpret_cast<unsigned long long*>(&view.state->batch_contributions),
         static_cast<unsigned long long>(accepted));
}
}
