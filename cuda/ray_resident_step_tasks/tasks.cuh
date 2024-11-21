#pragma once

#include "types.h"
#include <cuda_runtime.h>

namespace covoxslam::detail
{
__device__ inline bool record_resident_task(cuda_resident_view view, cuda_resident_task_view tasks,
    const int* voxel, std::uint32_t ray, std::uint32_t step)
{
    const auto offset = tasks.offsets[ray];

    if (step >= tasks.counts[ray] || offset > tasks.capacity || step >= tasks.capacity - offset) {
        atomicOr(&view.state->error, resident_contribution_capacity_error);
        return false;
    }
    tasks.tasks[offset + step] = {{voxel[0], voxel[1], voxel[2]}, {ray, 0}};
    return true;
}

__device__ inline bool evaluate_resident_task(const cuda_device_ray& ray,
    cuda_resident_prepared_ray prepared, const int* voxel, cuda_ray_parameters config,
    float& distance, bool& numeric_error)
{
    float projected{};

    for (int axis = 0; axis < 3; ++axis)
    {
        const float center = static_cast<float>((double(voxel[axis]) + 0.5) * config.voxel_size);

        if (!isfinite(center)) { numeric_error = true; return false; }
         projected += (center - ray.origin[axis]) * prepared.direction[axis];
    }

    if (!isfinite(projected)) { numeric_error = true; return false; }
    if (projected < prepared.start || projected > prepared.end) return false;
    distance = prepared.distance - projected;
    return true;
}
}
