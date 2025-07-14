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

}
}
