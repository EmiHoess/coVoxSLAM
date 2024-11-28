#pragma once

#include "common.cuh"
#include "dda_types.h"
#include <math_constants.h>
#include <climits>

namespace covoxslam::detail
{
__device__ inline resident_warp_dda resident_warp_begin(
    cuda_resident_view view, cuda_device_ray ray, cuda_ray_parameters config)
{
    resident_warp_dda walk{};
    walk.ray = ray;

    for (int axis = 0; axis < 3; ++axis) {
        if (!isfinite(ray.origin[axis]) || !isfinite(ray.endpoint[axis])) return walk;
        walk.direction[axis] = ray.endpoint[axis] - ray.origin[axis];
    }
    walk.distance = static_cast<float>(hypot(hypot(double(walk.direction[0]), double(walk.direction[1])), double(walk.direction[2])));

    if (!isfinite(walk.distance) || walk.distance < config.minimum_ray_length_m ||
        !isfinite(ray.weight) || ray.weight <= 0) return walk;
    const bool clearing = walk.distance > config.maximum_ray_length_m;

    if (clearing && (!config.allow_clear || !config.voxel_carving)) return walk;
    const double surface_start = fmax(0.0, double(walk.distance) - config.truncation_distance);
    walk.start = config.voxel_carving ? 0.0 : surface_start;
    walk.end = clearing ? fmin(surface_start, double(config.maximum_ray_length_m)) : double(walk.distance) + config.truncation_distance;

    if (walk.end <= walk.start) return walk;
    walk.entered = walk.start;

    for (int axis = 0; axis < 3; ++axis)
    {
        walk.direction[axis] /= walk.distance;
        const double coordinate = floor((double(ray.origin[axis]) + walk.direction[axis] * walk.start) / config.voxel_size);

        if (!isfinite(coordinate) || coordinate < INT_MIN || coordinate > INT_MAX) {
            atomicOr(&view.state->error, resident_traversal_error);
            return walk;
        }
        walk.voxel[axis] = static_cast<int>(coordinate);
        walk.step[axis] = walk.direction[axis] > 0 ? 1 : walk.direction[axis] < 0 ? -1 : 0;

        if (!walk.step[axis]) { walk.next[axis] = walk.delta[axis] = CUDART_INF; continue; }
        const double boundary = config.voxel_size * (coordinate + (walk.step[axis] > 0 ? 1 : 0));
        walk.next[axis] = fmax(walk.start, (boundary - ray.origin[axis]) / walk.direction[axis]);
         walk.delta[axis] = double(config.voxel_size) / fabs(double(walk.direction[axis]));
    }
    walk.maximum_steps = static_cast<unsigned>(ceil((walk.end - walk.start) / config.voxel_size) * 3) + 16;
    walk.active = true;
    return walk;
}

__device__ inline bool resident_warp_sample(cuda_resident_view view, resident_warp_dda& walk,
    cuda_ray_parameters config, unsigned& voxel, double& weighted_distance, double& weight, float* raw_distance = nullptr)
{
    if (!walk.active) return false;
    const double boundary = fmin(walk.next[0], fmin(walk.next[1], walk.next[2]));
    float projected{};

    for (int axis = 0; axis < 3; ++axis) {
        const float center = static_cast<float>((double(walk.voxel[axis]) + 0.5) * config.voxel_size);

        if (!isfinite(center)) {
            walk.active = false;
            atomicOr(&view.state->error, resident_traversal_error);
            return false;
        }
        const float offset = __fsub_rn(center, walk.ray.origin[axis]);
        projected = axis ? __fmaf_rn(offset, walk.direction[axis], projected) : __fmul_rn(offset, walk.direction[axis]);
    }
    if (!isfinite(projected)) {
        walk.active = false;
        atomicOr(&view.state->error, resident_traversal_error);
        return false;
    }
    const bool accepted = boundary > walk.entered && projected >= walk.start && projected <= walk.end;

    if (accepted)
    {
        const cuda_tile_coordinate tile{walk.voxel[0] / 8 - (walk.voxel[0] % 8 < 0),
            walk.voxel[1] / 8 - (walk.voxel[1] % 8 < 0), walk.voxel[2] / 8 - (walk.voxel[2] % 8 < 0)};
        const auto slot = resident_find_tile(view, tile);

        if (slot >= view.tile_capacity) {
            walk.active = false;
            atomicOr(&view.state->error, resident_scatter_error);
            return false;
        }
        voxel = slot * 512 + (walk.voxel[0] - tile.x * 8) + 8 * (walk.voxel[1] - tile.y * 8) + 64 * (walk.voxel[2] - tile.z * 8);
        weight = walk.ray.weight;
        if (raw_distance) *raw_distance = walk.distance - projected;
        weighted_distance = weight * fminf(config.truncation_distance,
            fmaxf(-config.truncation_distance, walk.distance - projected));
        ++walk.contributions;
    }
    if (!isfinite(boundary) || boundary >= walk.end) { walk.active = false; return accepted; }

    for (int axis = 0; axis < 3; ++axis)
    {
        if (walk.next[axis] > boundary + 1.0e-6) continue;
        const auto candidate = static_cast<long long>(walk.voxel[axis]) + walk.step[axis];

        if (candidate < INT_MIN || candidate > INT_MAX) {
            walk.active = false;
            atomicOr(&view.state->error, resident_traversal_error);
            return false;
        }
        walk.voxel[axis] = static_cast<int>(candidate);
         walk.next[axis] += walk.delta[axis];
    }
    walk.entered = boundary;

    if (++walk.iteration >= walk.maximum_steps) {
        walk.active = false;
        atomicOr(&view.state->error, resident_traversal_error);
    }
    return accepted;
}
}
