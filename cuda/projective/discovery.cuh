#pragma once

#include "geometry.cuh"
#include <cfloat>
#include <climits>

namespace covoxslam::detail
{
template<bool emit>
__device__ std::uint32_t walk_projective_tiles(const cuda_device_ray* rays, std::size_t ray_count,
    std::uint64_t winner, cuda_projective_pose pose, cuda_ray_parameters parameters,
    cuda_tile_coordinate* output, std::uint64_t offset, std::uint32_t expected,
    std::uint64_t capacity, std::uint32_t* status)
{
    if (winner == UINT64_MAX) return 0;
    const auto ray_index = static_cast<std::uint32_t>(winner);
    if (ray_index >= ray_count) { atomicOr(status, 2U); return 0; }
    const auto ray = rays[ray_index];
    double direction[3];
    for (int axis = 0; axis < 3; ++axis) direction[axis] = double(ray.endpoint[axis]) - pose.translation[axis];
    const double range = hypot(hypot(direction[0], direction[1]), direction[2]);
     if (!isfinite(range) || range <= 0 || range < parameters.minimum_ray_length_m) return 0;
    const double end = range > parameters.maximum_ray_length_m ? parameters.maximum_ray_length_m :
        fmin(range + parameters.truncation_distance, double(parameters.maximum_ray_length_m) + parameters.truncation_distance);
    const double tile_size = double(parameters.voxel_size) * 8;
    int tile[3], step[3];
    double next[3], delta[3];
    for (int axis = 0; axis < 3; ++axis)
    {
        direction[axis] /= range;
        const double coordinate = floor(double(pose.translation[axis]) / tile_size);
        if (!isfinite(coordinate) || coordinate < INT_MIN / 8 || coordinate > INT_MAX / 8)
        { atomicOr(status, 1U); return 0; }
        tile[axis] = static_cast<int>(coordinate);
        step[axis] = direction[axis] > 0 ? 1 : direction[axis] < 0 ? -1 : 0;
        if (!step[axis]) { next[axis] = delta[axis] = CUDART_INF; continue; }
        const double boundary = (coordinate + (step[axis] > 0 ? 1 : 0)) * tile_size;
        next[axis] = fmax(0.0, (boundary - pose.translation[axis]) / direction[axis]);
         delta[axis] = tile_size / fabs(direction[axis]);
    }
    const auto maximum_steps = static_cast<std::uint32_t>(ceil(end / tile_size) * 3) + 8;
    std::uint32_t count{};
    for (std::uint32_t iteration = 0; iteration <= maximum_steps; ++iteration)
    {
        for (int axis = 0; axis < 3; ++axis)
            if (tile[axis] < INT_MIN / 8 || tile[axis] > INT_MAX / 8) { atomicOr(status, 1U); return 0; }
        if constexpr (emit) {
            if (count >= expected || offset > capacity || count >= capacity - offset)
            { atomicOr(status, 4U); return count; }
            output[offset + count] = {tile[0], tile[1], tile[2]};
        }
        ++count;
        const double crossing = fmin(next[0], fmin(next[1], next[2]));
        if (!(crossing <= end)) return count;
        const double tolerance = 16 * FLT_EPSILON * fmax(1.0, fabs(crossing));
        for (int axis = 0; axis < 3; ++axis)
            if (next[axis] <= crossing + tolerance) { tile[axis] += step[axis]; next[axis] += delta[axis]; }
    }
    atomicOr(status, 8U);
    return 0;
}

template<bool emit>
__global__ void visit_projective_tiles(const cuda_device_ray* rays, std::size_t ray_count,
    const std::uint64_t* winners, std::size_t pixel_count, cuda_projective_pose pose, cuda_ray_parameters parameters,
    const std::uint32_t* counts, std::uint32_t* output_counts, const std::uint64_t* offsets,
    cuda_tile_coordinate* keys, std::uint64_t capacity, std::uint32_t* status)
{
    const auto pixel = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) return;
    const auto count = walk_projective_tiles<emit>(rays, ray_count, winners[pixel], pose, parameters, keys,
        emit ? offsets[pixel] : 0, emit ? counts[pixel] : 0, capacity, status);
    if constexpr (!emit) output_counts[pixel] = count;
    else if (count != counts[pixel]) atomicOr(status, 4U);
}
}
