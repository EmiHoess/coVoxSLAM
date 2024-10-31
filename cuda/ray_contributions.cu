#include "sparse_mapping.h"

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
template<bool emit>
__device__ std::uint32_t walk_ray(const cuda_device_ray& ray, cuda_ray_parameters config,
    cuda_ray_contribution* output, std::uint64_t offset, std::uint32_t expected,
    std::uint64_t capacity, std::uint32_t* status)
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
            atomicOr(status, 1U);
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
    const auto maximum_steps = static_cast<std::uint32_t>(ceil((end - start) / config.voxel_size) * 3) + 16;
    std::uint32_t count{};
    double entered{start};
    bool complete{};
    for (std::uint32_t iteration = 0; iteration < maximum_steps; ++iteration)
    {
        const double boundary = fmin(next[0], fmin(next[1], next[2]));
        float projected{};
        for (int axis = 0; axis < 3; ++axis) {
            const float center = static_cast<float>((double(voxel[axis]) + 0.5) * config.voxel_size);
            if (!isfinite(center)) {
                atomicOr(status, 1U);
                return count;
            }
             projected += (center - ray.origin[axis]) * direction[axis];
        }
        if (!isfinite(projected)) {
            atomicOr(status, 1U);
            return count;
        }
        if (boundary > entered && projected >= start && projected <= end)
        {
            if constexpr (emit) {
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
                atomicOr(status, 1U);
                return count;
            }
            voxel[axis] = static_cast<int>(candidate);
            next[axis] += delta[axis];
        }
        entered = boundary;
    }
    if (!complete) atomicOr(status, 2U);
    if constexpr (emit) if (count != expected) atomicOr(status, 4U);
    return count;
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
