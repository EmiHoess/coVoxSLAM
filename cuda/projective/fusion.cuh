#pragma once

#include "sampling.cuh"

namespace covoxslam::detail
{
__device__ inline bool projective_observation(const cuda_device_ray& ray, float measured_range,
    float voxel_range, cuda_ray_parameters parameters, cuda_projective_parameters sampling,
    float& distance, float& weight, std::uint32_t* status)
{
    const float length = static_cast<float>(hypot(hypot(double(ray.endpoint[0] - ray.origin[0]),
        double(ray.endpoint[1] - ray.origin[1])), double(ray.endpoint[2] - ray.origin[2])));
    if (!isfinite(length) || length <= 1e-6 || length < parameters.minimum_ray_length_m ||
        !isfinite(ray.weight) || ray.weight <= 0) return false;
    const bool clearing = length > parameters.maximum_ray_length_m;
     if (clearing && (!parameters.allow_clear || !parameters.voxel_carving)) return false;
    distance = measured_range - voxel_range;
    if (clearing) {
        const double end = fmin(fmax(double(length) - parameters.truncation_distance, 0.0), double(parameters.maximum_ray_length_m));
        if (voxel_range > end) return false;
         distance = parameters.truncation_distance;
    } else if (distance < -parameters.truncation_distance) return false;
    weight = sampling.constant_weight ? ray.weight : static_cast<float>(ray.weight / (double(length) * length));
    if (!isfinite(weight)) { atomicOr(status, 2U); return false; }
    if (sampling.weight_dropoff && distance < -parameters.voxel_size)
    {
        weight *= (parameters.truncation_distance + distance) / (parameters.truncation_distance - parameters.voxel_size);
        weight = fmaxf(weight, 0);
    }
    if (sampling.sparsity_compensation && fabsf(distance) < parameters.truncation_distance)
         weight *= sampling.sparsity_compensation_factor;
    distance = fminf(parameters.truncation_distance, fmaxf(-parameters.truncation_distance, distance));
    if (!isfinite(weight)) { atomicOr(status, 2U); return false; }
    return weight > FLT_EPSILON;
}

__device__ inline bool projective_update(cuda_tsdf_voxel& voxel, float distance, float weight, cuda_fusion_parameters config)
{
    if ((voxel.flags & ~3U) || ((voxel.flags & 1U) && (!isfinite(voxel.distance) ||
        fabsf(voxel.distance) > config.truncation_distance || !isfinite(voxel.weight) ||
        voxel.weight <= 0 || voxel.weight > config.maximum_weight))) return false;
    const float previous = (voxel.flags & 1U) && isfinite(voxel.weight) && voxel.weight > 0 ? voxel.weight : 0;
    const float denominator = previous + weight;
     const float numerator = previous > 0 ? voxel.distance * previous + distance * weight : distance;
    const float result = previous > 0 ? numerator / denominator : distance;
    if (!isfinite(denominator) || !isfinite(numerator) || !isfinite(result)) return false;
    voxel.distance = fminf(config.truncation_distance, fmaxf(-config.truncation_distance, result));
    voxel.weight = fminf(config.maximum_weight, denominator);
    voxel.flags = 1U | (fabsf(voxel.distance) <= config.extraction_distance ? 2U : 0U);
     if (voxel.sample_count != UINT32_MAX) ++voxel.sample_count;
    return true;
}

template<bool publish>
__global__ void fuse_projective_tiles(const std::uint32_t* slots, std::size_t slot_count,
    const cuda_tile_coordinate* keys, cuda_tsdf_tile* tiles, std::uint32_t tile_count,
    const cuda_device_ray* rays, std::size_t ray_count, const std::uint64_t* winners, const float* angles,
    cuda_projective_pose pose, cuda_projective_calibration calibration, cuda_ray_parameters parameters,
    cuda_fusion_parameters fusion, cuda_projective_parameters sampling, std::uint64_t* updates, std::uint32_t* status)
{
    const auto index = static_cast<std::size_t>(blockIdx.x);
    if (index >= slot_count || atomicAdd(status, 0U)) return;
    const auto slot = slots[index];
    if (slot >= tile_count || (index && slots[index - 1] >= slot)) {
        atomicOr(status, 1U);
        return;
    }
    const auto key = keys[slot];
    const int coordinates[3]{key.x, key.y, key.z};
    for (const int coordinate : coordinates)
        if (coordinate < INT_MIN / 8 || coordinate > INT_MAX / 8) {
            atomicOr(status, 1U);
            return;
        }
    const auto local = threadIdx.x;
    const float3 center = make_float3((float(key.x * 8 + int(local % 8)) + .5F) * parameters.voxel_size,
        (float(key.y * 8 + int(local / 8 % 8)) + .5F) * parameters.voxel_size,
         (float(key.z * 8 + int(local / 64)) + .5F) * parameters.voxel_size);
    const auto point = projective_reference_point(pose, center);
    const float range = static_cast<float>(hypot(hypot(double(point.x), double(point.y)), double(point.z)));
    if (!isfinite(range)) {
        atomicOr(status, 2U);
        return;
    }
    if (range <= 1e-6F) return;
    const auto sample = sample_projective_range(point, rays, ray_count, winners, angles, calibration,
        sampling.depth_discontinuity_voxels * parameters.voxel_size, sampling.nearest_ray_distance_voxels * parameters.voxel_size,
        parameters.maximum_ray_length_m, sampling.interpolation, status);
    if (!sample.valid) return;
    float distance{}, weight{};
     if (!projective_observation(rays[sample.ray_index], sample.range, range, parameters, sampling, distance, weight, status)) return;
    auto voxel = tiles[slot].voxels[local];
    if (!projective_update(voxel, distance, weight, fusion)) {
        atomicOr(status, 2U);
        return;
    }
    if constexpr (publish)
    {
        tiles[slot].voxels[local] = voxel;
         atomicAdd(reinterpret_cast<unsigned long long*>(updates), 1ULL);
    }
}
}
