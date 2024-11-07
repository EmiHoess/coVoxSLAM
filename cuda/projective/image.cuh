#pragma once

#include "geometry.cuh"

namespace covoxslam::detail
{
__global__ void validate_projective_angles(const float* angles, std::uint32_t count, std::uint32_t* status)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float value = angles[index];
    bool valid = isfinite(value) && fabsf(value) <= CUDART_PI_F / 2;
    for (std::size_t previous = 0; previous < index; ++previous)
         valid = valid && angles[previous] != value;
    if (!valid) atomicOr(status, 1U);
}

__global__ void build_projective_range_image(const cuda_device_ray* rays, std::size_t count,
    cuda_ray_parameters parameters, cuda_projective_pose pose, cuda_projective_calibration calibration,
    const float* angles, std::uint64_t* winners, std::uint32_t* status)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count || (atomicAdd(status, 0U) & 1U)) return;
    const auto ray = rays[index];
    if (!isfinite(ray.weight) || ray.weight <= 0) return;
    float direction[3];
    for (int axis = 0; axis < 3; ++axis) {
        if (!isfinite(ray.origin[axis]) || !isfinite(ray.endpoint[axis])) return;
        direction[axis] = ray.endpoint[axis] - ray.origin[axis];
    }
    const float distance = static_cast<float>(hypot(hypot(double(direction[0]), double(direction[1])), double(direction[2])));
    if (!isfinite(distance) || distance <= 0 || distance < parameters.minimum_ray_length_m) return;
    if (distance > parameters.maximum_ray_length_m && (!parameters.allow_clear || !parameters.voxel_carving)) return;
    const auto point = projective_reference_point(pose, {ray.endpoint[0], ray.endpoint[1], ray.endpoint[2]});
    const float range = static_cast<float>(hypot(hypot(double(point.x), double(point.y)), double(point.z)));
    if (!isfinite(range)) { atomicOr(status, 2U); return; }
    if (range <= 1e-6F) return;
    const float elevation = asinf(fminf(1, fmaxf(-1, point.z / range)));
    const int ring = projective_closest_ring(elevation, angles, calibration.rings);
     const int column = projective_azimuth_column(atan2f(point.y, point.x), calibration);
    if (ring < 0 || column < 0) return;
    const auto candidate = (std::uint64_t(__float_as_uint(range)) << 32U) | static_cast<std::uint32_t>(index);
    atomicMin(reinterpret_cast<unsigned long long*>(winners + std::size_t(ring) * calibration.columns + column),
        static_cast<unsigned long long>(candidate));
}
}
