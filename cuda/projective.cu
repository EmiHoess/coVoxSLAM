#include "sparse_mapping.h"
#include "projective/geometry.cuh"
#include "projective/policy_sampling.cuh"

#include <cmath>
#include <limits>

namespace covoxslam::detail
{
bool valid_projective_shape(cuda_projective_calibration calibration) noexcept
{
    const auto maximum = static_cast<std::uint32_t>(std::numeric_limits<int>::max());
    return calibration.rings && calibration.rings <= maximum && calibration.columns &&
        calibration.columns <= maximum && std::isfinite(calibration.azimuth_start_rad) &&
        std::isfinite(calibration.azimuth_coverage_rad) && calibration.azimuth_coverage_rad > 0 &&
        calibration.azimuth_coverage_rad <= 2 * CUDART_PI_F &&
         std::size_t(calibration.rings) <= std::numeric_limits<std::size_t>::max() / 8 / calibration.columns;
}

bool valid_projective_pose(cuda_projective_pose pose) noexcept {
    double squared{};
    for (const auto value : pose.translation) if (!std::isfinite(value)) return false;
    for (const auto value : pose.rotation_xyzw)
    {
        if (!std::isfinite(value)) return false;
        squared += double(value) * value;
    }
    return std::abs(squared - 1) <= 1e-5;
}

bool valid_projective_calibration(cuda_projective_calibration calibration,
    const float* vertical_angles, std::size_t angle_count) noexcept
{
    if (!valid_projective_shape(calibration) || !vertical_angles || angle_count != calibration.rings) return false;
    for (std::size_t index = 0; index < angle_count; ++index) {
        const float value = vertical_angles[index];
        if (!std::isfinite(value) || std::abs(value) > CUDART_PI_F / 2) return false;
        for (std::size_t previous = 0; previous < index; ++previous)
             if (vertical_angles[previous] == value) return false;
    }
    return true;
}
}

#include "projective/image.cuh"

namespace covoxslam::detail
{
int cuda_build_projective_range_image(const cuda_device_ray* rays, std::size_t count,
    cuda_ray_parameters parameters, cuda_projective_pose pose, cuda_projective_calibration calibration,
    const float* vertical_angles, std::uint64_t* winners, std::size_t pixel_capacity,
    std::uint32_t* status, void* stream) noexcept
{
    if (!valid_projective_shape(calibration) || !valid_projective_pose(pose) || !vertical_angles || !winners || !status ||
        pixel_capacity < std::size_t(calibration.rings) * calibration.columns ||
        count > std::numeric_limits<std::uint32_t>::max() || (count && !rays) ||
        !std::isfinite(parameters.minimum_ray_length_m) || parameters.minimum_ray_length_m < 0 ||
        !std::isfinite(parameters.maximum_ray_length_m) || parameters.maximum_ray_length_m < parameters.minimum_ray_length_m)
        return static_cast<int>(cudaErrorInvalidValue);
    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    auto error = cudaMemsetAsync(status, 0, sizeof(*status), cuda_stream);
    if (error != cudaSuccess) return static_cast<int>(error);
    error = cudaMemsetAsync(winners, 0xFF, std::size_t(calibration.rings) * calibration.columns * 8, cuda_stream);
     if (error != cudaSuccess) return static_cast<int>(error);
    const auto ring_blocks = static_cast<unsigned>((std::size_t(calibration.rings) + 127) / 128);
    validate_projective_angles<<<ring_blocks, 128, 0, cuda_stream>>>(vertical_angles, calibration.rings, status);
    error = cudaGetLastError();
    if (error != cudaSuccess || !count) return static_cast<int>(error);
    const auto blocks = static_cast<unsigned>((count + 255) / 256);
    build_projective_range_image<<<blocks, 256, 0, cuda_stream>>>(
        rays, count, parameters, pose, calibration, vertical_angles, winners, status);
    return static_cast<int>(cudaGetLastError());
}
}

#include "projective/discovery.cuh"

namespace covoxslam::detail
{
namespace
{
bool valid_projective_discovery(std::size_t ray_count, std::size_t pixel_count,
    cuda_projective_pose pose, cuda_ray_parameters parameters)
{
    return valid_projective_pose(pose) && ray_count <= UINT32_MAX && pixel_count <= UINT32_MAX &&
        std::isfinite(parameters.voxel_size) && parameters.voxel_size > 0 &&
        std::isfinite(parameters.truncation_distance) && parameters.truncation_distance > 0 &&
        std::isfinite(parameters.minimum_ray_length_m) && parameters.minimum_ray_length_m >= 0 &&
        std::isfinite(parameters.maximum_ray_length_m) && parameters.maximum_ray_length_m > 0 &&
        parameters.maximum_ray_length_m >= parameters.minimum_ray_length_m &&
         (double(parameters.maximum_ray_length_m) + parameters.truncation_distance) / parameters.voxel_size < 10000000;
}

bool valid_projective_tile_policy(std::size_t pixels, cuda_projective_discovery_options options)
{
    if (!options.calibrated_rays) return options.factor == 1;
    return options.factor && valid_projective_shape(options.calibration) && options.angles &&
         pixels == std::size_t(options.calibration.rings) * options.calibration.columns;
}
}

int cuda_count_projective_tiles(const cuda_device_ray* rays, std::size_t ray_count,
    const std::uint64_t* winners, std::size_t pixel_count, cuda_projective_pose pose,
    cuda_ray_parameters parameters, std::uint32_t* counts, std::uint32_t* status, void* stream,
    cuda_projective_discovery_options options) noexcept
{
    if (!valid_projective_discovery(ray_count, pixel_count, pose, parameters) || !status ||
        !valid_projective_tile_policy(pixel_count, options) ||
        (ray_count && !rays) || (pixel_count && (!winners || !counts))) return static_cast<int>(cudaErrorInvalidValue);
    auto error = cudaMemsetAsync(status, 0, sizeof(*status), static_cast<cudaStream_t>(stream));
    if (error != cudaSuccess || !pixel_count) return static_cast<int>(error);
    if (options.calibrated_rays) {
        validate_projective_angles<<<static_cast<unsigned>((std::size_t(options.calibration.rings) + 127) / 128),
            128, 0, static_cast<cudaStream_t>(stream)>>>(options.angles, options.calibration.rings, status);
        error = cudaGetLastError();
        if (error != cudaSuccess) return static_cast<int>(error);
    }
    visit_projective_tiles<false><<<static_cast<unsigned>((pixel_count + 255) / 256), 256, 0,
        static_cast<cudaStream_t>(stream)>>>(rays, ray_count, winners, pixel_count, pose, parameters,
            nullptr, counts, nullptr, nullptr, 0, status, options);
    return static_cast<int>(cudaGetLastError());
}

int cuda_emit_projective_tiles(const cuda_device_ray* rays, std::size_t ray_count,
    const std::uint64_t* winners, std::size_t pixel_count, cuda_projective_pose pose,
    cuda_ray_parameters parameters, const std::uint32_t* counts, const std::uint64_t* offsets,
    cuda_tile_coordinate* keys, std::uint64_t capacity, std::uint32_t* status, void* stream,
    cuda_projective_discovery_options options) noexcept
{
    if (!valid_projective_discovery(ray_count, pixel_count, pose, parameters) || !status ||
        !valid_projective_tile_policy(pixel_count, options) ||
        capacity > std::numeric_limits<std::size_t>::max() / sizeof(cuda_tile_coordinate) ||
        (ray_count && !rays) || (pixel_count && (!winners || !counts || !offsets)) ||
        (capacity && !keys)) return static_cast<int>(cudaErrorInvalidValue);
    if (!pixel_count) return static_cast<int>(cudaSuccess);
    if (options.calibrated_rays) {
        validate_projective_angles<<<static_cast<unsigned>((std::size_t(options.calibration.rings) + 127) / 128),
            128, 0, static_cast<cudaStream_t>(stream)>>>(options.angles, options.calibration.rings, status);
        const auto error = cudaGetLastError();
        if (error != cudaSuccess) return static_cast<int>(error);
    }
    visit_projective_tiles<true><<<static_cast<unsigned>((pixel_count + 255) / 256), 256, 0,
        static_cast<cudaStream_t>(stream)>>>(rays, ray_count, winners, pixel_count, pose, parameters,
            counts, nullptr, offsets, keys, capacity, status, options);
    return static_cast<int>(cudaGetLastError());
}
}

#include "projective/sampling.cuh"

#include "projective/fusion.cuh"

namespace covoxslam::detail
{
int cuda_fuse_projective_tiles(const std::uint32_t* slots, std::size_t slot_count,
    const cuda_tile_coordinate* keys, cuda_tsdf_tile* tiles, std::uint32_t tile_count,
    const cuda_device_ray* rays, std::size_t ray_count, const std::uint64_t* winners, const float* angles,
    cuda_projective_pose pose, cuda_projective_calibration calibration, cuda_ray_parameters parameters,
    cuda_fusion_parameters fusion, cuda_projective_parameters sampling,
    std::uint64_t* updates, std::uint32_t* status, void* stream) noexcept
{
    const auto valid_nonnegative = [](float value) { return std::isfinite(value) && value >= 0; };
    if (!valid_projective_shape(calibration) || !valid_projective_discovery(ray_count, 0, pose, parameters) ||
        slot_count > INT_MAX || (slot_count && (!slots || !keys || !tiles)) || (ray_count && !rays) ||
        !winners || !angles || !updates || !status ||
        !std::isfinite(fusion.maximum_weight) || fusion.maximum_weight <= 0 ||
        !valid_nonnegative(fusion.extraction_distance) || fusion.extraction_distance > parameters.truncation_distance ||
        fusion.truncation_distance != parameters.truncation_distance ||
        !valid_nonnegative(sampling.depth_discontinuity_voxels) || !valid_nonnegative(sampling.nearest_ray_distance_voxels) ||
        !std::isfinite(sampling.depth_discontinuity_voxels * parameters.voxel_size) ||
        !std::isfinite(sampling.nearest_ray_distance_voxels * parameters.voxel_size) ||
        !std::isfinite(sampling.sparsity_compensation_factor) || sampling.sparsity_compensation_factor <= 0 ||
        (sampling.weight_dropoff && parameters.truncation_distance <= parameters.voxel_size) ||
        (sampling.interpolation != cuda_projective_interpolation::nearest &&
         sampling.interpolation != cuda_projective_interpolation::linear_nearest)) return static_cast<int>(cudaErrorInvalidValue);
    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    auto error = cudaMemsetAsync(updates, 0, sizeof(*updates), cuda_stream);
     if (error != cudaSuccess) return static_cast<int>(error);
    const auto blocks = static_cast<unsigned>((std::size_t(calibration.rings) + 127) / 128);
    validate_projective_angles<<<blocks, 128, 0, cuda_stream>>>(angles, calibration.rings, status);
    error = cudaGetLastError();
    if (error != cudaSuccess || !slot_count) return static_cast<int>(error);
    fuse_projective_tiles<false><<<static_cast<unsigned>(slot_count), 512, 0, cuda_stream>>>(slots, slot_count,
        keys, tiles, tile_count, rays, ray_count, winners, angles, pose, calibration, parameters, fusion, sampling, updates, status);
    error = cudaGetLastError();
    if (error != cudaSuccess) return static_cast<int>(error);
    fuse_projective_tiles<true><<<static_cast<unsigned>(slot_count), 512, 0, cuda_stream>>>(slots, slot_count,
        keys, tiles, tile_count, rays, ray_count, winners, angles, pose, calibration, parameters, fusion, sampling, updates, status);
    return static_cast<int>(cudaGetLastError());
}
}
