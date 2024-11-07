#include "sparse_mapping.h"
#include "projective/geometry.cuh"

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
