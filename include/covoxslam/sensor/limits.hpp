#pragma once

#include <cstdint>

namespace covoxslam::sensor_grid
{
inline constexpr std::uint32_t maximum_lookup_voxels = 1U << 24;
inline constexpr std::uint32_t maximum_dense_voxels = 1U << 24;
inline constexpr std::uint32_t maximum_sensor_rays = 1U << 20;
inline constexpr std::uint32_t maximum_prepared_intersections = 1U << 26;
}
