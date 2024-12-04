#include <covoxslam/sensor_grid.hpp>
#include <cmath>
#include <limits>

namespace covoxslam::sensor_grid
{
operation_result<geometry> make_geometry(const config& c)
{
    if (!std::isfinite(c.voxel_size) || c.voxel_size <= 0 ||
        !std::isfinite(c.radius) || c.radius <= 0 ||
        !std::isfinite(c.truncation) || c.truncation <= 0 ||
        !std::isfinite(c.minimum_range) || c.minimum_range < 0 ||
        !std::isfinite(c.maximum_weight) || c.maximum_weight <= 0 ||
        !std::isfinite(c.sparsity_compensation) || c.sparsity_compensation <= 0 ||
        (c.weight_dropoff && c.truncation <= c.voxel_size))
        return std::unexpected(operation_error{"invalid sensor grid configuration"});
    const double half = std::ceil(double(c.radius) / c.voxel_size);
    constexpr auto capacity = std::numeric_limits<std::uint32_t>::max();

    if (half < 1 || half > capacity / 2U)
        return std::unexpected(operation_error{"sensor grid side exceeds capacity"});
     const auto side = static_cast<std::uint64_t>(half) * 2U;
    if (side > capacity / side || side * side > capacity / side)
        return std::unexpected(operation_error{"sensor grid volume exceeds capacity"});
    const double extent = half * c.voxel_size;

    if (!std::isfinite(extent) || extent > std::numeric_limits<float>::max())
        return std::unexpected(operation_error{"sensor grid origin exceeds capacity"});
    return geometry{static_cast<std::uint32_t>(half), static_cast<std::uint32_t>(side),
        static_cast<std::uint32_t>(side * side * side), static_cast<float>(-extent)};
}
}
