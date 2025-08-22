#pragma once

#include <covoxslam/core/result.hpp>
#include <covoxslam/input/calibration.hpp>
#include <covoxslam/sensor_grid_types.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace covoxslam::sensor_grid
{
struct lookup
{
    config settings{};
    geometry grid{};
    std::uint32_t ray_count{}, selected_rays{}, reduction_factor{1};
     std::vector<std::uint32_t> offsets, active_voxels;
    std::vector<intersection> intersections;
};
operation_result<geometry> make_geometry(const config& settings);
struct lookup_limits
{
    std::uint32_t maximum_voxels{16777216}, maximum_rays{1048576};
     std::uint32_t maximum_intersections{67108864};
    std::uint64_t maximum_host_bytes{536870912};
};
static_assert(std::is_aggregate_v<lookup>);
static_assert(std::is_aggregate_v<lookup_limits>);
static_assert(std::is_standard_layout_v<lookup_limits>);
static_assert(std::is_trivially_copyable_v<lookup_limits>);
static_assert(sizeof(lookup_limits) == 24);
static_assert(offsetof(lookup_limits, maximum_voxels) == 0);
static_assert(offsetof(lookup_limits, maximum_rays) == 4);
static_assert(offsetof(lookup_limits, maximum_intersections) == 8);
static_assert(offsetof(lookup_limits, maximum_host_bytes) == 16);
static_assert(lookup_limits{}.maximum_voxels == 16777216);
static_assert(lookup_limits{}.maximum_rays == 1048576);
static_assert(lookup_limits{}.maximum_intersections == 67108864);
static_assert(lookup_limits{}.maximum_host_bytes == 536870912);
operation_result<lookup> build_lookup(const config& settings,
    std::span<const direction> directions, std::uint32_t reduction_factor = 1,
    lookup_limits limits = {});
operation_status validate_lookup(const lookup& table);
inline lookup_view host_view(const lookup& table)
{
    return {table.offsets.data(), table.intersections.data(), table.active_voxels.data(),
        static_cast<std::uint32_t>(table.active_voxels.size()), table.ray_count,
        static_cast<std::uint32_t>(table.intersections.size()), table.grid.voxels, table.settings};
}

inline operation_result<std::vector<direction>> calibrated_directions(const sensor& model)
{
    const float start = static_cast<float>(model.azimuth_start_rad);
    const float coverage = static_cast<float>(model.azimuth_coverage_rad);

    if (!std::isfinite(start) || !std::isfinite(coverage) || coverage <= 0 ||
        !model.rings || !model.columns || model.rings > 1048576 / model.columns ||
        model.vertical_angles_rad.size() != model.rings)
        return std::unexpected(operation_error{"invalid sensor calibration"});
    std::vector<direction> result;
    result.reserve(model.rings * model.columns);

    for (const auto elevation : model.vertical_angles_rad)
    {
        if (!std::isfinite(elevation))
            return std::unexpected(operation_error{"invalid sensor elevation"});
        const float horizontal = std::cos(elevation);

        for (std::size_t column = 0; column < model.columns; ++column) {
            const float relative = coverage * static_cast<float>(column) / static_cast<float>(model.columns);
             const float azimuth = model.counter_clockwise ? start + relative : start - relative;

            if (!std::isfinite(azimuth))
                return std::unexpected(operation_error{"sensor azimuth exceeds capacity"});
            result.push_back({horizontal * std::cos(azimuth), horizontal * std::sin(azimuth), std::sin(elevation)});
        }
    }
    return result;
}
}
