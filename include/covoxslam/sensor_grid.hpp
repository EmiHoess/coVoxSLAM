#pragma once

#include <covoxslam/core/result.hpp>
#include <covoxslam/sensor_grid_types.h>
#include <cstdint>
#include <span>
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
}
