#pragma once

#include <covoxslam/core/result.hpp>
#include <covoxslam/sensor_grid.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace covoxslam::sensor_grid
{
using unit_direction = std::array<double, 3>;
inline operation_result<unit_direction> normalize(direction input)
{
    const double norm = std::hypot(double(input.x), double(input.y), double(input.z));

    if (!std::isfinite(norm) || norm <= 0)
        return std::unexpected(operation_error{"sensor ray must be finite and nonzero"});
    return unit_direction{input.x / norm, input.y / norm, input.z / norm};
}
template<class emit_type>
operation_status walk(const geometry& grid, const config& c,
                      const unit_direction& ray, emit_type&& emit)
{
    std::array<int, 3> cell{}, step{};
     std::array<double, 3> next{};
    const auto crossing = [&](unsigned axis) {
        return step[axis] == 0 ? std::numeric_limits<double>::infinity() :
            double(cell[axis] + (step[axis] > 0)) * c.voxel_size / ray[axis];
    };

    for (unsigned axis = 0; axis < 3; ++axis)
    {
        step[axis] = ray[axis] > 0 ? 1 : ray[axis] < 0 ? -1 : 0;
        cell[axis] = step[axis] < 0 ? -1 : 0;
        next[axis] = crossing(axis);
    }
    double entry = 0;
    const int half = static_cast<int>(grid.half_side);

    while (entry < c.radius)
    {
        std::array<std::uint32_t, 3> index{};
        double projected = 0;

        for (unsigned axis = 0; axis < 3; ++axis) {
            if (cell[axis] < -half || cell[axis] >= half) return {};
             index[axis] = static_cast<std::uint32_t>(cell[axis] + half);
            projected += (cell[axis] + 0.5) * c.voxel_size * ray[axis];
        }
        const auto voxel = index[0] + grid.side * (index[1] + grid.side * index[2]);

        if (!std::isfinite(static_cast<float>(projected)))
            return std::unexpected(operation_error{"sensor projection exceeds capacity"});
        const auto status = emit(voxel, static_cast<float>(entry), static_cast<float>(projected));
        if (!status) return status;
        const double distance = std::min({next[0], next[1], next[2]});

        if (distance >= c.radius) break;
        const double tolerance = 8 * std::numeric_limits<double>::epsilon() * distance;

        for (unsigned axis = 0; axis < 3; ++axis)
            if (next[axis] <= distance + tolerance)
            {
                cell[axis] += step[axis];
                 next[axis] = crossing(axis);
            }
        entry = distance;
    }
    return {};
}
}
