#pragma once

#include <covoxslam/core/result.hpp>
#include <covoxslam/dataset.hpp>
#include <covoxslam/sensor_grid.hpp>
#include <cmath>

namespace covoxslam::sensor_grid
{
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
