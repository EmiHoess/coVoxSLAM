#pragma once
#include <covoxslam/core/geometry.hpp>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace covoxslam { struct sensor {
    std::string model, acquisition_model;
    std::size_t rings{}, columns{};
    double scan_rate_hz{};
    std::uint64_t scan_period_ns{};
    double azimuth_start_rad{}, azimuth_coverage_rad{};
    bool counter_clockwise{true};
    
    std::vector<float> vertical_angles_rad;
    float min_range_m{}, max_range_m{};
    pose body_from_lidar{};
}; }
