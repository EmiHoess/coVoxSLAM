#pragma once
#include <type_traits>
#include <cfloat>
#include <covoxslam/sensor/detail/types.hpp>

#ifndef covoxslam_sensor_accumulation_bits
#define covoxslam_sensor_accumulation_bits 64
#endif

namespace covoxslam::sensor_grid
{
inline constexpr unsigned accumulation_bits = covoxslam_sensor_accumulation_bits;
using gather_accumulator = std::conditional_t<accumulation_bits == 32, float, double>;
inline bool safe_accumulation(config settings, std::uint32_t maximum_entries)
{
    if constexpr (accumulation_bits == 64) return true;
    const double weight = settings.sparsity_compensation > 1 ? settings.sparsity_compensation : 1;
    const double total = double(maximum_entries) * weight;
    return total <= double(FLT_MAX) * .25 && total * settings.truncation <= double(FLT_MAX) * .25;
}
}
