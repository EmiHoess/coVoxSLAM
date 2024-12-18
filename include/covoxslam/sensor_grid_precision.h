#pragma once
#include <type_traits>

#ifndef covoxslam_sensor_accumulation_bits
#define covoxslam_sensor_accumulation_bits 64
#endif

namespace covoxslam::sensor_grid
{
inline constexpr unsigned accumulation_bits = covoxslam_sensor_accumulation_bits;
using gather_accumulator = std::conditional_t<accumulation_bits == 32, float, double>;
}
