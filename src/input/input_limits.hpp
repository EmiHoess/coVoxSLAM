#pragma once

#include <covoxslam/core/byte_units.hpp>
#include <cstddef>

namespace covoxslam::detail
{
inline constexpr std::size_t maximum_metadata_bytes = 16U * bytes_per_mebibyte;
inline constexpr std::size_t maximum_frame_beams = std::size_t{1} << 24;
inline constexpr std::size_t maximum_calibration_beam_product = std::size_t{1} << 24;
inline constexpr std::size_t maximum_trajectory_samples = std::size_t{1} << 24;
inline constexpr std::size_t maximum_trajectory_line_bytes = bytes_per_mebibyte;
inline constexpr std::size_t maximum_input_cache_bytes = bytes_per_gibibyte;
inline constexpr unsigned maximum_json_nesting = 64;
inline constexpr unsigned maximum_toml_nesting = 64;
inline constexpr std::size_t maximum_toml_key_components = 64;
}
