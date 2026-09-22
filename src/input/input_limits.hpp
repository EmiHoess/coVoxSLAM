#pragma once

#include <covoxslam/core/byte_units.hpp>
#include <cstddef>

namespace covoxslam::detail
{
inline constexpr std::size_t maximum_metadata_bytes = 16U * bytes_per_mebibyte;
inline constexpr unsigned maximum_json_nesting = 64;
inline constexpr unsigned maximum_toml_nesting = 64;
inline constexpr std::size_t maximum_toml_key_components = 64;
}
