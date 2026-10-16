#pragma once

#include <cstddef>
#ifndef __CUDACC__
#include <covoxslam/core/result.hpp>
#include <limits>
#endif

namespace covoxslam
{
inline constexpr std::size_t bytes_per_kibibyte = std::size_t{1} << 10;
inline constexpr std::size_t bytes_per_mebibyte = std::size_t{1} << 20;
inline constexpr std::size_t bytes_per_gibibyte = std::size_t{1} << 30;

enum class byte_unit : std::size_t
{
    byte = 1,
    kibibyte = bytes_per_kibibyte,
    mebibyte = bytes_per_mebibyte,
    gibibyte = bytes_per_gibibyte
};

constexpr std::size_t byte_unit_width(byte_unit unit) noexcept
{
    switch (unit) {
    case byte_unit::byte: return 1;
    case byte_unit::kibibyte: return bytes_per_kibibyte;
    case byte_unit::mebibyte: return bytes_per_mebibyte;
    case byte_unit::gibibyte: return bytes_per_gibibyte;
    }
    return 0;
}

#ifndef __CUDACC__
inline operation_result<std::size_t> byte_count(std::size_t count, byte_unit unit)
{
    const auto width = byte_unit_width(unit);
    if (!width || count > std::numeric_limits<std::size_t>::max() / width)
        return std::unexpected(operation_error{"byte count overflows or has an invalid unit"});
    return count * width;
}
#endif
}
