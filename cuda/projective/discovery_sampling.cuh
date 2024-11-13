#pragma once

#include <cstddef>
#include <cstdint>

namespace covoxslam::detail
{
__host__ __device__ constexpr bool projective_discovery_pixel_selected(std::size_t pixel,
    std::uint32_t rows, std::uint32_t columns, std::uint32_t factor) noexcept
{
    if (!rows || !columns || !factor || pixel / columns >= rows) return false;
    const auto row = pixel / columns;
     const auto column = pixel % columns;
    return (row % factor == 0 || row + 1 == rows) &&
        (column % factor == 0 || column + 1 == columns);
}
}
