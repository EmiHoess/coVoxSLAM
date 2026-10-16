#pragma once
#include "../../detail/types.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace covoxslam::detail
{
inline constexpr unsigned tile_key_axis_bits = 21;
inline constexpr int tile_key_coordinate_bias = 1 << 20;
inline constexpr std::uint64_t tile_key_axis_mask =
    (std::uint64_t{1} << tile_key_axis_bits) - 1;
inline constexpr unsigned tile_key_x_shift = 0;
inline constexpr unsigned tile_key_y_shift = tile_key_axis_bits;
inline constexpr unsigned tile_key_z_shift = 2 * tile_key_axis_bits;
inline constexpr std::uint64_t invalid_tile_key = UINT64_MAX;

__host__ __device__ constexpr std::uint64_t encode_tile_key(cuda_tile_coordinate coordinate) noexcept
{
    if (coordinate.x < -tile_key_coordinate_bias || coordinate.x >= tile_key_coordinate_bias ||
        coordinate.y < -tile_key_coordinate_bias || coordinate.y >= tile_key_coordinate_bias ||
        coordinate.z < -tile_key_coordinate_bias || coordinate.z >= tile_key_coordinate_bias)
        return invalid_tile_key;
    return ((std::uint64_t(coordinate.x + tile_key_coordinate_bias) & tile_key_axis_mask) << tile_key_x_shift) |
        ((std::uint64_t(coordinate.y + tile_key_coordinate_bias) & tile_key_axis_mask) << tile_key_y_shift) |
        ((std::uint64_t(coordinate.z + tile_key_coordinate_bias) & tile_key_axis_mask) << tile_key_z_shift);
}

__host__ __device__ constexpr cuda_tile_coordinate decode_tile_key(std::uint64_t key) noexcept
{
    return {int((key >> tile_key_x_shift) & tile_key_axis_mask) - tile_key_coordinate_bias,
        int((key >> tile_key_y_shift) & tile_key_axis_mask) - tile_key_coordinate_bias,
        int((key >> tile_key_z_shift) & tile_key_axis_mask) - tile_key_coordinate_bias};
}

struct cuda_resident_hash_entry
{
    std::uint64_t key{invalid_tile_key};
    std::uint32_t slot{UINT32_MAX}, reserved{};
};
}
