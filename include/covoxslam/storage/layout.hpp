#pragma once
#include <covoxslam/voxel/tile_shape.hpp>
#include <cstdint>

namespace covoxslam::storage
{
inline constexpr std::size_t pool_soa_voxel_stride_bytes = sizeof(std::uint32_t);
inline constexpr std::size_t pool_soa_tile_stride_bytes =
    tile_voxels * pool_soa_voxel_stride_bytes;
inline constexpr std::size_t pool_soa_scalar_alignment = alignof(std::uint32_t);

enum class layout { pooled_aos = 0, tile_aosoa = 1, pool_soa = 2 };
static_assert(static_cast<int>(layout::pooled_aos) == 0 && static_cast<int>(layout::tile_aosoa) == 1 &&
    static_cast<int>(layout::pool_soa) == 2);
}
