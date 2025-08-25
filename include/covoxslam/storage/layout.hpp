#pragma once
#include <cstddef>

namespace covoxslam::storage
{
inline constexpr std::size_t voxels_per_tile = 512;
enum class layout { pooled_aos = 0, tile_aosoa = 1, pool_soa = 2 };
static_assert(static_cast<int>(layout::pooled_aos) == 0 && static_cast<int>(layout::tile_aosoa) == 1 &&
    static_cast<int>(layout::pool_soa) == 2);
}
