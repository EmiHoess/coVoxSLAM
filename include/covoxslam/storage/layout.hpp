#pragma once
#include <covoxslam/voxel/tile_shape.hpp>

namespace covoxslam::storage
{
enum class layout { pooled_aos = 0, tile_aosoa = 1, pool_soa = 2 };
static_assert(static_cast<int>(layout::pooled_aos) == 0 && static_cast<int>(layout::tile_aosoa) == 1 &&
    static_cast<int>(layout::pool_soa) == 2);
}
