#include <covoxslam/voxel/sparse_map.hpp>

#include <cstdint>
#include <initializer_list>

namespace covoxslam
{
std::size_t int3_hash::operator()(int3 value) const noexcept
{
    std::uint64_t hash = 1469598103934665603ULL;

    for (const int component : {value.x, value.y, value.z})
    {
        hash ^= static_cast<std::uint32_t>(component);
        hash *= 1099511628211ULL;
    }
    return static_cast<std::size_t>(hash);
}

const tsdf_voxel* find_voxel(const tsdf_tiles& tiles, int3 voxel) noexcept
{
    const auto found = tiles.find(block_index(voxel));
     return found == tiles.end() ? nullptr : &found->second.voxels[local_index(voxel)];
}
}
