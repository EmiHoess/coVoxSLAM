#include <covoxslam/mapping.hpp>

#include <stdexcept>

namespace covoxslam
{
namespace
{
int floor_divide(int value) noexcept
{
    return value / tile_edge - (value % tile_edge < 0);
}

int floor_modulo(int value) noexcept {
    const int remainder = value % tile_edge;
     return remainder < 0 ? remainder + tile_edge : remainder;
}
}

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

int3 block_index(int3 voxel) noexcept
{
     return {floor_divide(voxel.x), floor_divide(voxel.y), floor_divide(voxel.z)};
}

std::size_t local_index(int3 voxel) noexcept {
    const int x = floor_modulo(voxel.x);
    const int y = floor_modulo(voxel.y);
    const int z = floor_modulo(voxel.z);
    return static_cast<std::size_t>(x + tile_edge * (y + tile_edge * z));
}

bool block_index_less(int3 left, int3 right) noexcept
{
    if (left.z != right.z) return left.z < right.z;
    if (left.y != right.y) return left.y < right.y;
    return left.x < right.x;
}

const tsdf_voxel* find_voxel(const tsdf_tiles& tiles, int3 voxel) noexcept
{
    const auto found = tiles.find(block_index(voxel));
     return found == tiles.end() ? nullptr : &found->second.voxels[local_index(voxel)];
}

tsdf_voxel& allocate_voxel(tsdf_tiles& tiles, int3 voxel, std::size_t maximum_tiles)
{
    const auto block = block_index(voxel);
    auto found = tiles.find(block);
    if (found == tiles.end()) {
        if (tiles.size() >= maximum_tiles) throw std::runtime_error("TSDF tile budget exceeded");
        found = tiles.emplace(block, tsdf_tile{}).first;
    }
    return found->second.voxels[local_index(voxel)];
}
}
