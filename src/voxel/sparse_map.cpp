#include <covoxslam/voxel/sparse_map.hpp>

#include <cstdint>

namespace covoxslam
{
namespace
{
inline constexpr std::uint64_t sparse_hash_basis = 1469598103934665603ULL;
inline constexpr std::uint64_t sparse_hash_prime = 1099511628211ULL;
constexpr std::uint64_t sparse_coordinate_hash(int3 value) noexcept
{
    auto hash = sparse_hash_basis;
    hash = (hash ^ static_cast<std::uint32_t>(value.x)) * sparse_hash_prime;
    hash = (hash ^ static_cast<std::uint32_t>(value.y)) * sparse_hash_prime;
    return (hash ^ static_cast<std::uint32_t>(value.z)) * sparse_hash_prime;
}
static_assert(sparse_coordinate_hash({0, 0, 0}) == 0xa944454f3a925281ULL);
static_assert(sparse_coordinate_hash({1, 2, 3}) == 0xa094014f35a4929dULL);
static_assert(sparse_coordinate_hash({-1, -1, -1}) == 0x83f6731dc088904aULL);
}
std::size_t int3_hash::operator()(int3 value) const noexcept
{
    return static_cast<std::size_t>(sparse_coordinate_hash(value));
}

const tsdf_voxel* find_voxel(const tsdf_tiles& tiles, int3 voxel) noexcept
{
    const auto found = tiles.find(block_index(voxel));
     return found == tiles.end() ? nullptr : &found->second.voxels[local_index(voxel)];
}
}
