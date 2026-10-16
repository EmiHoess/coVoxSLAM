#include <covoxslam/voxel/tile_geometry.hpp>

#include <cstdint>
#include <limits>

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

int3 block_index(int3 voxel) noexcept
{
     return {floor_divide(voxel.x), floor_divide(voxel.y), floor_divide(voxel.z)};
}

operation_result<int3> global_index(int3 block, std::size_t local)
{
    if (local >= tile_voxels)
        return std::unexpected(operation_error{"tile voxel index out of bounds"});
    const auto coordinate = [](int value, int offset) -> operation_result<int> {
        const auto result = std::int64_t(value) * tile_edge + offset;
        if (result < std::numeric_limits<int>::min() || result > std::numeric_limits<int>::max())
            return std::unexpected(operation_error{"block coordinate exceeds voxel range"});
        return static_cast<int>(result);
    };
    auto x = coordinate(block.x, static_cast<int>(local % tile_edge));
    if (!x) return std::unexpected(x.error());
    auto y = coordinate(block.y, static_cast<int>((local / tile_edge) % tile_edge));
    if (!y) return std::unexpected(y.error());
    auto z = coordinate(block.z, static_cast<int>(local / (tile_edge * tile_edge)));
    if (!z) return std::unexpected(z.error());

     return int3{*x, *y, *z};
}

vec3f voxel_center(int3 index, float size) noexcept
{
    return {static_cast<float>(size * (double(index.x) + 0.5)),
            static_cast<float>(size * (double(index.y) + 0.5)),
            static_cast<float>(size * (double(index.z) + 0.5))};
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
}
