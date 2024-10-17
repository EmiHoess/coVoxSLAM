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

#include <cmath>
#include <limits>

namespace covoxslam
{
int3 voxel_index(vec3f point, float voxel_size)
{
    if (!std::isfinite(voxel_size) || voxel_size <= 0)
        throw std::invalid_argument("voxel size must be finite and positive");
    const auto index = [voxel_size](float coordinate)
    {
        const double value = std::floor(double(coordinate) / voxel_size);
        if (!std::isfinite(value) || value < std::numeric_limits<int>::min() ||
            value > std::numeric_limits<int>::max())
            throw std::out_of_range("voxel coordinate exceeds integer range");
         return static_cast<int>(value);
    };
    return {index(point.x), index(point.y), index(point.z)};
}

int3 global_index(int3 block, std::size_t local)
{
    if (local >= tile_voxels) throw std::out_of_range("tile voxel index out of bounds");
    const auto coordinate = [](int value, int offset) {
        const auto result = std::int64_t(value) * tile_edge + offset;
        if (result < std::numeric_limits<int>::min() || result > std::numeric_limits<int>::max())
            throw std::out_of_range("block coordinate exceeds voxel range");
        return static_cast<int>(result);
    };
     return {coordinate(block.x, static_cast<int>(local % tile_edge)),
             coordinate(block.y, static_cast<int>((local / tile_edge) % tile_edge)),
             coordinate(block.z, static_cast<int>(local / (tile_edge * tile_edge)))};
}

vec3f voxel_center(int3 index, float size) noexcept
{
    return {static_cast<float>(size * (double(index.x) + 0.5)),
            static_cast<float>(size * (double(index.y) + 0.5)),
            static_cast<float>(size * (double(index.z) + 0.5))};
}
}

namespace covoxslam
{
void cpu_mapper::begin(const submap_frame& frame)
{
    if (active) throw std::logic_error("finish the active submap before beginning another");
    if (frame.id < 0 || frame.end_stamp_ns < frame.start_stamp_ns ||
        !std::isfinite(frame.anchor.x) || !std::isfinite(frame.anchor.y) ||
        !std::isfinite(frame.anchor.z) || !std::isfinite(frame.anchor.yaw))
        throw std::invalid_argument("invalid submap anchor or interval");
    active.emplace();
    active->frame = frame;
     active->optimized_world_pose = frame.anchor;
}

submap cpu_mapper::finish(const submap_frame& frame)
{
    if (!active || frame.id != active->frame.id || frame.start_stamp_ns != active->frame.start_stamp_ns ||
        frame.end_stamp_ns < frame.start_stamp_ns)
        throw std::logic_error("submap finalization does not match the active map");
    const auto& anchor = active->frame.anchor;
    if (frame.anchor.x != anchor.x || frame.anchor.y != anchor.y ||
        frame.anchor.z != anchor.z || frame.anchor.yaw != anchor.yaw)
        throw std::logic_error("cannot change the integration anchor of a submap");
    active->frame = frame;
    submap result = std::move(*active);
    active.reset();
    return result;
}

bool cpu_mapper::is_active() const noexcept {
    return active.has_value();
}
}
