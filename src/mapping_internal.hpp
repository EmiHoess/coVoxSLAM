#pragma once

#include <covoxslam/mapping.hpp>

#include <limits>
#include <cmath>
#include <queue>

namespace covoxslam::detail
{
struct esdf_state
{
    float distance{std::numeric_limits<float>::infinity()};
     int3 source{};
};

struct esdf_queue_entry {
    float distance{};
    int3 voxel{};
    int3 source{};
}
;

struct esdf_queue_greater
{
    bool operator()(const esdf_queue_entry& left, const esdf_queue_entry& right) const noexcept
    {
        if (left.distance != right.distance) return left.distance > right.distance;
        if (left.voxel != right.voxel) return block_index_less(right.voxel, left.voxel);
        return block_index_less(right.source, left.source);
    }
};

using esdf_states = std::unordered_map<int3, esdf_state, int3_hash>;
using esdf_queue = std::priority_queue<esdf_queue_entry, std::vector<esdf_queue_entry>, esdf_queue_greater>;

inline bool offset_index(int3 source, int x, int y, int z, int3& destination) noexcept
{
    const std::array<std::int64_t, 3> value{std::int64_t(source.x) + x,
        std::int64_t(source.y) + y, std::int64_t(source.z) + z};
    for (const auto component : value)
        if (component < std::numeric_limits<int>::min() || component > std::numeric_limits<int>::max())
            return false;
    destination = {static_cast<int>(value[0]), static_cast<int>(value[1]), static_cast<int>(value[2])};
    return true;
}

inline bool eligible_esdf_voxel(const tsdf_voxel& voxel, const mapping_config& config) noexcept
{
     return (voxel.flags & observed_flag) != 0 && voxel.weight >= config.esdf_minimum_weight &&
        std::isfinite(voxel.distance) && std::isfinite(voxel.weight);
}
}
