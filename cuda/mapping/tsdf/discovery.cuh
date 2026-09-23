#pragma once
#include "fusion.cuh"
#include <covoxslam/mapping/config.hpp>

namespace covoxslam::sensor_grid
{
enum discovery_error : std::uint32_t { discovery_coordinate = 1, discovery_capacity = 2, discovery_hash = 4, discovery_pool = 8, discovery_ownership = 16 };
struct discovery_state
{
    std::uint32_t count{}, error{}, tile_count{}, new_count{};
};
struct discovery_view {
    std::uint64_t* hash{}, *keys{};
    discovery_state* state{};
    std::uint32_t capacity{}, hash_capacity{};
};
inline std::uint32_t discovery_hash_size(std::uint32_t capacity) noexcept
{
    if (!capacity || capacity > maximum_mapping_tiles) return 0;
    std::uint32_t size{2};

    while (size < capacity * 2) size *= 2;
    return size;
}
int cuda_sensor_discover(field_view source, const std::uint8_t* observed,
    detail::cuda_projective_pose transform, discovery_view view, void* stream);
}
