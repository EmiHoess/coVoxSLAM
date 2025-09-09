#pragma once
#include "fusion.cuh"

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
    if (!capacity || capacity > 1048576) return 0;
    std::uint32_t size{2};

    while (size < capacity * 2) size *= 2;
    return size;
}
inline detail::cuda_tile_coordinate decode_discovery_key(std::uint64_t key) noexcept
{
    constexpr int bias = 1 << 20;
    return {int(key & 2097151) - bias, int((key >> 21) & 2097151) - bias, int(key >> 42) - bias};
}
int cuda_sensor_discover(field_view source, const std::uint8_t* observed,
    detail::cuda_projective_pose transform, discovery_view view, void* stream);
}
