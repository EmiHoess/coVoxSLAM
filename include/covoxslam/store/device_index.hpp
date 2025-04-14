#pragma once
#include <covoxslam/store/key.hpp>
#include <cstddef>
#include <cstdint>

namespace covoxslam::store
{
struct device_hash_entry
{
    block_key key;
    std::uint64_t reuse_generation{};
    std::uint32_t slot{}, state{};
};
struct device_index_view
{
    const device_hash_entry* entries{};
    std::size_t capacity{};
    std::uint64_t submap{}, allocation_generation{};
    std::uint32_t field{};
    int device{-1};
};
struct device_lookup
{
    std::uint32_t slot{}, found{};
    std::uint64_t reuse_generation{};
};
#if defined(__CUDACC__)
__device__ inline std::uint64_t device_key_hash(block_key key)
{
    auto hash = key.submap ^ UINT64_C(0x9e3779b97f4a7c15);
    const std::int32_t coordinates[]{key.x, key.y, key.z};

    for (auto coordinate : coordinates)
        hash ^= std::uint32_t(coordinate) + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) + (hash >> 2);
    hash ^= hash >> 30;
    hash *= UINT64_C(0xbf58476d1ce4e5b9);
    hash ^= hash >> 27;
    hash *= UINT64_C(0x94d049bb133111eb);
    return hash ^ (hash >> 31);
}
__device__ inline bool device_key_equal(block_key a, block_key b)
{
    return a.submap == b.submap && a.x == b.x && a.y == b.y && a.z == b.z;
}
__device__ inline device_lookup find_block(device_index_view view, block_key key)
{
    if (!view.entries || !view.capacity || key.submap != view.submap || !view.allocation_generation) return {};
    auto bucket = device_key_hash(key) & (view.capacity - 1);

    for (std::size_t probe = 0; probe < view.capacity; ++probe) {
        const auto& entry = view.entries[bucket];
        if (!entry.state) return {};
        if (entry.state == 1 && device_key_equal(entry.key, key)) return {entry.slot, 1, entry.reuse_generation};
        bucket = (bucket + 1) & (view.capacity - 1);
    }
    return {};
}
#endif
int build_device_index(device_hash_entry* entries, std::size_t capacity, const device_hash_entry* records,
    std::size_t count, unsigned* errors, void* stream) noexcept;
}
