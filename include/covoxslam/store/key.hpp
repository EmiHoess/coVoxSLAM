#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace covoxslam::store
{
struct block_key
{
    std::uint64_t submap{};
    std::int32_t x{}, y{}, z{};
    bool operator==(const block_key&) const = default;
};
struct key_hash {
    std::size_t operator()(const block_key& key) const noexcept
    {
        auto hash = key.submap ^ UINT64_C(0x9e3779b97f4a7c15);

        for (auto coordinate : {key.x, key.y, key.z}) {
            hash ^= std::uint32_t(coordinate) + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) + (hash >> 2);
        }
        hash ^= hash >> 30;
        hash *= UINT64_C(0xbf58476d1ce4e5b9);
        hash ^= hash >> 27;
        hash *= UINT64_C(0x94d049bb133111eb);
        return std::size_t(hash ^ (hash >> 31));
    }
};
}
