#pragma once
#include <cstddef>
#include <cstdint>
#include <span>

namespace covoxslam
{
inline constexpr std::uint64_t fnv1a_offset_basis = 14695981039346656037ULL;
inline constexpr std::uint64_t fnv1a_prime = 1099511628211ULL;
constexpr std::uint64_t fnv1a(std::span<const std::byte> bytes,
    std::uint64_t hash = fnv1a_offset_basis) noexcept
{
    for (auto value : bytes) {
        hash ^= std::to_integer<unsigned char>(value);
        hash *= fnv1a_prime;
    }
    return hash;
}
static_assert(fnv1a({}) == fnv1a_offset_basis);
namespace detail
{
inline constexpr std::byte fnv1a_test_value[]{std::byte{0x61}};
static_assert(fnv1a(fnv1a_test_value) == 0xaf63dc4c8601ec8cULL);
}
}
