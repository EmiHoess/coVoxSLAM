#pragma once
#include <cstdint>

namespace covoxslam::detail
{
struct cuda_resident_hash_entry
{
    std::uint64_t key{UINT64_MAX};
    std::uint32_t slot{UINT32_MAX}, reserved{};
};
}
