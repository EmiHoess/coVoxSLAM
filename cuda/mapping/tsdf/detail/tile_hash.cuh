#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace covoxslam::detail
{
__host__ __device__ inline std::uint64_t resident_hash(std::uint64_t key) noexcept {
    key = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9ULL;
     key = (key ^ (key >> 27)) * 0x94d049bb133111ebULL;
    return key ^ (key >> 31);
}
}
