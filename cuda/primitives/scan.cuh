#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace covoxslam::detail
{
inline constexpr std::size_t scan_block_threads = 256;
constexpr std::size_t scan_block_count(std::size_t count) noexcept
{
    return count / scan_block_threads + std::size_t(count % scan_block_threads != 0);
}
int cuda_scan_counts(const std::uint32_t* counts, std::uint64_t* offsets,
                     std::uint64_t* block_sums, std::size_t count, void* stream) noexcept;
}
