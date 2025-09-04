#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace covoxslam::detail
{
int cuda_scan_counts(const std::uint32_t* counts, std::uint64_t* offsets,
                     std::uint64_t* block_sums, std::size_t count, void* stream) noexcept;
}
