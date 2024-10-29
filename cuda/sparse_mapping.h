#pragma once

#include <cstddef>
#include <cstdint>

namespace covoxslam::detail
{
inline constexpr std::uint32_t cuda_mapping_abi_version = 1;

struct cuda_device_ray {
    float origin[3]{};
    float endpoint[3]{};
     float weight{1.0F};
    std::uint64_t stamp_ns{};
    std::uint32_t frame_id{};
    std::uint32_t column{};
    std::uint32_t ring{};
    std::uint32_t ordinal{};
};

int cuda_scan_counts(const std::uint32_t* counts, std::uint64_t* offsets,
                     std::uint64_t* block_sums, std::size_t count, void* stream) noexcept;
}
