#pragma once
#include <covoxslam/sensor_grid.hpp>
#include <covoxslam/sensor_grid_sampling.h>

namespace covoxslam::sensor_grid
{
enum class transfer_arithmetic : std::uint32_t { fp64, fp32 };
enum class source_mask_mode : std::uint32_t { dense, compact };
enum class discovery_mode : std::uint32_t { cpu, gpu_host, gpu_resident };
enum class candidate_mode : std::uint32_t { whole_tiles, voxel_mask, compacted_control, compact_fp32, append_fp32, compact_fp64, append_fp64, coarse_mask, whole_tile_control };
constexpr bool candidate_uses_mask(candidate_mode mode) noexcept
{
    return mode != candidate_mode::whole_tiles && mode != candidate_mode::whole_tile_control;
}
constexpr std::uint32_t candidate_mask_words(candidate_mode mode) noexcept {
    return candidate_uses_mask(mode) ? (mode == candidate_mode::coarse_mask ? 1 : 16) : 0;
}
constexpr bool candidate_uses_list(candidate_mode mode) noexcept
{
    return mode == candidate_mode::compact_fp32 || mode == candidate_mode::append_fp32 ||
        mode == candidate_mode::compact_fp64 || mode == candidate_mode::append_fp64;
}
struct transfer_options
{
    source_mask_mode source_mask{source_mask_mode::dense};
    transfer_arithmetic arithmetic{transfer_arithmetic::fp64};
    bool specialized{};
    candidate_mode candidates{candidate_mode::whole_tiles};
    std::uint32_t candidate_capacity{1048576};
    discovery_mode discovery{discovery_mode::gpu_resident};
};
}
