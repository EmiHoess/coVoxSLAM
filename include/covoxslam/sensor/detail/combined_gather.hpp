#pragma once
#include <covoxslam/sensor/detail/gather_layout.hpp>

namespace covoxslam::sensor_grid
{
enum class combined_path : std::uint32_t { invalid, thread, warp, pipelined_warp, block };
#if defined(__CUDACC__)
__host__ __device__
#endif
constexpr bool accepts_uncapped_policy(gather_policy policy) noexcept
{
    return policy.max_rays == 0 && !policy.first_entry_only;
}
struct combined_schedule
{
    std::uint32_t short_list_max{combined_short_list_max}, block_list_min{combined_block_list_min};
    std::uint32_t ray_index_bits{wide_ray_index_bits};
    execution_mode execution{execution_mode::stream};
};
struct combined_sample
{
    compact_ray ray{};
    float entry_t{}, projected{};
};

#if defined(__CUDACC__)
__host__ __device__
#endif
constexpr combined_path select_combined_path(std::uint32_t length, gather_policy policy,
    combined_schedule schedule) noexcept
{
    if (!length || !schedule.short_list_max || schedule.block_list_min <= schedule.short_list_max ||
        (schedule.ray_index_bits != narrow_ray_index_bits && schedule.ray_index_bits != wide_ray_index_bits) ||
        (policy.first_entry_only && policy.max_rays > 1)) return combined_path::invalid;
    if (policy.first_entry_only || length <= schedule.short_list_max) return combined_path::thread;
    if (policy.max_rays) return combined_path::warp;
    return length >= schedule.block_list_min ? combined_path::block : combined_path::pipelined_warp;
}

#if defined(__CUDACC__)
__host__ __device__
#endif
constexpr combined_schedule make_combined_schedule(std::uint32_t ray_count, gather_options options) noexcept
{
    return {options.short_list_max, options.block_list_min,
        ray_count > 0 && ray_count <= narrow_ray_capacity ? narrow_ray_index_bits : wide_ray_index_bits,
        options.execution};
}
}
