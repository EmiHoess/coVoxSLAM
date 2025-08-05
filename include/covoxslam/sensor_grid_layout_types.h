#pragma once
#include <covoxslam/sensor_grid_types.h>

namespace covoxslam::sensor_grid
{
enum class gather_backend : std::uint32_t { csr, warp_soa_prepared, thread_interleaved_prepared, warp_soa_hybrid_prepared,
    warp_soa_subwarp_prepared, hybrid_short_interleaved, hybrid_aligned, hybrid_ray16, hybrid_compact,
    hybrid_pipelined, hybrid_block_long, hybrid_combined, hybrid_combined_uncapped };
constexpr bool combined_gather(gather_backend backend) noexcept
{
    return backend == gather_backend::hybrid_combined;
}
enum class execution_mode : std::uint32_t { stream, cuda_graph };
struct gather_options
{
    gather_backend backend{gather_backend::hybrid_combined};
    std::uint32_t short_list_max{16}, block_list_min{1024};
    execution_mode execution{execution_mode::stream};
};
struct prepared_ray { float range{}, end{}; std::uint32_t flags{}; };
struct alignas(8) compact_ray { float range{}, end{}; };
struct row_range { std::uint32_t start{}, count{}; };
struct layout_view
{
    const std::uint32_t* starts{};
    const std::uint32_t* lengths{};
    const std::uint32_t* cells{};
    const std::uint32_t* ray_ids{};
    const float* entry_t{};
     const float* projected{};
    std::uint32_t rows{}, entries{}, stride{1};
    const std::uint32_t* row_ids{};
    std::uint32_t short_rows{};
    std::uint32_t bin_ends[2][4]{};
    const row_range* ranges{};
    const std::uint16_t* ray_ids16{};
    std::uint32_t short_stride{1};
    std::uint32_t block_rows{};
};
}
