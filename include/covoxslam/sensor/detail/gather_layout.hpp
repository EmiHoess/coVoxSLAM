#pragma once
#include <covoxslam/sensor/detail/types.hpp>
#include <cstddef>
#include <type_traits>

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
static_assert(std::is_same_v<std::underlying_type_t<gather_backend>, std::uint32_t>);
static_assert(static_cast<std::uint32_t>(gather_backend::csr) == 0);
static_assert(static_cast<std::uint32_t>(gather_backend::warp_soa_prepared) == 1);
static_assert(static_cast<std::uint32_t>(gather_backend::thread_interleaved_prepared) == 2);
static_assert(static_cast<std::uint32_t>(gather_backend::warp_soa_hybrid_prepared) == 3);
static_assert(static_cast<std::uint32_t>(gather_backend::warp_soa_subwarp_prepared) == 4);
static_assert(static_cast<std::uint32_t>(gather_backend::hybrid_short_interleaved) == 5);
static_assert(static_cast<std::uint32_t>(gather_backend::hybrid_aligned) == 6);
static_assert(static_cast<std::uint32_t>(gather_backend::hybrid_ray16) == 7);
static_assert(static_cast<std::uint32_t>(gather_backend::hybrid_compact) == 8);
static_assert(static_cast<std::uint32_t>(gather_backend::hybrid_pipelined) == 9);
static_assert(static_cast<std::uint32_t>(gather_backend::hybrid_block_long) == 10);
static_assert(static_cast<std::uint32_t>(gather_backend::hybrid_combined) == 11);
static_assert(static_cast<std::uint32_t>(gather_backend::hybrid_combined_uncapped) == 12);
static_assert(combined_gather(gather_backend::hybrid_combined));
static_assert(!combined_gather(gather_backend::hybrid_combined_uncapped));
static_assert(std::is_same_v<std::underlying_type_t<execution_mode>, std::uint32_t>);
static_assert(static_cast<std::uint32_t>(execution_mode::stream) == 0);
static_assert(static_cast<std::uint32_t>(execution_mode::cuda_graph) == 1);

static_assert(std::is_aggregate_v<gather_options>);
static_assert(std::is_standard_layout_v<gather_options>);
static_assert(std::is_trivially_copyable_v<gather_options>);
static_assert(sizeof(gather_options) == 16);
static_assert(alignof(gather_options) == alignof(std::uint32_t));
static_assert(offsetof(gather_options, backend) == 0);
static_assert(offsetof(gather_options, short_list_max) == 4);
static_assert(offsetof(gather_options, block_list_min) == 8);
static_assert(offsetof(gather_options, execution) == 12);
static_assert(gather_options{}.backend == gather_backend::hybrid_combined);
static_assert(gather_options{}.short_list_max == 16);
static_assert(gather_options{}.block_list_min == 1024);
static_assert(gather_options{}.execution == execution_mode::stream);

static_assert(std::is_aggregate_v<prepared_ray>);
static_assert(std::is_standard_layout_v<prepared_ray>);
static_assert(std::is_trivially_copyable_v<prepared_ray>);
static_assert(sizeof(prepared_ray) == 12);
static_assert(alignof(prepared_ray) == alignof(std::uint32_t));
static_assert(offsetof(prepared_ray, range) == 0);
static_assert(offsetof(prepared_ray, end) == 4);
static_assert(offsetof(prepared_ray, flags) == 8);
static_assert(std::is_aggregate_v<compact_ray>);
static_assert(std::is_standard_layout_v<compact_ray>);
static_assert(std::is_trivially_copyable_v<compact_ray>);
static_assert(sizeof(compact_ray) == 8);
static_assert(alignof(compact_ray) == 8);
static_assert(offsetof(compact_ray, range) == 0);
static_assert(offsetof(compact_ray, end) == 4);
static_assert(std::is_aggregate_v<row_range>);
static_assert(std::is_standard_layout_v<row_range>);
static_assert(std::is_trivially_copyable_v<row_range>);
static_assert(sizeof(row_range) == 8);
static_assert(alignof(row_range) == alignof(std::uint32_t));
static_assert(offsetof(row_range, start) == 0);
static_assert(offsetof(row_range, count) == 4);

static_assert(sizeof(void*) == 8);
static_assert(std::is_aggregate_v<layout_view>);
static_assert(std::is_standard_layout_v<layout_view>);
static_assert(std::is_trivially_copyable_v<layout_view>);
static_assert(sizeof(layout_view) == 136);
static_assert(alignof(layout_view) == alignof(void*));
static_assert(offsetof(layout_view, starts) == 0);
static_assert(offsetof(layout_view, lengths) == 8);
static_assert(offsetof(layout_view, cells) == 16);
static_assert(offsetof(layout_view, ray_ids) == 24);
static_assert(offsetof(layout_view, entry_t) == 32);
static_assert(offsetof(layout_view, projected) == 40);
static_assert(offsetof(layout_view, rows) == 48);
static_assert(offsetof(layout_view, entries) == 52);
static_assert(offsetof(layout_view, stride) == 56);
static_assert(offsetof(layout_view, row_ids) == 64);
static_assert(offsetof(layout_view, short_rows) == 72);
static_assert(offsetof(layout_view, bin_ends) == 76);
static_assert(offsetof(layout_view, ranges) == 112);
static_assert(offsetof(layout_view, ray_ids16) == 120);
static_assert(offsetof(layout_view, short_stride) == 128);
static_assert(offsetof(layout_view, block_rows) == 132);
static_assert(layout_view{}.stride == 1);
static_assert(layout_view{}.short_stride == 1);
}
