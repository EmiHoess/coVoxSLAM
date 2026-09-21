#pragma once

#include <covoxslam/core/byte_units.hpp>
#include <covoxslam/core/result.hpp>
#include <covoxslam/sensor/lookup.hpp>
#include <covoxslam/sensor/detail/gather_layout.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

namespace covoxslam::sensor_grid
{
struct update_times { float upload_ms{}, gather_ms{}, prepare_ms{}, bin_ms[4]{}; double graph_setup_ms{}; bool graph_reused{}; };
struct execution_statistics
{
    std::uint64_t graph_builds{}, graph_hits{}, invalidations{};
    double setup_ms{}, invalidation_ms{};
};
struct gather_statistics { std::uint32_t short_rows{}, warp_rows{}, block_rows{}, logical_entries{}, stored_entries{}; };
static_assert(std::is_aggregate_v<update_times>);
static_assert(std::is_standard_layout_v<update_times>);
static_assert(std::is_trivially_copyable_v<update_times>);
static_assert(sizeof(update_times) == 48);
static_assert(alignof(update_times) == alignof(double));
static_assert(offsetof(update_times, upload_ms) == 0);
static_assert(offsetof(update_times, bin_ms) == 12);
static_assert(offsetof(update_times, graph_setup_ms) == 32);
static_assert(offsetof(update_times, graph_reused) == 40);
static_assert(update_times{}.graph_setup_ms == 0);
static_assert(!update_times{}.graph_reused);

static_assert(std::is_aggregate_v<execution_statistics>);
static_assert(std::is_standard_layout_v<execution_statistics>);
static_assert(std::is_trivially_copyable_v<execution_statistics>);
static_assert(sizeof(execution_statistics) == 40);
static_assert(alignof(execution_statistics) == alignof(std::uint64_t));
static_assert(offsetof(execution_statistics, graph_builds) == 0);
static_assert(offsetof(execution_statistics, invalidations) == 16);
static_assert(offsetof(execution_statistics, setup_ms) == 24);
static_assert(offsetof(execution_statistics, invalidation_ms) == 32);

static_assert(std::is_aggregate_v<gather_statistics>);
static_assert(std::is_standard_layout_v<gather_statistics>);
static_assert(std::is_trivially_copyable_v<gather_statistics>);
static_assert(sizeof(gather_statistics) == 20);
static_assert(alignof(gather_statistics) == alignof(std::uint32_t));
static_assert(offsetof(gather_statistics, short_rows) == 0);
static_assert(offsetof(gather_statistics, warp_rows) == 4);
static_assert(offsetof(gather_statistics, block_rows) == 8);
static_assert(offsetof(gather_statistics, logical_entries) == 12);
static_assert(offsetof(gather_statistics, stored_entries) == 16);
class gpu_grid
{
    struct implementation;
    std::unique_ptr<implementation> state;
    gpu_grid();
public:
    static operation_result<std::unique_ptr<gpu_grid>> create(const lookup& table,
        int device = 0, std::size_t maximum_bytes = bytes_per_gibibyte, gather_options options = {});
    ~gpu_grid();
     gpu_grid(const gpu_grid&) = delete;
    gpu_grid& operator=(const gpu_grid&) = delete;
    std::size_t device_bytes() const noexcept;
    int device_index() const noexcept;
    gather_statistics statistics() const noexcept;
    execution_statistics execution_statistics_value() const noexcept;
    operation_status invalidate_graphs();
    field_view device_field() const noexcept;
    operation_result<update_times> update(std::span<const float> ranges, gather_policy policy = {});
    operation_result<std::vector<voxel>> download();
};
static_assert(sizeof(gpu_grid) == sizeof(void*));
static_assert(alignof(gpu_grid) == alignof(void*));
static_assert(!std::is_copy_constructible_v<gpu_grid>);
static_assert(!std::is_copy_assignable_v<gpu_grid>);
static_assert(!std::is_move_constructible_v<gpu_grid>);
static_assert(!std::is_move_assignable_v<gpu_grid>);
}
