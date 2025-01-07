#pragma once

#include <covoxslam/sensor_grid.hpp>
#include <covoxslam/sensor_grid_layout_types.h>
#include <memory>

namespace covoxslam::sensor_grid
{
struct update_times { float upload_ms{}, gather_ms{}, prepare_ms{}, bin_ms[4]{}; double graph_setup_ms{}; bool graph_reused{}; };
struct execution_statistics
{
    std::uint64_t graph_builds{}, graph_hits{}, invalidations{};
    double setup_ms{}, invalidation_ms{};
};
struct gather_statistics { std::uint32_t short_rows{}, warp_rows{}, block_rows{}, logical_entries{}, stored_entries{}; };
class gpu_grid
{
    struct implementation;
    std::unique_ptr<implementation> state;
    gpu_grid();
public:
    static operation_result<std::unique_ptr<gpu_grid>> create(const lookup& table,
        int device = 0, std::size_t maximum_bytes = 1073741824, gather_options options = {});
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
}
