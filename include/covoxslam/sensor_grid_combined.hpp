#pragma once
#include <covoxslam/sensor_grid_map.hpp>

namespace covoxslam::sensor_grid
{
struct combined_times
{
    update_times gather{};
    transfer_times transfer{};
    double integration_ms{}, readback_ms{};
    std::uint64_t completed_scans{};
};
class combined_map
{
    std::unique_ptr<gpu_grid> grid;
    std::unique_ptr<gpu_map> map;
    combined_times totals{};
    bool failed{};
    combined_map() = default;
public:
    static operation_result<std::unique_ptr<combined_map>> create(const lookup& table,
        std::uint32_t capacity = 32768, int device = 0, std::size_t maximum_bytes = 1073741824,
        gather_options options = {gather_backend::hybrid_combined});
    operation_result<combined_times> integrate(std::span<const float> ranges, pose local_from_sensor,
        interpolation mode = interpolation::trilinear, gather_policy policy = {});
    operation_result<tsdf_tiles> download();
    operation_status reset();
    combined_times statistics() const noexcept { return totals; }
    execution_statistics graph_statistics() const noexcept { return grid->execution_statistics_value(); }
    combined_map(const combined_map&) = delete;
    combined_map& operator=(const combined_map&) = delete;
};
}
