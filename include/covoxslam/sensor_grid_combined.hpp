#pragma once
#include <covoxslam/sensor_grid_map.hpp>

namespace covoxslam::sensor_grid
{
struct mapping_preset
{
    mapping_config mapping;
    gather_options gather;
    transfer_options transfer;
    interpolation sampling{interpolation::trilinear};
    std::size_t ring_stride{1}, column_stride{1};
};
operation_result<mapping_preset> combined_nearest_preset();
bool matches_combined_nearest(const mapping_preset& options) noexcept;
struct combined_times
{
    update_times gather{};
    transfer_times transfer{};
    double integration_ms{}, readback_ms{};
    std::uint64_t completed_scans{}, kernel_launches{};
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
        gather_options options = {gather_backend::hybrid_combined}, transfer_options transfer = {},
        bool resident_soa = false);
    operation_result<combined_times> integrate(std::span<const float> ranges, pose local_from_sensor,
        interpolation mode = interpolation::trilinear, gather_policy policy = {});
    operation_result<tsdf_tiles> download();
    operation_result<std::size_t> retain(
        store::coordinated_store<storage::layout::pool_soa, storage::schema::esdf>& destination,
        std::uint64_t id, pose4 anchor);
    operation_status reset();
    combined_times statistics() const noexcept { return totals; }
    execution_statistics graph_statistics() const noexcept { return grid->execution_statistics_value(); }
    combined_map(const combined_map&) = delete;
    combined_map& operator=(const combined_map&) = delete;
};
}
