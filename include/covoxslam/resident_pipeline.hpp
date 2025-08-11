#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/store_pipeline.hpp>
#include <covoxslam/sensor_grid_combined.hpp>

namespace covoxslam
{
struct resident_pipeline_metrics
{
    double handoff_ms{}, esdf_ms{}, readback_ms{}, materialize_ms{};
    double esdf_gpu_ms{}, completion_ms{};
    std::uint64_t retained_tiles{}, copied_device_bytes{}, downloaded_bytes{};
};
class resident_pipeline
{
    struct implementation;
    std::unique_ptr<implementation> state_;
    resident_pipeline();
public:
    ~resident_pipeline();
    resident_pipeline(const resident_pipeline&) = delete;
    resident_pipeline& operator=(const resident_pipeline&) = delete;
    static operation_result<std::unique_ptr<resident_pipeline>> create(
        const store_pipeline_options& options, const pgo::options& solver);
    operation_status close(sensor_grid::combined_map& mapper, submap& metadata);
    operation_result<stored_pipeline_result> finish(std::span<submap> maps,
        const pgo::options& solver, bool shadow);
    resident_pipeline_metrics statistics() const;
};
}
