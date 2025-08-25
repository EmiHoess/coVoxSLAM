#pragma once

#include <covoxslam/core/result.hpp>
#include <covoxslam/mapping/config.hpp>
#include <covoxslam/pgo.hpp>
#include <covoxslam/pipeline/frame_pipeline.hpp>
#include <covoxslam/pipeline/submap.hpp>
#include <covoxslam/mapping/tsdf/integrator.hpp>
#include <covoxslam/storage/schema.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace covoxslam
{
enum class stored_pgo_fields { copied, borrowed, resident };

struct store_pipeline_options
{
    bool enabled{true};
    bool resident_mapping{true};
    bool negative_band{};
    esdf_backend esdf_generator{esdf_backend::cuda_block_band_soa};
    int device{};
    storage::layout layout{storage::layout::pool_soa};
    stored_pgo_fields fields{stored_pgo_fields::copied};
    std::filesystem::path checkpoint_root;
    std::size_t cpu_bytes{1073741824}, gpu_bytes{1073741824};
};

struct stored_pipeline_result
{
    std::string backend{"none"}, hash, publication{"none"}, report;
};

operation_status validate_store_pipeline(
    const store_pipeline_options& store, const pgo::options& solver);
operation_status validate_stored_export(
    std::span<const submap> maps, const store_pipeline_options& store);

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

struct mapping_session_options
{
    pipeline_options frames;
    mapping_config mapping;
    sensor_grid::gather_options gather;
    sensor_grid::transfer_options transfer;
    sensor_grid::interpolation sampling{sensor_grid::interpolation::nearest};
    store_pipeline_options store;
    pgo::options solver;
    std::size_t maximum_device_bytes{2ULL << 30};
    bool shadow{};
};

struct mapping_session_result
{
    pipeline_result frames;
    std::vector<submap> maps;
    stored_pipeline_result completion;
    resident_pipeline_metrics residency;
};

class mapping_session
{
    struct implementation;
    std::unique_ptr<implementation> state_;
    mapping_session();
public:
    ~mapping_session();
    mapping_session(const mapping_session&) = delete;
    mapping_session& operator=(const mapping_session&) = delete;
    mapping_session(mapping_session&&) noexcept;
    mapping_session& operator=(mapping_session&&) noexcept;
    static operation_result<std::unique_ptr<mapping_session>> create(
        const mapping_session_options& options);
    operation_result<mapping_session_result> run(const dataset& input);
};

static_assert(static_cast<int>(stored_pgo_fields::copied) == 0);
static_assert(static_cast<int>(stored_pgo_fields::borrowed) == 1);
static_assert(static_cast<int>(stored_pgo_fields::resident) == 2);
static_assert(std::is_same_v<std::underlying_type_t<stored_pgo_fields>, int>);
static_assert(std::is_aggregate_v<store_pipeline_options>);
static_assert(std::is_aggregate_v<stored_pipeline_result>);
static_assert(std::is_aggregate_v<resident_pipeline_metrics>);
static_assert(std::is_standard_layout_v<resident_pipeline_metrics>);
static_assert(std::is_trivially_copyable_v<resident_pipeline_metrics>);
static_assert(sizeof(resident_pipeline_metrics) == 72);
static_assert(alignof(resident_pipeline_metrics) == alignof(double));
static_assert(offsetof(resident_pipeline_metrics, handoff_ms) == 0);
static_assert(offsetof(resident_pipeline_metrics, completion_ms) == 40);
static_assert(offsetof(resident_pipeline_metrics, retained_tiles) == 48);
static_assert(offsetof(resident_pipeline_metrics, downloaded_bytes) == 64);
static_assert(sizeof(resident_pipeline) == sizeof(void*));
static_assert(std::is_aggregate_v<mapping_session_options>);
static_assert(std::is_aggregate_v<mapping_session_result>);
static_assert(sizeof(mapping_session) == sizeof(void*));
static_assert(!std::is_copy_constructible_v<mapping_session>);
static_assert(!std::is_copy_assignable_v<mapping_session>);
static_assert(std::is_nothrow_move_constructible_v<mapping_session>);
static_assert(std::is_nothrow_move_assignable_v<mapping_session>);
}
