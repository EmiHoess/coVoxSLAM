#pragma once

#include <covoxslam/core/result.hpp>
#include <covoxslam/mapping/config.hpp>
#include <covoxslam/map_store/coordinator.hpp>
#include <covoxslam/pgo/pose_graph.hpp>
#include <covoxslam/pipeline/frame_pipeline.hpp>
#include <covoxslam/pipeline/submap.hpp>
#include <covoxslam/mapping/tsdf/integrator.hpp>
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/storage/schema.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
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

namespace covoxslam::store
{
template<storage::layout l> operation_status import_active_submap(coordinated_store<l>& maps,
    const submap& source, bool band_topology = false)
{
    const auto id = source.frame.id;
    const auto& initial = source.frame.anchor;
    auto checked = maps.create(id, {initial.x, initial.y, initial.z, initial.yaw});
    if (!checked) return checked;
    const auto load = [&]<storage::schema s>(const auto& tiles) -> operation_status {
        using value_type = storage::value<s>;
        if (tiles.size() > source.config.maximum_tiles ||
            tiles.size() > std::numeric_limits<std::size_t>::max() / 512 / sizeof(value_type))
            return std::unexpected(operation_error{"canonical field exceeds import bounds"});
        std::vector<block_key> keys;
        std::vector<value_type> values;
        keys.reserve(tiles.size());
        values.reserve(tiles.size() * 512);

        for (const auto& [index, tile] : tiles) {
            keys.push_back({id, index.x, index.y, index.z});

            for (const auto& value : tile.voxels)
                values.push_back(value);
        }
        return maps.template import_field<s>(id, keys, values, s == storage::schema::tsdf ? 0 : 1);
    };
    checked = load.template operator()<storage::schema::tsdf>(source.tsdf);
    if (!checked) { static_cast<void>(maps.retire(id)); return checked; }
    if (band_topology) {
        std::vector<block_key> keys;
        keys.reserve(source.tsdf.size());

        for (const auto& [index, tile] : source.tsdf)
            keys.push_back({id, index.x, index.y, index.z});
        std::vector<storage::esdf_value> empty(keys.size() * 512);
        checked = maps.template import_field<storage::schema::esdf>(id, keys, empty, 1);
        if (!checked) { static_cast<void>(maps.retire(id)); return checked; }
    }
    else if (!source.esdf.empty()) {
        checked = load.template operator()<storage::schema::esdf>(source.esdf);
        if (!checked) { static_cast<void>(maps.retire(id)); return checked; }
    }
    auto current = maps.snapshot(id, 1, std::numeric_limits<std::size_t>::max());
    if (!current) return std::unexpected(current.error());
    const auto& optimized = source.optimized_world_pose;
    const pose_publication update{id, current->record.versions, {optimized.x, optimized.y, optimized.z, optimized.yaw}};
    return maps.publish_poses({&update, 1});
}
template<storage::layout l> operation_status import_band_topology(coordinated_store<l>& maps, const submap& source)
{
    return import_active_submap(maps, source, true);
}
template<storage::layout l> operation_status import_submap(coordinated_store<l>& maps, const submap& source)
{
    auto checked = import_active_submap(maps, source);
    if (!checked) return checked;
    return maps.finalize(source.frame.id, source.esdf.empty() ? 1U : 3U);
}
}
