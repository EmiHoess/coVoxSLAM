#pragma once
#include <covoxslam/sensor_grid_map.hpp>
#include <covoxslam/sensor_grid_calibration.hpp>
#include <covoxslam/sensor_grid_precision.h>
#include "cli_lookup.hpp"
#include "sensor_gather_cli.hpp"
#include <algorithm>
#include <chrono>

namespace covoxslam
{
enum class sensor_mapping_option { unknown, backend, interpolation, gather, threshold };
inline constexpr cli_option_entry<sensor_mapping_option> sensor_mapping_options[]{
    {"--backend", sensor_mapping_option::backend}, {"--sensor-grid-interpolation", sensor_mapping_option::interpolation},
    {"--sensor-grid-gather", sensor_mapping_option::gather},
    {"--sensor-grid-short-list-max", sensor_mapping_option::threshold}
};
enum class sensor_sampling_value { unknown, nearest, tetrahedral, trilinear, overlap, neighborhood8, neighborhood27 };
inline constexpr cli_option_entry<sensor_sampling_value> sensor_sampling_values[]{
    {"nearest", sensor_sampling_value::nearest}, {"tetrahedral", sensor_sampling_value::tetrahedral},
    {"trilinear", sensor_sampling_value::trilinear}, {"exact-overlap", sensor_sampling_value::overlap},
    {"neighborhood8", sensor_sampling_value::neighborhood8}, {"neighborhood27", sensor_sampling_value::neighborhood27}
};
inline const char* sensor_sampling_name(sensor_grid::interpolation mode)
{
    switch (mode) {
    case sensor_grid::interpolation::nearest: return "sensor-nearest";
    case sensor_grid::interpolation::tetrahedral: return "sensor-tetrahedral";
    case sensor_grid::interpolation::trilinear: return "sensor-trilinear";
    case sensor_grid::interpolation::exact_overlap: return "sensor-exact-overlap";
    case sensor_grid::interpolation::neighborhood8: return "sensor-neighborhood8";
    case sensor_grid::interpolation::neighborhood27: return "sensor-neighborhood27";
    }
    return "invalid";
}
inline operation_result<pipeline_callbacks> sensor_mapping_callbacks(const sensor& model, mapping_config config,
    int device, std::optional<operation_error>& failure, std::function<void(const submap&)> publish,
    std::function<void(submap&)> finalize, sensor_grid::interpolation mode, sensor_grid::gather_options gather,
    std::size_t maximum_bytes)
{
    if (!config.valid() || config.maximum_tiles > 1048576)
        return std::unexpected(operation_error{"invalid sensor mapping configuration"});
    auto directions = sensor_grid::calibrated_directions(model);
    if (!directions) return std::unexpected(directions.error());
    sensor_grid::config settings;
    settings.voxel_size = config.voxel_size;
     settings.radius = config.maximum_ray_length_m;
    settings.truncation = config.truncation_distance;
    settings.minimum_range = config.minimum_ray_length_m;
    settings.maximum_weight = config.maximum_weight;
    auto table = sensor_grid::build_lookup(settings, *directions);
    if (!table) return std::unexpected(table.error());
    auto grid = sensor_grid::gpu_grid::create(*table, device, maximum_bytes, gather);
    if (!grid) return std::unexpected(grid.error());
    auto map = sensor_grid::gpu_map::create(*table, static_cast<std::uint32_t>(config.maximum_tiles), device,
        maximum_bytes - (*grid)->device_bytes());
    if (!map) return std::unexpected(map.error());
    struct context
    {
        std::unique_ptr<sensor_grid::gpu_grid> grid;
        std::unique_ptr<sensor_grid::gpu_map> map;
        std::vector<float> ranges;
        mapping_statistics statistics;
    };
    auto state = std::make_shared<context>();
    state->grid = std::move(*grid); state->map = std::move(*map);
    state->ranges.resize(directions->size());
    pipeline_callbacks callbacks;
    callbacks.begin = [state, &failure](const submap_frame&) {
        state->statistics = {};
        state->statistics.cuda_accumulator_bits = sensor_grid::accumulation_bits;
        const auto result = state->map->reset();
        if (!result) failure = result.error();
    };
    callbacks.cancelled = [&failure] { return failure.has_value(); };
    callbacks.integrate_scan = [state, &failure, columns = model.columns, minimum = settings.minimum_range, mode](const submap_frame&,
        const frame_descriptor&, std::span<const ray> rays, const pose& transform)
    {
        if (failure) return;
        const auto start = std::chrono::steady_clock::now();
        std::size_t accepted{};
        std::fill(state->ranges.begin(), state->ranges.end(), 0.f);

        for (const auto& ray : rays) {
            state->ranges[ray.ring * columns + ray.column] = ray.measured_range_m;
            accepted += std::isfinite(ray.measured_range_m) && ray.measured_range_m >= minimum && ray.measured_range_m > 0;
        }
        const auto updated = state->grid->update(state->ranges);
        if (!updated) { failure = updated.error(); return; }
        const auto transferred = state->map->integrate(*state->grid, transform, mode);
        if (!transferred) { failure = transferred.error(); return; }
        state->statistics.rays_submitted += rays.size();
        state->statistics.rays_integrated += accepted;
        state->statistics.rays_rejected += rays.size() - accepted;
        state->statistics.voxel_updates += transferred->updates;
        state->statistics.integration_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    };
    callbacks.finish = [state, &failure, config, publish, finalize](const submap_frame& frame)
    {
        if (failure) return;
        auto tiles = state->map->download();
        if (!tiles) { failure = tiles.error(); return; }
        submap completed;
        completed.frame = frame; completed.optimized_world_pose = frame.anchor;
        completed.config = config; completed.statistics = state->statistics;
        completed.tsdf = std::move(*tiles);
        completed.statistics.allocated_tiles = completed.tsdf.size();
        if (config.esdf_generator == esdf_backend::cpu_blocks_batch)
            build_cpu_esdf(completed.tsdf, completed.esdf, config);
        if (finalize) finalize(completed);
        if (publish) publish(completed);
    };
    return callbacks;
}
}
