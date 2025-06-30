#pragma once
#include <covoxslam/sensor_grid_map.hpp>
#include <covoxslam/sensor_grid_combined.hpp>
#include <covoxslam/sensor_grid_calibration.hpp>
#include <covoxslam/sensor_grid_precision.h>
#include <covoxslam/resident_pipeline.hpp>
#include "cli_lookup.hpp"
#include "sensor_gather_cli.hpp"
#include <algorithm>
#include <chrono>

namespace covoxslam
{
enum class sensor_backend_choice { unknown, sensor, combined };
inline constexpr cli_option_entry<sensor_backend_choice> sensor_backend_choices[]{
    {"cuda-sensor-grid-combined", sensor_backend_choice::combined}
};
enum class sensor_mapping_option { unknown, backend, interpolation, gather, threshold, block_threshold, execution, source_mask, transfer_precision, transfer_kernel, candidates, candidate_capacity, discovery };
inline constexpr cli_option_entry<sensor_mapping_option> sensor_mapping_options[]{
    {"--backend", sensor_mapping_option::backend}, {"--sensor-grid-interpolation", sensor_mapping_option::interpolation},
    {"--sensor-grid-gather", sensor_mapping_option::gather},
    {"--sensor-grid-short-list-max", sensor_mapping_option::threshold},
    {"--sensor-grid-block-list-min", sensor_mapping_option::block_threshold},
    {"--sensor-grid-execution", sensor_mapping_option::execution},
    {"--sensor-grid-source-mask", sensor_mapping_option::source_mask},
    {"--sensor-grid-transfer-precision", sensor_mapping_option::transfer_precision},
    {"--sensor-grid-transfer", sensor_mapping_option::transfer_kernel},
    {"--sensor-grid-candidates", sensor_mapping_option::candidates},
    {"--sensor-grid-candidate-capacity", sensor_mapping_option::candidate_capacity},
    {"--sensor-grid-discovery", sensor_mapping_option::discovery}
};
enum class discovery_choice { unknown, cpu, gpu_host, gpu_resident };
inline constexpr cli_option_entry<discovery_choice> discovery_choices[]{
    {"gpu-resident", discovery_choice::gpu_resident}
};
enum class candidate_choice { unknown, whole_tiles, voxel_mask, compacted_control, compact_fp32, append_fp32, compact_fp64, append_fp64, coarse_mask, whole_tile_control };
inline constexpr cli_option_entry<candidate_choice> candidate_choices[]{
    {"whole-tiles", candidate_choice::whole_tiles}
};
enum class source_mask_choice { unknown, dense, compact };
inline constexpr cli_option_entry<source_mask_choice> source_mask_choices[]{
    {"dense", source_mask_choice::dense}
};
enum class transfer_precision_choice { unknown, fp32, fp64 };
enum class transfer_kernel_choice { unknown, reference, specialized };
inline constexpr cli_option_entry<transfer_kernel_choice> transfer_kernel_choices[]{
    {"reference", transfer_kernel_choice::reference}
};
inline constexpr cli_option_entry<transfer_precision_choice> transfer_precision_choices[]{
    {"64", transfer_precision_choice::fp64}
};
enum class sensor_sampling_value { unknown, nearest, tetrahedral, trilinear, overlap, neighborhood8, neighborhood27 };
inline constexpr cli_option_entry<sensor_sampling_value> sensor_sampling_values[]{
    {"nearest", sensor_sampling_value::nearest}
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
    std::size_t maximum_bytes, sensor_grid::transfer_options transfer = {}, resident_pipeline* retained = nullptr)
{
    if (!retained || !sensor_grid::combined_gather(gather.backend) || mode != sensor_grid::interpolation::nearest || finalize)
        return std::unexpected(operation_error{"resident mapping requires combined nearest without a host finalizer"});
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
    struct context
    {
        std::unique_ptr<sensor_grid::combined_map> combined;
        std::vector<float> ranges;
        mapping_statistics statistics;
    };
    auto state = std::make_shared<context>();
    auto combined = sensor_grid::combined_map::create(*table,
        static_cast<std::uint32_t>(config.maximum_tiles), device, maximum_bytes, gather, transfer, true);

    if (!combined) return std::unexpected(combined.error());
    state->combined = std::move(*combined);
    state->ranges.resize(directions->size());
    pipeline_callbacks callbacks;
    callbacks.begin = [state, &failure, transfer](const submap_frame&) {
        state->statistics = {};
        state->statistics.cuda_accumulator_bits = sensor_grid::accumulation_bits;
        state->statistics.sensor_transfer_bits = transfer.arithmetic == sensor_grid::transfer_arithmetic::fp32 ? 32 : 64;
        const auto result = state->combined->reset();
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
        const auto result = state->combined->integrate(state->ranges, transform, mode);

        if (!result) { failure = result.error(); return; }
        const auto updates = result->transfer.updates;
        state->statistics.rays_submitted += rays.size();
        state->statistics.rays_integrated += accepted;
        state->statistics.rays_rejected += rays.size() - accepted;
        state->statistics.voxel_updates += updates;
        state->statistics.integration_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    };
    callbacks.finish = [state, &failure, config, publish, retained](const submap_frame& frame)
    {
        if (failure) return;
        submap completed;
        completed.frame = frame; completed.optimized_world_pose = frame.anchor;
        completed.config = config; completed.statistics = state->statistics;
        {
            const auto timing = state->combined->statistics();
            completed.statistics.sensor_gather_ms = timing.gather.gather_ms;
            completed.statistics.sensor_upload_ms = timing.gather.upload_ms;
            completed.statistics.sensor_graph_setup_ms = timing.gather.graph_setup_ms;
            completed.statistics.sensor_mask_ms = timing.transfer.mask_ms;
            completed.statistics.sensor_candidates_ms = timing.transfer.candidates_ms;
            completed.statistics.sensor_candidate_overflows = timing.transfer.candidate_overflows;
            completed.statistics.sensor_discovery_ms = timing.transfer.discovery_ms;
            completed.statistics.sensor_gpu_discovery_ms = timing.transfer.gpu_discovery_ms;
            completed.statistics.sensor_gpu_allocation_ms = timing.transfer.gpu_allocation_ms;
            completed.statistics.sensor_resident_dispatch_ms = timing.transfer.resident_dispatch_ms;
            completed.statistics.sensor_fusion_ms = timing.transfer.fusion_ms;
            completed.statistics.sensor_readback_ms = timing.readback_ms;
            completed.statistics.sensor_candidate_voxels = timing.transfer.candidate_voxels;
            completed.statistics.sensor_kernel_launches = timing.kernel_launches;
        }
        const auto closed = retained->close(*state->combined, completed);

        if (!closed) { failure = closed.error(); return; }
        if (publish) publish(completed);
    };
    return callbacks;
}
}
