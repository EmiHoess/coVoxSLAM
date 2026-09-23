#include <covoxslam/pipeline/mapping_session.hpp>
#include <covoxslam/sensor/lookup.hpp>
#include <covoxslam/sensor/detail/precision.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

namespace covoxslam
{
namespace
{
struct mapping_callback_state
{
    std::unique_ptr<sensor_grid::combined_map> combined;
    std::vector<float> ranges;
    mapping_statistics statistics;
};

operation_status validate_mapping_session(const mapping_session_options& options)
{
    if (!sensor_grid::matches_combined_nearest({options.mapping, options.gather,
            options.transfer, options.sampling, options.frames.ring_stride,
            options.frames.column_stride}))
        return std::unexpected(operation_error{"explicit options conflict with combined-nearest preset"});
    if (!options.frames.submap_duration_ns || !options.frames.ring_stride ||
        !options.frames.column_stride)
        return std::unexpected(operation_error{"duration and sampling strides must be positive"});
    const auto stored = validate_store_pipeline(options.store, options.solver);
    if (!stored) return std::unexpected(stored.error());
    if (!options.store.resident_mapping)
        return std::unexpected(operation_error{
            "resident mapping requires combined nearest without a host finalizer"});
    if (!sensor_grid::combined_gather(options.gather.backend) ||
        options.sampling != sensor_grid::interpolation::nearest ||
        options.transfer.discovery != sensor_grid::discovery_mode::gpu_resident)
        return std::unexpected(operation_error{
            "resident mapping storage requires combined nearest with resident discovery"});
    if (!options.mapping.valid() || options.mapping.maximum_tiles > maximum_mapping_tiles)
        return std::unexpected(operation_error{"invalid sensor mapping configuration"});
    return {};
}

operation_result<pipeline_callbacks> make_mapping_callbacks(const dataset& input,
    const mapping_session_options& options, resident_pipeline& resident,
    std::vector<submap>& maps)
{
    auto directions = sensor_grid::calibrated_directions(input.sensor);
    if (!directions) return std::unexpected(directions.error());
    const auto settings = sensor_grid::sensor_config_from_mapping(options.mapping);
    auto table = sensor_grid::build_lookup(settings, *directions);
    if (!table) return std::unexpected(table.error());
    auto combined = sensor_grid::combined_map::create(*table,
        static_cast<std::uint32_t>(options.mapping.maximum_tiles), options.store.device,
        options.maximum_device_bytes, options.gather, options.transfer, true);
    if (!combined) return std::unexpected(combined.error());
    auto state = std::make_shared<mapping_callback_state>();
    state->combined = std::move(*combined);
    state->ranges.resize(directions->size());
    pipeline_callbacks callbacks;
    callbacks.begin = [state, transfer = options.transfer](const submap_frame&) -> operation_status {
        const auto result = state->combined->reset();
        if (!result) return std::unexpected(result.error());
        state->statistics = {};
        state->statistics.cuda_accumulator_bits = sensor_grid::accumulation_bits;
        state->statistics.sensor_transfer_bits = transfer.arithmetic ==
            sensor_grid::transfer_arithmetic::fp32 ? 32 : 64;
        return {};
    };
    callbacks.integrate_scan = [state, columns = input.sensor.columns,
        minimum = settings.minimum_range, mode = options.sampling](const submap_frame&,
        const frame_descriptor&, std::span<const ray> rays, const pose& transform) -> operation_status
    {
        const auto start = std::chrono::steady_clock::now();
        std::size_t accepted{};
        std::fill(state->ranges.begin(), state->ranges.end(), 0.f);

        for (const auto& ray : rays) {
            state->ranges[ray.ring * columns + ray.column] = ray.measured_range_m;
            accepted += std::isfinite(ray.measured_range_m) &&
                ray.measured_range_m >= minimum && ray.measured_range_m > 0;
        }
        const auto result = state->combined->integrate(state->ranges, transform, mode);

        if (!result) return std::unexpected(result.error());
        const auto updates = result->transfer.updates;
        state->statistics.rays_submitted += rays.size();
        state->statistics.rays_integrated += accepted;
        state->statistics.rays_rejected += rays.size() - accepted;
        state->statistics.voxel_updates += updates;
        state->statistics.integration_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        return {};
    };
    callbacks.finish = [state, config = options.mapping, &resident, &maps](
        const submap_frame& frame) -> operation_status
    {
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
        const auto closed = resident.close(*state->combined, completed);

        if (!closed) return std::unexpected(closed.error());
        maps.push_back(completed);
        return {};
    };
    return callbacks;
}
}

struct mapping_session::implementation
{
    mapping_session_options options;
    std::unique_ptr<resident_pipeline> resident;
    bool consumed{};
};

mapping_session::mapping_session() = default;
mapping_session::~mapping_session() = default;
mapping_session::mapping_session(mapping_session&&) noexcept = default;
mapping_session& mapping_session::operator=(mapping_session&&) noexcept = default;

operation_result<std::unique_ptr<mapping_session>> mapping_session::create(
    const mapping_session_options& options)
{
    const auto valid = validate_mapping_session(options);
    if (!valid) return std::unexpected(valid.error());
    auto resident = resident_pipeline::create(options.store, options.solver);
    if (!resident) return std::unexpected(resident.error());
    auto owner = std::unique_ptr<mapping_session>(new mapping_session);
    owner->state_ = std::make_unique<implementation>();
    owner->state_->options = options;
    owner->state_->resident = std::move(*resident);
    return owner;
}

operation_result<mapping_session_result> mapping_session::run(const dataset& input)
{
    if (!state_ || state_->consumed)
        return std::unexpected(operation_error{"mapping session is unavailable"});
    state_->consumed = true;
    std::vector<submap> maps;
    auto callbacks = make_mapping_callbacks(input, state_->options, *state_->resident, maps);
    if (!callbacks) return std::unexpected(callbacks.error());
    auto frames = run_frame_pipeline(input, state_->options.frames, *callbacks);
    if (!frames) return std::unexpected(frames.error());
    if (frames->cancelled)
        return std::unexpected(operation_error{"mapping cancelled"});
    stored_pipeline_result completion;
    if (state_->options.solver.enabled || state_->options.store.enabled) {
        auto stored = state_->resident->finish(
            maps, state_->options.solver, state_->options.shadow);
        if (!stored) return std::unexpected(operation_error{
            "stored pipeline stopped before export: " + stored.error().message});
        const auto ready = validate_stored_export(maps, state_->options.store);
        if (!ready) return std::unexpected(ready.error());
        completion = std::move(*stored);
    }
    mapping_session_result result;
    result.frames = std::move(*frames);
    result.maps = std::move(maps);
    result.completion = std::move(completion);
    result.residency = state_->resident->statistics();
    return result;
}
}
