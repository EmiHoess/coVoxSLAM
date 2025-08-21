#include <covoxslam/pipeline/mapping_session.hpp>
#include <unordered_map>
#include <unordered_set>
#if defined(covoxslam_has_cuda) && defined(covoxslam_has_urd)
#include "store_pipeline_impl.hpp"
#endif

namespace covoxslam
{
struct resident_pipeline::implementation
{
    store_pipeline_options options;
    resident_pipeline_metrics metrics;
    std::unordered_map<std::uint64_t, mapping_config> configurations;
    bool failed{}, finished{};
#if defined(covoxslam_has_cuda) && defined(covoxslam_has_urd)
    std::unique_ptr<store::coordinated_store<storage::layout::pool_soa>> maps;
#endif
};
resident_pipeline::resident_pipeline() = default;
resident_pipeline::~resident_pipeline() = default;
resident_pipeline_metrics resident_pipeline::statistics() const { return state_->metrics; }
operation_result<std::unique_ptr<resident_pipeline>> resident_pipeline::create(
    const store_pipeline_options& options, const pgo::options& solver)
{
    auto checked = validate_store_pipeline(options, solver);
    if (!checked) return std::unexpected(checked.error());

    if (!options.enabled || options.layout != storage::layout::pool_soa ||
        options.esdf_generator != esdf_backend::cuda_block_band_soa)
        return std::unexpected(operation_error{"resident mapping requires SoA storage and SoA band ESDF"});
#if defined(covoxslam_has_cuda) && defined(covoxslam_has_urd)
    checked = storage::cuda_storage_status(cudaSetDevice(options.device));
    if (!checked) return std::unexpected(checked.error());
    auto owner = std::unique_ptr<resident_pipeline>(new resident_pipeline);
    owner->state_ = std::make_unique<implementation>();
    owner->state_->options = options;
    store::residency_policy policy;
    policy.limits = {options.cpu_bytes, 128U * 1024U * 1024U, options.gpu_bytes};
     owner->state_->maps = std::make_unique<store::coordinated_store<storage::layout::pool_soa>>(policy);
    return owner;
#else
    return std::unexpected(operation_error{"resident mapping requires CUDA and collection support"});
#endif
}
operation_status resident_pipeline::close(sensor_grid::combined_map& mapper, submap& metadata)
{
    if (state_->failed || state_->finished || !metadata.config.valid() ||
        !metadata.tsdf.empty() || !metadata.esdf.empty() || state_->configurations.contains(metadata.frame.id))
        return std::unexpected(operation_error{"resident closure requires healthy device mapping"});
#if defined(covoxslam_has_cuda) && defined(covoxslam_has_urd)
    state_->failed = true;
    auto checked = storage::cuda_storage_status(cudaSetDevice(state_->options.device));
    if (!checked) return checked;
    const auto started = std::chrono::steady_clock::now();
    auto count = mapper.retain(*state_->maps, metadata.frame.id, metadata.frame.anchor);
    if (!count) return std::unexpected(count.error());
    const auto copied = std::chrono::steady_clock::now();
    state_->metrics.handoff_ms += std::chrono::duration<double, std::milli>(copied - started).count();
    state_->metrics.retained_tiles += *count;
     state_->metrics.copied_device_bytes += *count * storage::canonical_tile_bytes<storage::schema::tsdf>;
    store::band_esdf_options settings;
    settings.field.voxel_size = metadata.config.voxel_size;
    settings.field.truncation_distance = metadata.config.truncation_distance;
    settings.field.minimum_weight = metadata.config.esdf_minimum_weight;
    settings.field.site_band_m = metadata.config.esdf_site_band_m;
    settings.field.maximum_distance_m = metadata.config.esdf_maximum_distance_m;
    settings.field.include_negative_truncation_band = state_->options.negative_band;
    auto generated = store::generate_band_esdf(*state_->maps, metadata.frame.id, settings);

    if (!generated) return std::unexpected(generated.error());
    checked = state_->maps->finalize(metadata.frame.id, 3);
    if (!checked) return checked;
    state_->metrics.esdf_ms += generated->total_ms;
    state_->metrics.esdf_gpu_ms += generated->gpu_ms;
    metadata.statistics.allocated_tiles = *count;
    metadata.statistics.esdf_ms = generated->total_ms;
    metadata.statistics.cuda_uploaded_bytes += generated->upload_bytes;
    metadata.config.esdf_generator = esdf_backend::cuda_block_band_soa;
    state_->configurations.emplace(metadata.frame.id, metadata.config);
    state_->maps->collect();
    state_->failed = false;
    return {};
#else
    return std::unexpected(operation_error{"resident closure requires CUDA and collection support"});
#endif
}
operation_result<stored_pipeline_result> resident_pipeline::finish(std::span<submap> maps,
    const pgo::options& solver, bool shadow)
{
    if (state_->failed || state_->finished || maps.empty())
        return std::unexpected(operation_error{"resident completion is unavailable"});
    auto valid = validate_store_pipeline(state_->options, solver);
    if (!valid) return std::unexpected(valid.error());
    constexpr auto canonical_tile_bytes = storage::canonical_tile_bytes<storage::schema::tsdf> +
        storage::canonical_tile_bytes<storage::schema::esdf>;

    if (maps.size() != state_->configurations.size() ||
        state_->metrics.retained_tiles > state_->options.cpu_bytes / canonical_tile_bytes)
        return std::unexpected(operation_error{"resident export selection or payload budget is invalid"});
    std::unordered_set<std::uint64_t> identities;

    for (const auto& map : maps) {
        auto found = state_->configurations.find(map.frame.id);
        if (found == state_->configurations.end() || !identities.insert(map.frame.id).second ||
            !map.tsdf.empty() || !map.esdf.empty())
            return std::unexpected(operation_error{"resident completion identities or staging conflict"});
        const auto& config = found->second;

        if (!map.config.valid() || map.config.voxel_size != config.voxel_size ||
            map.config.truncation_distance != config.truncation_distance ||
            map.config.esdf_site_band_m != config.esdf_site_band_m ||
            map.config.esdf_minimum_weight != config.esdf_minimum_weight ||
            map.config.esdf_maximum_distance_m != config.esdf_maximum_distance_m ||
            map.config.esdf_generator != config.esdf_generator)
            return std::unexpected(operation_error{"resident field configuration changed before completion"});
    }
#if defined(covoxslam_has_cuda) && defined(covoxslam_has_urd)
    state_->failed = true;
    const auto completion_started = std::chrono::steady_clock::now();
    auto checked = storage::cuda_storage_status(cudaSetDevice(state_->options.device));
    if (!checked) return std::unexpected(checked.error());

    for (auto& map : maps)
    {
        const auto id = map.frame.id;
        const auto started = std::chrono::steady_clock::now();
        auto captured = store::capture_geometry(*state_->maps, std::span(&id, 1), 3,
            {map.config.voxel_size, map.config.esdf_maximum_distance_m},
            state_->options.cpu_bytes, state_->options.cpu_bytes);
        if (!captured) return std::unexpected(captured.error());
        const auto copied = std::chrono::steady_clock::now();
        state_->metrics.readback_ms += std::chrono::duration<double, std::milli>(copied - started).count();
        auto tsdf = store::geometry_volume<storage::schema::tsdf>(*captured, id, state_->options.cpu_bytes);
        if (!tsdf) return std::unexpected(tsdf.error());
        auto esdf = store::geometry_volume<storage::schema::esdf>(*captured, id, state_->options.cpu_bytes);

        if (!esdf) return std::unexpected(esdf.error());
        if (tsdf->tsdf.size() != esdf->esdf.size())
            return std::unexpected(operation_error{"resident output field topology differs"});
        const auto bytes = tsdf->tsdf.size() * storage::canonical_tile_bytes<storage::schema::tsdf> +
            esdf->esdf.size() * storage::canonical_tile_bytes<storage::schema::esdf>;
        map.tsdf = std::move(tsdf->tsdf);
         map.esdf = std::move(esdf->esdf);
        map.statistics.cuda_downloaded_bytes += bytes;
        state_->metrics.downloaded_bytes += bytes;
        state_->metrics.materialize_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - copied).count();
    }
    auto completed = detail::stored_pipeline(maps, state_->options, solver,
        shadow, *state_->maps);
    if (!completed) return std::unexpected(completed.error());
    state_->metrics.completion_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - completion_started).count();
    const auto& metrics = state_->metrics;
    std::ostringstream report;
    report << "resident handoff ms " << metrics.handoff_ms << ", ESDF derive ms " << metrics.esdf_ms
        << ", ESDF GPU intervals ms " << metrics.esdf_gpu_ms << ", readback ms " << metrics.readback_ms
        << ", canonical materialization ms " << metrics.materialize_ms << '\n'
        << "resident TSDF device copy bytes " << metrics.copied_device_bytes
        << ", canonical download bytes " << metrics.downloaded_bytes
        << ", TSDF host staging upload bytes 0, negative band " << state_->options.negative_band << '\n'
        << "resident completion wall ms " << metrics.completion_ms << " (includes readback, materialization and PGO)\n";
    completed->report = report.str() + completed->report;
    state_->finished = true;
    state_->failed = false;
    return completed;
#else
    return std::unexpected(operation_error{"resident completion requires CUDA and collection support"});
#endif
}
}
