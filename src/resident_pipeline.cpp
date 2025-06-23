#include <covoxslam/resident_pipeline.hpp>
#if defined(covoxslam_has_cuda) && defined(covoxslam_has_urd)
#include "store_pipeline_impl.hpp"
#endif

namespace covoxslam
{
struct resident_pipeline::implementation
{
    store_pipeline_options options;
    resident_pipeline_metrics metrics;
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
        !metadata.tsdf.empty() || !metadata.esdf.empty())
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
     state_->metrics.copied_device_bytes += *count * 512 * sizeof(storage::tsdf_value);
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
    metadata.statistics.allocated_tiles = *count;
    metadata.statistics.esdf_ms = generated->total_ms;
    metadata.statistics.cuda_uploaded_bytes += generated->upload_bytes;
    metadata.config.esdf_generator = esdf_backend::cuda_block_band_soa;
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
#if defined(covoxslam_has_cuda) && defined(covoxslam_has_urd)
    state_->failed = true;
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
        const auto bytes = tsdf->tsdf.size() * 512 * sizeof(storage::tsdf_value) +
            esdf->esdf.size() * 512 * sizeof(storage::esdf_value);
        map.tsdf = std::move(tsdf->tsdf);
         map.esdf = std::move(esdf->esdf);
        map.statistics.cuda_downloaded_bytes += bytes;
        state_->metrics.downloaded_bytes += bytes;
        state_->metrics.materialize_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - copied).count();
    }
    auto completed = detail::stored_pipeline<storage::layout::pool_soa>(maps, state_->options, solver,
        shadow, state_->maps.get());
    if (!completed) return std::unexpected(completed.error());
    state_->finished = true;
    state_->failed = false;
    return completed;
#else
    return std::unexpected(operation_error{"resident completion requires CUDA and collection support"});
#endif
}
}
