#pragma once
#include <covoxslam/store_pipeline.hpp>
#include <covoxslam/store/canonical_import.hpp>
#include <covoxslam/store/geometry_capture.hpp>
#include <covoxslam/store/pgo_cpu.hpp>
#include <covoxslam/store/pgo_cuda.hpp>
#include <covoxslam/store/pgo_cache.hpp>
#include <covoxslam/pgo_publication.hpp>
#include <chrono>
#include <sstream>
#if defined(covoxslam_has_cuda) && defined(covoxslam_has_urd)
#include <covoxslam/store/gpu_band_esdf.hpp>
#include <covoxslam/store/band_esdf_materialize.hpp>
#endif

namespace covoxslam::detail
{
inline operation_result<store::checkpoint_parameters> stored_field_parameters(
    const submap& map, const store_pipeline_options& options)
{
    const auto& config = map.config;

    if (!config.valid() || (options.esdf_generator == esdf_backend::cuda_block_band_soa &&
        (config.esdf_generator != options.esdf_generator || map.esdf.size() != map.tsdf.size())))
        return std::unexpected(operation_error{"stored field identity does not match completed generation"});
    return store::checkpoint_parameters{config.voxel_size, config.truncation_distance,
        config.maximum_weight, config.esdf_maximum_distance_m, std::uint64_t(config.esdf_generator)};
}
#if defined(covoxslam_has_pgo)
template<storage::layout l> operation_result<std::vector<store::pgo_field_binding>>
bind_stored_fields(const store::geometry_snapshot<l, storage::schema::esdf>& captured,
    std::span<const std::uint64_t> ids)
{
    std::vector<store::pgo_field_binding> result;
    result.reserve(ids.size());

    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        const auto* map = captured.find(ids[i]);
        if (!map || !map->tsdf || !map->esdf || map->record.state != store::lifecycle::finalized)
            return std::unexpected(operation_error{"solver requires finalized canonical fields"});
        const auto& version = map->record.versions;

        if (!version.tsdf || !version.esdf || version.tsdf != version.esdf_source)
            return std::unexpected(operation_error{"solver ESDF source differs from captured TSDF"});
        result.push_back({ids[i], i, version});
    }
    return result;
}
#endif
template<storage::layout l> operation_status generate_stored_band(store::coordinated_store<l>& retained,
    submap& map, const store_pipeline_options& options, std::ostream& report)
{
#if defined(covoxslam_has_cuda) && defined(covoxslam_has_urd)
    if constexpr (l == storage::layout::pool_soa)
    {
        using namespace store;
        auto checked = storage::cuda_storage_status(cudaSetDevice(options.device));
        if (!checked) return checked;
        checked = import_band_topology(retained, map);
        if (!checked) return checked;
        const auto id = map.frame.id;
        checked = retained.synchronize_gpu(id, 3, options.cpu_bytes);
        if (!checked) return checked;
        const auto& config = map.config;
        band_esdf_options settings;
        settings.field.voxel_size = config.voxel_size;
        settings.field.truncation_distance = config.truncation_distance;
        settings.field.minimum_weight = config.esdf_minimum_weight;
        settings.field.site_band_m = config.esdf_site_band_m;
        settings.field.maximum_distance_m = config.esdf_maximum_distance_m;
        auto generated = generate_band_esdf(retained, id, settings);

        if (!generated) return std::unexpected(generated.error());
        auto downloaded = download_band_fields(retained, id,
            {config.voxel_size, config.esdf_maximum_distance_m}, options.cpu_bytes, *generated);
        if (!downloaded) return std::unexpected(downloaded.error());
        band_field_staging<l> staging(std::move(*downloaded), id);
        checked = staging.validate(retained);
        if (!checked) return checked;
        auto values = staging.values(options.cpu_bytes);
        if (!values) return std::unexpected(values.error());
        checked = retained.finalize(id, 3);
        if (!checked) return checked;
        map.esdf = **values;
        map.config.esdf_generator = esdf_backend::cuda_block_band_soa;
        map.statistics.esdf_ms = generated->total_ms + generated->readback_ms;
        map.statistics.cuda_uploaded_bytes += generated->upload_bytes;
        map.statistics.cuda_downloaded_bytes += generated->download_bytes;
        report << "SoA ESDF submap " << id << ", tiles " << generated->tiles << ", rounds " << generated->rounds
            << ", allocation and replacement ms " << generated->allocation_ms
            << ", upload submission ms " << generated->upload_ms
            << ", propagation wall ms " << generated->propagation_ms
            << ", publication wall ms " << generated->publication_ms
            << ", GPU stage intervals ms " << generated->gpu_ms << ", host wait ms " << generated->wait_ms
            << ", readback wall ms " << generated->readback_ms << ", derive wall ms " << generated->total_ms
            << ", scratch bytes " << generated->scratch_bytes << ", peak GPU bytes " << generated->peak_device_bytes
            << ", metadata upload bytes " << generated->upload_bytes << ", readback bytes " << generated->download_bytes
            << " (nested intervals, not additive)\n";
        return {};
    }
#endif
    return std::unexpected(operation_error{"stored band ESDF requires CUDA SoA support"});
}
template<storage::layout l> operation_result<stored_pipeline_result> stored_pipeline(
    std::span<submap> maps, const store_pipeline_options& options, const pgo::options& solver, bool shadow)
{
    using namespace store;
    const auto started = std::chrono::steady_clock::now();
    const bool generated_fields = options.esdf_generator == esdf_backend::cuda_block_band_soa;
    residency_policy policy;
    policy.limits = {options.cpu_bytes, 128U * 1024U * 1024U, options.gpu_bytes};
    coordinated_store<l> retained(policy);
    std::vector<std::uint64_t> ids;
#if defined(covoxslam_has_pgo)
    std::vector<pgo_field_binding> bindings;
#endif
    bool has_esdf{};
    std::ostringstream report;

    for (auto& map : maps) {
        auto checked = generated_fields ?
            generate_stored_band(retained, map, options, report) : import_submap(retained, map);
        if (!checked) return std::unexpected(checked.error());
        ids.push_back(map.frame.id);
        if (ids.size() == 1) has_esdf = !map.esdf.empty();
        if (has_esdf != !map.esdf.empty())
            return std::unexpected(operation_error{"stored pipeline requires a consistent field selection"});
    }
    const auto imported = std::chrono::steady_clock::now();
    auto export_ready = validate_stored_export(maps, options);
    if (!export_ready) return std::unexpected(export_ready.error());
    stored_pipeline_result output;
    constexpr const char* layouts[]{"aos", "aosoa", "soa"};
    constexpr const char* field_modes[]{"copied", "borrowed", "resident"};
    report << "store layout " << layouts[unsigned(options.layout)] << ", PGO fields "
        << field_modes[unsigned(options.fields)] << ", checkpoint format 1, field view API 1\n"
        << "mapping bridge finalized canonical staging, publication verification CPU\n";
    report << "store import ms " << std::chrono::duration<double, std::milli>(imported - started).count() << '\n';
    if (generated_fields)
        report << "ESDF direct SoA generation, retained map store, canonical CPU mirror\n";
#if defined(covoxslam_has_pgo)
    if (solver.enabled) {
        if (!has_esdf) return std::unexpected(operation_error{"stored PGO requires ESDF fields"});
        const auto& config = maps.front().config;
        geometry_parameters geometry{config.voxel_size, config.esdf_maximum_distance_m, geometry_pose::optimized};
        auto host = capture_geometry(retained, ids, 3, geometry, options.cpu_bytes);
        if (!host) return std::unexpected(host.error());
        auto graph = pgo::make_submap_graph(maps);
        if (!graph) return std::unexpected(graph.error());
        auto candidates = pgo::find_candidates(maps);
        if (!candidates) return std::unexpected(candidates.error());
        auto checked = pgo::append_registration(*graph, maps, *candidates, {}, options.fields == stored_pgo_fields::copied);
        if (!checked) return std::unexpected(checked.error());
        auto counts = pgo::summarize_candidates(*candidates);
        if (!counts) return std::unexpected(counts.error());

        auto selected = bind_stored_fields(*host, ids);
        if (!selected) return std::unexpected(selected.error());
        bindings = std::move(*selected);
        checked = validate_pgo_bindings(*graph, *host, bindings);
        if (!checked) return std::unexpected(checked.error());
        auto fields = borrow_pgo_fields(*host, bindings);
        if (!fields) return std::unexpected(fields.error());
        auto frozen = options.fields == stored_pgo_fields::copied ? pgo::freeze_graph(*graph) :
            pgo::freeze_field_graph(*graph, *fields);
        if (!frozen) return std::unexpected(frozen.error());
        operation_result<ceronoceres::SolveResult> solved = std::unexpected(operation_error{"no solver selected"});
        preparation_cache cache;
        switch (options.fields) {
        case stored_pgo_fields::copied:
            solved = pgo::solve_native(*frozen, solver);
            break;
        case stored_pgo_fields::borrowed: {
            auto prepared = cache.prepare(*graph, *fields);
            if (!prepared) return std::unexpected(prepared.error());
            solved = solve_prepared_fields(*prepared, solver);
            break;
        }
        case stored_pgo_fields::resident: {
#if defined(covoxslam_has_cuda)
            checked = storage::cuda_storage_status(cudaSetDevice(solver.device));
            if (!checked) return std::unexpected(checked.error());

            for (auto id : ids) {
                auto state = retained.status(id);
                if (!state) return std::unexpected(state.error());

                if (state->gpu != state->content) {
                    checked = retained.synchronize_gpu(id, 3, options.cpu_bytes);
                    if (!checked) return std::unexpected(checked.error());
                }
            }
            auto resident = retained.lease_resident(ids, 3, options.gpu_bytes);
            if (!resident) return std::unexpected(resident.error());
            auto gpu_fields = resident_pgo_fields<l, storage::schema::esdf>(*resident, bindings, geometry);
            if (!gpu_fields) return std::unexpected(gpu_fields.error());
            auto prepared = cache.prepare(*graph, *gpu_fields);
            if (!prepared) return std::unexpected(prepared.error());
            solved = solve_prepared_fields(*prepared, solver);
#else
            return std::unexpected(operation_error{"resident fields require CUDA support"});
#endif
            break;
        }
        }
        if (!solved) return std::unexpected(solved.error());
        output.backend = ceronoceres::backend_name(solved->backend);
        output.hash = frozen->hash();
        output.publication = "rejected";
        checked = pgo::write_solver_report(report, *solved);
        if (!checked) return std::unexpected(checked.error());
        checked = write_preparation_metrics(report, cache.metrics(), solved->telemetry);
        if (!checked) return std::unexpected(checked.error());
        if (solved->solution_usable) {
            const auto verification = std::chrono::steady_clock::now();
            pgo::optimization_result optimized{std::move(*frozen), std::move(*solved), *counts, pgo::publication_hash(maps)};
            auto update = pgo::prepare_publication(maps, optimized);
            if (!update) return std::unexpected(update.error());
            std::vector<pose_publication> poses;

            for (std::size_t i = 0; i < ids.size(); ++i) {
                const auto& pose = update->poses()[i];
                poses.push_back({ids[i], bindings[i].expected, {pose.x, pose.y, pose.z, pose.yaw}});
            }
            if (!shadow) {
                checked = retained.publish_poses(poses);
                if (!checked) return std::unexpected(checked.error());
            }
            checked = pgo::publish_poses(maps, *update, shadow ? pgo::publication_mode::shadow : pgo::publication_mode::apply);
            if (!checked) return std::unexpected(checked.error());
            output.publication = shadow ? "shadow" : "applied";
            report << "publication verification ms " << std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - verification).count() << '\n';
        }
    }
#else
    if (solver.enabled) return std::unexpected(operation_error{"PGO support was not built"});
#endif
    if (!options.checkpoint_root.empty()) {
        std::error_code error;
        std::filesystem::create_directories(options.checkpoint_root, error);
        if (error) return std::unexpected(operation_error{"cannot create checkpoint directory"});

        for (const auto& map : maps) {
            auto parameters = stored_field_parameters(map, options);
            if (!parameters) return std::unexpected(parameters.error());
            auto checked = retained.save(map.frame.id, options.checkpoint_root / std::to_string(map.frame.id),
                *parameters, options.cpu_bytes);
            if (!checked) return std::unexpected(checked.error());
        }
    }
    const auto memory = retained.memory();
    report << "store CPU payload bytes " << memory.used[0] << ", GPU allocation bytes " << memory.used[2] << '\n'
        << "store pipeline wall ms " << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count() << '\n';
    output.report = std::move(report).str();
    return output;
}
}
