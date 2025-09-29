#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/map_store/coordinator.hpp>
#include <covoxslam/pgo/registration.hpp>
#include <covoxslam/pgo/solver.hpp>
#include <covoxslam/pipeline/mapping_session.hpp>
#include <covoxslam/map_store/detail/geometry_capture.hpp>
#include <covoxslam/pgo/publication.hpp>
#include <covoxslam/pipeline/detail/optimization_store.hpp>
#include <chrono>
#include <sstream>
#if defined(covoxslam_has_cuda) && defined(covoxslam_has_urd)
#include <covoxslam/mapping/esdf/detail/gpu_band_esdf.hpp>
#include <covoxslam/mapping/esdf/detail/band_esdf_materialize.hpp>
#endif

namespace covoxslam::detail
{
inline operation_error stored_field_error(std::uint64_t id, const operation_error& error)
{
    return operation_error{"submap " + std::to_string(id) + " ESDF generation or publication: " + error.message};
}
inline operation_result<store::checkpoint_parameters> stored_field_parameters(
    const submap& map, const store_pipeline_options& options)
{
    const auto& config = map.config;

    if (!config.valid() || config.esdf_generator != options.esdf_generator || map.esdf.size() != map.tsdf.size())
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
inline operation_result<stored_pipeline_result> stored_pipeline(
    std::span<submap> maps, const store_pipeline_options& options, const pgo::options& solver, bool shadow,
    store::coordinated_store<storage::layout::pool_soa>& retained)
{
    using namespace store;
    constexpr auto l = storage::layout::pool_soa;
    const auto started = std::chrono::steady_clock::now();

    if (maps.empty() ||
        options.esdf_generator != esdf_backend::cuda_block_band_soa)
        return std::unexpected(operation_error{"completion requires retained SoA band fields"});
    std::vector<std::uint64_t> ids;
#if defined(covoxslam_has_pgo)
    std::vector<pgo_field_binding> bindings;
#endif
    bool has_esdf{};
    std::ostringstream report;

    for (auto& map : maps) {
        auto checked = [&]() -> operation_status {
            auto state = retained.status(map.frame.id);
            if (!state) return std::unexpected(state.error());

            if (state->state != lifecycle::finalized || state->cpu != state->content ||
                !state->content[0] || !state->content[1] || state->esdf_source != state->content[0])
                return std::unexpected(operation_error{"retained completion requires finalized current fields"});
            if (options.fields == stored_pgo_fields::resident && state->gpu != state->content)
                return std::unexpected(operation_error{"resident solver fields became stale before completion"});
            return {};
        }();
        if (!checked) return std::unexpected(stored_field_error(map.frame.id, checked.error()));
        ids.push_back(map.frame.id);
        if (ids.size() == 1) has_esdf = !map.esdf.empty();
        if (has_esdf != !map.esdf.empty())
            return std::unexpected(operation_error{"stored pipeline requires a consistent field selection"});
    }
    const auto imported = std::chrono::steady_clock::now();
    auto export_ready = validate_stored_export(maps, options);
    if (!export_ready) return std::unexpected(export_ready.error());
    stored_pipeline_result output;
    constexpr const char* field_modes[]{"copied", "borrowed", "resident"};
    report << "store layout soa, PGO fields "
        << field_modes[unsigned(options.fields)] << ", checkpoint format 1, field view API 1\n"
        << "mapping resident fields, deferred canonical readback, publication verification CPU\n";
    report << "store import ms " << std::chrono::duration<double, std::milli>(imported - started).count() << '\n';
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

            auto resident = retained.lease_resident(ids, 3, options.gpu_bytes);
            if (!resident) return std::unexpected(resident.error());
            auto gpu_fields = resident_pgo_fields<l, storage::schema::esdf>(*resident, bindings, geometry);
            if (!gpu_fields) return std::unexpected(gpu_fields.error());
            report << "PGO uses retained device fields without TSDF or ESDF upload\n";
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
