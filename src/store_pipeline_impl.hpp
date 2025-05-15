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

namespace covoxslam::detail
{
template<storage::layout l> operation_result<stored_pipeline_result> stored_pipeline(
    std::span<submap> maps, const store_pipeline_options& options, const pgo::options& solver, bool shadow)
{
    using namespace store;
    const auto started = std::chrono::steady_clock::now();
    residency_policy policy;
    policy.limits = {options.cpu_bytes, 128U * 1024U * 1024U, options.gpu_bytes};
    coordinated_store<l> retained(policy);
    std::vector<std::uint64_t> ids;
    std::vector<pgo_field_binding> bindings;
    bool has_esdf{};

    for (const auto& map : maps) {
        auto checked = import_submap(retained, map);
        if (!checked) return std::unexpected(checked.error());
        ids.push_back(map.frame.id);
        if (ids.size() == 1) has_esdf = !map.esdf.empty();
        if (has_esdf != !map.esdf.empty())
            return std::unexpected(operation_error{"stored pipeline requires a consistent field selection"});
    }
    const auto imported = std::chrono::steady_clock::now();
    stored_pipeline_result output;
    std::ostringstream report;
    report << "store import ms " << std::chrono::duration<double, std::milli>(imported - started).count() << '\n';
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

        for (std::size_t i = 0; i < ids.size(); ++i)
            bindings.push_back({ids[i], i, host->find(ids[i])->record.versions});
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
                checked = retained.synchronize_gpu(id, 3, options.cpu_bytes);
                if (!checked) return std::unexpected(checked.error());
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
            const auto& config = map.config;
            const checkpoint_parameters parameters{config.voxel_size, config.truncation_distance,
                config.maximum_weight, config.esdf_maximum_distance_m, std::uint64_t(config.esdf_generator)};
            auto checked = retained.save(map.frame.id, options.checkpoint_root / std::to_string(map.frame.id),
                parameters, options.cpu_bytes);
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
