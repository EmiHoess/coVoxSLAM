#include <covoxslam/pgo/solver.hpp>
#include "contracts.hpp"
#include <cmath>

#if defined(covoxslam_has_pgo)
#include "solver_internal.hpp"
#include <abseil/field_views.hpp>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <tuple>

namespace covoxslam::pgo
{
operation_result<abseil::native_lm_options> configure_linear(
    linear_options linear, abseil::native_lm_options lm)
{
    lm.pcg_maximum_iterations = linear.maximum_iterations;
    lm.pcg_tolerance = linear.tolerance;
    const auto valid = validate_lm(lm);

    if (!valid) return std::unexpected(valid.error());
    return lm;
}
operation_status validate_solve_result(const abseil::solve_result& result,
    std::size_t nodes, abseil::backend expected_backend)
{
    if (result.selected_backend != expected_backend ||
        unsigned(result.stop) > unsigned(abseil::termination::numerical_breakdown))
        return std::unexpected(operation_error{"unexpected native solver result identity"});
    bool eligible{};
    switch (result.stop) {
    case abseil::termination::converged_cost:
    case abseil::termination::converged_gradient:
    case abseil::termination::converged_update:
    case abseil::termination::maximum_iterations: eligible = true; break;
    case abseil::termination::no_free_nodes:
    case abseil::termination::maximum_time:
    case abseil::termination::rejection_limit:
    case abseil::termination::singular_graph:
    case abseil::termination::numerical_breakdown: break;
    }

    if (result.solution_usable && !eligible)
        return std::unexpected(operation_error{"native solver failure cannot publish poses"});

    if (result.solution_usable && (result.poses.size() != nodes || !detail::valid_cost(result.initial_cost) ||
        !detail::valid_cost(result.final_cost) || result.final_cost.total() >
            result.initial_cost.total() + objective_convergence_tolerance *
                std::max(1.0, result.initial_cost.total())))
        return std::unexpected(operation_error{"invalid native solver pose count or objective"});

    for (const auto& p : result.poses)
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) || !std::isfinite(p.yaw))
            return std::unexpected(operation_error{"native solver returned a nonfinite pose"});
    return {};
}
operation_result<std::vector<pose4>> solution_poses(const abseil::solve_result& result, std::size_t nodes)
{
    const auto valid = validate_solve_result(result, nodes, result.selected_backend);
    if (!valid) return std::unexpected(valid.error());

    if (!result.solution_usable)
        return std::unexpected(operation_error{"native solver did not return a usable solution"});
    std::vector<pose4> poses;
     poses.reserve(nodes);

    for (const auto& p : result.poses) poses.push_back({p.x, p.y, p.z, abseil::wrap_yaw(p.yaw)});
    return poses;
}
operation_result<solve_statistics> summarize_solve(const abseil::solve_result& result)
{
    const auto valid = validate_solve_result(result, result.poses.size(), result.selected_backend);
    if (!valid) return std::unexpected(valid.error());
    solve_statistics report;
    report.termination = result.stop;
    report.usable = result.solution_usable;
    report.initial_cost = result.initial_cost;
    report.final_cost = result.final_cost;
    switch (result.stop) {
    case abseil::termination::converged_cost:
    case abseil::termination::converged_gradient:
    case abseil::termination::converged_update: report.converged = true; break;
    default: break;
    }

    for (const auto& iteration : result.iterations)
    {
        report.accepted_steps += iteration.accepted;
        report.rejected_steps += !iteration.accepted;
        report.linear_iterations += iteration.linear_iterations;
    }
    return report;
}
operation_status validate_lm(const abseil::native_lm_options& value)
{
    const double positive[]{value.initial_damping, value.minimum_damping, value.maximum_damping,
        value.maximum_translation_step_m, value.maximum_yaw_step_rad, value.pcg_tolerance};
    const double nonnegative[]{value.minimum_cost_change, value.minimum_gradient_norm, value.minimum_update_norm};

    if (!value.maximum_rejected_steps || value.maximum_rejected_steps > maximum_solver_iterations ||
        !value.pcg_maximum_iterations || value.pcg_maximum_iterations > maximum_solver_iterations ||
        value.initial_damping < value.minimum_damping || value.maximum_damping < value.initial_damping ||
        !std::all_of(std::begin(positive), std::end(positive), [](double x) { return std::isfinite(x) && x > 0; }) ||
        !std::all_of(std::begin(nonnegative), std::end(nonnegative), [](double x) { return std::isfinite(x) && x >= 0; }))
        return std::unexpected(operation_error{"invalid native LM settings"});
    return {};
}
operation_result<abseil::solve_options> cpu_solve_options(
    const options& options, const abseil::native_lm_options& lm)
{
    const auto valid = validate_options(options);
    if (!valid) return std::unexpected(valid.error());
    const auto valid_lm = validate_lm(lm);
    if (!valid_lm) return std::unexpected(valid_lm.error());

    if (!options.enabled || options.solver != backend::native_cpu)
        return std::unexpected(operation_error{"native CPU solve requires enabled CPU settings"});
    abseil::solve_options result;
    result.selected_backend = abseil::backend::native_cpu;
    result.common = {options.maximum_iterations, options.maximum_seconds};
    result.specific = abseil::native_cpu_options{lm};
    return result;
}
operation_result<abseil::solve_result> solve_cpu(const abseil::prepared_problem& problem,
    const options& options, const abseil::native_lm_options& lm)
{
    auto configured = cpu_solve_options(options, lm);
    if (!configured) return std::unexpected(configured.error());

    if (problem.empty()) return std::unexpected(operation_error{"cannot solve an empty prepared graph"});
    auto result = abseil::solve(problem, *configured);
    if (!result) return std::unexpected(operation_error{result.error().message});
    const auto valid = validate_solve_result(*result, problem.node_count(), configured->selected_backend);
    if (!valid) return std::unexpected(valid.error());
    return std::move(*result);
}
}

namespace covoxslam::pgo
{
operation_result<residual_comparison> compare_residuals(const residual_report& cpu, const residual_report& cuda)
{
    if (cpu.solver != abseil::backend::native_cpu || cuda.solver != abseil::backend::native_cuda_cold ||
        !detail::valid_cost(cpu.final) || !detail::valid_cost(cuda.final))
        return std::unexpected(operation_error{"residual comparison requires CPU and CUDA native reports"});
    residual_comparison result;
    result.final_difference = {cuda.final.relative - cpu.final.relative, cuda.final.absolute - cpu.final.absolute,
        cuda.final.registration - cpu.final.registration};
    result.same_admission = cpu.registration.accepted_rows == cuda.registration.accepted_rows &&
        cpu.registration.rejections == cuda.registration.rejections;
    return result;
}
operation_result<residual_report> report_residuals(const abseil::solve_result& result)
{
    const auto valid = validate_solve_result(result, result.poses.size(), result.selected_backend);
    if (!valid) return std::unexpected(valid.error());

    if ((result.selected_backend != abseil::backend::native_cpu && result.selected_backend != abseil::backend::native_cuda_cold) ||
        !detail::valid_cost(result.initial_cost) || !detail::valid_cost(result.final_cost))
        return std::unexpected(operation_error{"invalid native residual report"});
    return residual_report{result.selected_backend, result.initial_cost, result.final_cost,
        result.registration, result.solution_usable};
}
operation_status validate_linearization(const abseil::linearization_snapshot& value, std::size_t active_nodes)
{
    if (active_nodes > maximum_pose_graph_nodes || value.gradient.size() != active_nodes * pose_dimension ||
        value.sparse_blocks.size() > std::numeric_limits<std::size_t>::max() / normal_equation_block_cardinality ||
        value.hessian_blocks.size() != value.sparse_blocks.size() * normal_equation_block_cardinality ||
        !detail::valid_cost(value.cost))
        return std::unexpected(operation_error{"invalid external equation dimensions or cost"});
    const auto finite = [](double x) { return std::isfinite(x); };

    if (!std::all_of(value.gradient.begin(), value.gradient.end(), finite) ||
        !std::all_of(value.hessian_blocks.begin(), value.hessian_blocks.end(), finite))
        return std::unexpected(operation_error{"nonfinite external normal equations"});
    std::vector<bool> diagonal(active_nodes);
    std::optional<std::array<std::int32_t, 2>> previous;

    for (const auto& block : value.sparse_blocks)
    {
        if (block[0] < 0 || block[1] < block[0] || std::size_t(block[1]) >= active_nodes ||
            (previous && block <= *previous))
            return std::unexpected(operation_error{"invalid external sparse block sequence"});
        if (block[0] == block[1]) diagonal[block[0]] = true;
        previous = block;
    }

    if (std::find(diagonal.begin(), diagonal.end(), false) != diagonal.end())
        return std::unexpected(operation_error{"external normal equations omit a diagonal block"});
    return {};
}
operation_result<equation_statistics> summarize_linearization(
    const abseil::linearization_snapshot& snapshot, std::size_t active_nodes)
{
    const auto valid = validate_linearization(snapshot, active_nodes);
    if (!valid) return std::unexpected(valid.error());
    equation_statistics result;
    result.active_nodes = active_nodes;
    result.scalar_dimension = snapshot.gradient.size();
    result.sparse_blocks = snapshot.sparse_blocks.size();
    result.cost = snapshot.cost;
    result.registration = snapshot.registration;

    for (const auto value : snapshot.gradient)
        result.maximum_absolute_gradient = std::max(result.maximum_absolute_gradient, std::abs(value));
    return result;
}
operation_result<abseil::linearization_snapshot> linearize_graph(
    const abseil::prepared_problem& problem, std::span<const pose4> poses)
{
    if (!abseil::cpu_fields_accessible(problem))
        return std::unexpected(operation_error{"host linearization requires host field allocations"});
    auto values = detail::checked_poses(problem, poses);

    if (!values) return std::unexpected(values.error());
    auto result = abseil::linearize(problem, *values);
    if (!result) return std::unexpected(operation_error{result.error().message});
    const auto valid = validate_linearization(*result, problem.active_node_count());
    if (!valid) return std::unexpected(valid.error());
    return std::move(*result);
}
operation_result<abseil::cost_breakdown> evaluate_graph(
    const abseil::prepared_problem& problem, std::span<const pose4> poses)
{
    if (!abseil::cpu_fields_accessible(problem))
        return std::unexpected(operation_error{"host evaluation requires host field allocations"});
    auto values = detail::checked_poses(problem, poses);

    if (!values) return std::unexpected(values.error());
    const auto result = abseil::evaluate(problem, *values);
    if (!result) return std::unexpected(operation_error{result.error().message});
    if (!detail::valid_cost(result->cost))
        return std::unexpected(operation_error{"external graph evaluation produced an invalid cost"});
    return result->cost;
}
}

namespace covoxslam::pgo
{
operation_status write_iteration_report(std::ostream& output, const abseil::solve_result& result)
{
    const auto summary = summarize_solve(result);
    if (!summary) return std::unexpected(summary.error());
    std::ostringstream text;
    text << std::setprecision(std::numeric_limits<double>::max_digits10)
         << "iteration accepted before after damping gradient update linear_iterations seconds linear_seconds\n";

    for (const auto& step : result.iterations)
    {
        if (!std::isfinite(step.objective_before) || step.objective_before < 0 ||
            (step.accepted && (!std::isfinite(step.objective_after) || step.objective_after < 0)) ||
            !std::isfinite(step.damping) || step.damping <= 0 ||
            !std::isfinite(step.iteration_seconds) || step.iteration_seconds < 0 ||
            !std::isfinite(step.linear_solver_seconds) || step.linear_solver_seconds < 0)
            return std::unexpected(operation_error{"invalid reported native iteration"});
        text << step.iteration << ' ' << step.accepted << ' ' << step.objective_before << ' '
            << step.objective_after << ' ' << step.damping << ' ' << step.gradient_norm << ' '
            << step.update_norm << ' ' << step.linear_iterations << ' ' << step.iteration_seconds << ' '
            << step.linear_solver_seconds << '\n';
    }
    output << text.str();
    if (!output) return std::unexpected(operation_error{"cannot write native iteration diagnostics"});
    return {};
}
operation_status write_solver_report(std::ostream& output, const abseil::solve_result& result)
{
    const auto residuals = report_residuals(result);
    if (!residuals) return std::unexpected(residuals.error());
    const auto summary = summarize_solve(result);
    if (!summary) return std::unexpected(summary.error());
    const auto& t = result.telemetry;
    std::ostringstream text;
    text << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "solver " << abseil::backend_name(result.selected_backend) << '\n'
        << "termination " << abseil::termination_name(result.stop) << '\n'
        << "usable " << summary->usable << " converged " << summary->converged << '\n'
        << "relative_cost " << residuals->initial.relative << ' ' << residuals->final.relative << '\n'
        << "absolute_cost " << residuals->initial.absolute << ' ' << residuals->final.absolute << '\n'
        << "registration_cost " << residuals->initial.registration << ' ' << residuals->final.registration << '\n'
        << "registration_rows " << residuals->registration.accepted_rows << '\n'
        << "accepted_steps " << summary->accepted_steps << " rejected_steps " << summary->rejected_steps << '\n'
        << "linear_iterations " << summary->linear_iterations << '\n'
        << "validation_ms " << t.validation_ms << " preparation_ms " << t.preparation_ms << '\n'
        << "upload_ms " << t.upload_ms << " evaluation_ms " << t.evaluation_ms << '\n'
        << "assembly_ms " << t.assembly_ms << " linear_solver_ms " << t.linear_solver_ms << '\n'
        << "update_ms " << t.update_ms << " synchronization_ms " << t.synchronization_ms << '\n'
        << "download_ms " << t.download_ms << " total_ms " << t.total_ms << '\n'
        << "upload_bytes " << t.host_to_device_bytes << " download_bytes " << t.device_to_host_bytes << '\n'
        << "peak_device_bytes " << t.peak_device_bytes << " synchronizations " << t.synchronization_count << '\n';

    for (std::size_t index = 0; index < residuals->registration.rejections.size(); ++index)
        text << "registration_rejection " << index << ' ' << residuals->registration.rejections[index] << '\n';
    output << text.str();

    if (!output) return std::unexpected(operation_error{"cannot write solver diagnostics"});
    return {};
}
}

namespace covoxslam::pgo
{
operation_result<solve_snapshot> freeze_field_graph(graph_snapshot graph,
    std::shared_ptr<const abseil::field_snapshot> fields)
{
    auto extents = measure_snapshot(graph);
    if (!extents) return std::unexpected(extents.error());
    if (!fields || fields->device != -1)
        return std::unexpected(operation_error{"publication verification requires host fields"});
    auto prepared = abseil::prepare_fields(graph.problem, std::move(fields));
    if (!prepared) return std::unexpected(operation_error{prepared.error().message});
    return solve_snapshot{std::make_shared<const graph_snapshot>(std::move(graph)),
        std::move(*prepared), *extents};
}
operation_result<abseil::solve_options> cuda_solve_options(
    const options& options, const abseil::native_lm_options& lm)
{
    const auto valid = validate_options(options);
    if (!valid) return std::unexpected(valid.error());
    const auto valid_lm = validate_lm(lm);
    if (!valid_lm) return std::unexpected(valid_lm.error());

    if (!options.enabled || options.solver != backend::cuda_cold || !options.memory_budget_bytes)
        return std::unexpected(operation_error{"cold CUDA solve requires enabled settings and a memory budget"});
    abseil::solve_options result;
    result.selected_backend = abseil::backend::native_cuda_cold;
    result.common = {options.maximum_iterations, options.maximum_seconds};
    abseil::native_cuda_cold_options native;
    native.lm = lm;
    native.device_index = options.device;
    native.memory_budget_bytes = options.memory_budget_bytes;
    result.specific = native;
    return result;
}
operation_result<abseil::solver_telemetry> cuda_telemetry(const abseil::solve_result& result)
{
    const auto valid = validate_solve_result(result, result.poses.size(), abseil::backend::native_cuda_cold);
    if (!valid) return std::unexpected(valid.error());
    return result.telemetry;
}
operation_result<snapshot_extents> measure_snapshot(const graph_snapshot& graph)
{
    const auto valid = validate_graph(graph);
    if (!valid) return std::unexpected(valid.error());
    const auto& problem = graph.problem;
    snapshot_extents result;
    result.nodes = problem.nodes.size();
    result.relative_factors = problem.relative_factors.size();
    result.absolute_factors = problem.absolute_factors.size();
    result.groups = problem.registration_groups.size();
    result.samples = problem.registration_samples.size();
    result.tsdf_tiles = problem.tsdf_tiles.size();
    result.esdf_tiles = problem.esdf_tiles.size();
    const std::pair<std::size_t, std::size_t> arrays[]{
        {problem.nodes.size(), sizeof(abseil::node)},
        {graph.node_ids.size(), sizeof(std::uint64_t)},
        {problem.relative_factors.size(), sizeof(abseil::relative_factor)},
        {problem.absolute_factors.size(), sizeof(abseil::absolute_factor)},
        {problem.registration_groups.size(), sizeof(abseil::registration_group)},
        {problem.registration_samples.size(), sizeof(abseil::registration_sample)},
        {problem.volumes.size(), sizeof(abseil::field_volume)},
        {problem.tsdf_tiles.size(), sizeof(abseil::tsdf_tile)},
        {problem.esdf_tiles.size(), sizeof(abseil::esdf_tile)}};

    for (const auto& [count, size] : arrays)
    {
        if (count > std::size_t(std::numeric_limits<std::int32_t>::max()) ||
            count > (std::numeric_limits<std::uint64_t>::max() - result.record_bytes) / size)
            return std::unexpected(operation_error{"graph snapshot exceeds index or byte bounds"});
        result.record_bytes += count * size;
    }
    const auto tile_extent_valid = [](const auto& tiles) {
        constexpr auto lower = std::numeric_limits<std::int32_t>::min() / tile_edge;
        constexpr auto upper = (std::numeric_limits<std::int32_t>::max() - tile_edge) / tile_edge;

        for (const auto& tile : tiles)
            if (tile.x < lower || tile.y < lower || tile.z < lower ||
                tile.x > upper || tile.y > upper || tile.z > upper) return false;
        return true;
    };

    if (!tile_extent_valid(problem.tsdf_tiles) || !tile_extent_valid(problem.esdf_tiles))
        return std::unexpected(operation_error{"graph tile coordinates exceed voxel index bounds"});
    return result;
}
operation_result<solve_snapshot> freeze_graph(graph_snapshot graph)
{
    const auto extents = measure_snapshot(graph);
    if (!extents) return std::unexpected(extents.error());
    const auto less = [](const auto& left, const auto& right) {
        return std::tie(left.volume, left.x, left.y, left.z) < std::tie(right.volume, right.x, right.y, right.z);
    };
    std::sort(graph.problem.tsdf_tiles.begin(), graph.problem.tsdf_tiles.end(), less);
    std::sort(graph.problem.esdf_tiles.begin(), graph.problem.esdf_tiles.end(), less);
    auto prepared = prepare_graph(graph);

    if (!prepared) return std::unexpected(prepared.error());
    return solve_snapshot{std::make_shared<const graph_snapshot>(std::move(graph)), std::move(*prepared), *extents};
}
operation_status validate_snapshot_result(const solve_snapshot& snapshot, const abseil::solve_result& result)
{
    const auto valid = validate_solve_result(result, snapshot.extents().nodes, result.selected_backend);
    if (!valid) return valid;

    if (result.selected_backend != abseil::backend::native_cpu && result.selected_backend != abseil::backend::native_cuda_cold)
        return std::unexpected(operation_error{"snapshot result did not use a native solver"});
    if (!result.solution_usable) return {};

    for (std::size_t index = 0; index < snapshot.extents().nodes; ++index)
    {
        const auto& node = snapshot.graph().problem.nodes[index];
        if (!node.constant) continue;
        const auto& value = result.poses[index];
        const auto& initial = node.initial;

        if (!same_pose(value, initial))
            return std::unexpected(operation_error{"native solver changed a fixed graph pose"});
    }
    const auto& t = result.telemetry;
    const double times[]{t.validation_ms, t.preparation_ms, t.upload_ms, t.evaluation_ms,
        t.assembly_ms, t.linear_solver_ms, t.update_ms, t.synchronization_ms, t.download_ms, t.total_ms};

    for (double time : times)
        if (!std::isfinite(time) || time < 0)
            return std::unexpected(operation_error{"native solver returned invalid timing data"});
    return {};
}
operation_result<abseil::solve_result> solve_cuda(const solve_snapshot& snapshot,
    const options& options, const abseil::native_lm_options& lm)
{
    const auto configured = cuda_solve_options(options, lm);
    if (!configured) return std::unexpected(configured.error());
    auto checked = abseil::solve_cuda_checked(snapshot.prepared(), configured->common,
        std::get<abseil::native_cuda_cold_options>(configured->specific));

    if (!checked) return std::unexpected(operation_error{checked.error().message});
    auto result = std::move(*checked);
    const auto valid = validate_snapshot_result(snapshot, result);

    if (!valid) return std::unexpected(valid.error());
    return result;
}
operation_result<abseil::solve_result> solve_native(const solve_snapshot& snapshot,
    const options& options, const abseil::native_lm_options& lm)
{
    operation_result<abseil::solve_result> result = std::unexpected(operation_error{"invalid native solver selection"});
    switch (options.solver) {
    case backend::native_cpu: result = solve_cpu(snapshot.prepared(), options, lm); break;
    case backend::cuda_cold: result = solve_cuda(snapshot, options, lm); break;
    }
    if (!result) return result;
    const auto valid = validate_snapshot_result(snapshot, *result);
    if (!valid) return std::unexpected(valid.error());
    return result;
}
}
#endif

namespace covoxslam::pgo
{
operation_status validate_options(const options& value)
{
    if (!value.enabled) return {};
    if (!available(value.solver)) return std::unexpected(operation_error{"selected native PGO backend is unavailable"});
    if (!value.maximum_iterations || value.maximum_iterations > maximum_solver_iterations ||
        !std::isfinite(value.maximum_seconds) || value.maximum_seconds <= 0 || value.device < 0)
        return std::unexpected(operation_error{"invalid PGO execution limits"});
    return {};
}
bool available(backend solver) noexcept
{
#if defined(covoxslam_has_pgo)
    switch (solver) {
    case backend::native_cpu: return abseil::backend_available(abseil::backend::native_cpu);
    case backend::cuda_cold: return abseil::backend_available(abseil::backend::native_cuda_cold);
    }
#else
    static_cast<void>(solver);
#endif
    return false;
}
}
