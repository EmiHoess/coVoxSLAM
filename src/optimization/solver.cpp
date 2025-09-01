#include <covoxslam/optimization/solver.hpp>
#include <cmath>

#if defined(covoxslam_has_pgo)
#include "solver_internal.hpp"
#include <ceronoceres/field_views.hpp>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>

namespace covoxslam::pgo
{
operation_result<ceronoceres::NativeLmOptions> configure_linear(
    linear_options linear, ceronoceres::NativeLmOptions lm)
{
    lm.pcg_maximum_iterations = linear.maximum_iterations;
    lm.pcg_tolerance = linear.tolerance;
    const auto valid = validate_lm(lm);

    if (!valid) return std::unexpected(valid.error());
    return lm;
}
operation_status validate_solve_result(const ceronoceres::SolveResult& result,
    std::size_t nodes, ceronoceres::Backend expected_backend)
{
    if (result.backend != expected_backend ||
        unsigned(result.termination) > unsigned(ceronoceres::Termination::BackendFailure))
        return std::unexpected(operation_error{"unexpected native solver result identity"});
    bool eligible{};
    switch (result.termination) {
    case ceronoceres::Termination::ConvergedCost:
    case ceronoceres::Termination::ConvergedGradient:
    case ceronoceres::Termination::ConvergedUpdate:
    case ceronoceres::Termination::MaximumIterations: eligible = true; break;
    case ceronoceres::Termination::NoFreeNodes:
    case ceronoceres::Termination::MaximumTime:
    case ceronoceres::Termination::RejectionLimit:
    case ceronoceres::Termination::SingularGraph:
    case ceronoceres::Termination::InvalidProblem:
    case ceronoceres::Termination::BackendUnavailable:
    case ceronoceres::Termination::BackendFailure: break;
    }

    if (result.solution_usable && !eligible)
        return std::unexpected(operation_error{"native solver failure cannot publish poses"});

    if (result.solution_usable && (result.poses.size() != nodes || !detail::valid_cost(result.initial_cost) ||
        !detail::valid_cost(result.final_cost) || result.final_cost.total() >
            result.initial_cost.total() + 1.0e-10 * std::max(1.0, result.initial_cost.total())))
        return std::unexpected(operation_error{"invalid native solver pose count or objective"});

    for (const auto& p : result.poses)
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) || !std::isfinite(p.yaw))
            return std::unexpected(operation_error{"native solver returned a nonfinite pose"});
    return {};
}
operation_result<std::vector<pose4>> solution_poses(const ceronoceres::SolveResult& result, std::size_t nodes)
{
    const auto valid = validate_solve_result(result, nodes, result.backend);
    if (!valid) return std::unexpected(valid.error());

    if (!result.solution_usable)
        return std::unexpected(operation_error{"native solver did not return a usable solution"});
    std::vector<pose4> poses;
     poses.reserve(nodes);

    for (const auto& p : result.poses) poses.push_back({p.x, p.y, p.z, ceronoceres::wrap_yaw(p.yaw)});
    return poses;
}
operation_result<solve_statistics> summarize_solve(const ceronoceres::SolveResult& result)
{
    const auto valid = validate_solve_result(result, result.poses.size(), result.backend);
    if (!valid) return std::unexpected(valid.error());
    solve_statistics report;
    report.termination = result.termination;
    report.usable = result.solution_usable;
    report.initial_cost = result.initial_cost;
    report.final_cost = result.final_cost;
    switch (result.termination) {
    case ceronoceres::Termination::ConvergedCost:
    case ceronoceres::Termination::ConvergedGradient:
    case ceronoceres::Termination::ConvergedUpdate: report.converged = true; break;
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
operation_status validate_lm(const ceronoceres::NativeLmOptions& value)
{
    const double positive[]{value.initial_damping, value.minimum_damping, value.maximum_damping,
        value.maximum_translation_step_m, value.maximum_yaw_step_rad, value.pcg_tolerance};
    const double nonnegative[]{value.minimum_cost_change, value.minimum_gradient_norm, value.minimum_update_norm};

    if (!value.maximum_rejected_steps || value.maximum_rejected_steps > 1000000 ||
        !value.pcg_maximum_iterations || value.pcg_maximum_iterations > 1000000 ||
        value.initial_damping < value.minimum_damping || value.maximum_damping < value.initial_damping ||
        !std::all_of(std::begin(positive), std::end(positive), [](double x) { return std::isfinite(x) && x > 0; }) ||
        !std::all_of(std::begin(nonnegative), std::end(nonnegative), [](double x) { return std::isfinite(x) && x >= 0; }))
        return std::unexpected(operation_error{"invalid native LM settings"});
    return {};
}
operation_result<ceronoceres::SolveOptions> cpu_solve_options(
    const options& options, const ceronoceres::NativeLmOptions& lm)
{
    const auto valid = validate_options(options);
    if (!valid) return std::unexpected(valid.error());
    const auto valid_lm = validate_lm(lm);
    if (!valid_lm) return std::unexpected(valid_lm.error());

    if (!options.enabled || options.solver != backend::native_cpu)
        return std::unexpected(operation_error{"native CPU solve requires enabled CPU settings"});
    ceronoceres::SolveOptions result;
    result.backend = ceronoceres::Backend::NativeCpu;
    result.common = {options.maximum_iterations, options.maximum_seconds};
    result.backend_options = ceronoceres::NativeCpuOptions{lm};
    return result;
}
operation_result<ceronoceres::SolveResult> solve_cpu(const ceronoceres::PreparedProblem& problem,
    const options& options, const ceronoceres::NativeLmOptions& lm)
{
    auto configured = cpu_solve_options(options, lm);
    if (!configured) return std::unexpected(configured.error());

    if (problem.empty()) return std::unexpected(operation_error{"cannot solve an empty prepared graph"});
    auto result = ceronoceres::solve(problem, *configured);
    const auto valid = validate_solve_result(result, problem.node_count(), configured->backend);
    if (!valid) return std::unexpected(valid.error());
    return result;
}
}

namespace covoxslam::pgo
{
operation_result<residual_comparison> compare_residuals(const residual_report& cpu, const residual_report& cuda)
{
    if (cpu.solver != ceronoceres::Backend::NativeCpu || cuda.solver != ceronoceres::Backend::NativeCudaCold ||
        !detail::valid_cost(cpu.final) || !detail::valid_cost(cuda.final))
        return std::unexpected(operation_error{"residual comparison requires CPU and CUDA native reports"});
    residual_comparison result;
    result.final_difference = {cuda.final.relative - cpu.final.relative, cuda.final.absolute - cpu.final.absolute,
        cuda.final.registration - cpu.final.registration};
    result.same_admission = cpu.registration.accepted_rows == cuda.registration.accepted_rows &&
        cpu.registration.rejections == cuda.registration.rejections;
    return result;
}
operation_result<residual_report> report_residuals(const ceronoceres::SolveResult& result)
{
    const auto valid = validate_solve_result(result, result.poses.size(), result.backend);
    if (!valid) return std::unexpected(valid.error());

    if ((result.backend != ceronoceres::Backend::NativeCpu && result.backend != ceronoceres::Backend::NativeCudaCold) ||
        !detail::valid_cost(result.initial_cost) || !detail::valid_cost(result.final_cost))
        return std::unexpected(operation_error{"invalid native residual report"});
    return residual_report{result.backend, result.initial_cost, result.final_cost,
        result.registration, result.solution_usable};
}
operation_status validate_linearization(const ceronoceres::LinearizationSnapshot& value, std::size_t active_nodes)
{
    if (active_nodes > 1048576 || value.gradient.size() != active_nodes * 4 ||
        value.sparse_blocks.size() > std::numeric_limits<std::size_t>::max() / 16 ||
        value.hessian_blocks.size() != value.sparse_blocks.size() * 16 || !detail::valid_cost(value.cost))
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
    const ceronoceres::LinearizationSnapshot& snapshot, std::size_t active_nodes)
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
operation_result<ceronoceres::LinearizationSnapshot> linearize_graph(
    const ceronoceres::PreparedProblem& problem, std::span<const pose4> poses)
{
    if (!ceronoceres::cpu_fields_accessible(problem))
        return std::unexpected(operation_error{"host linearization requires host field allocations"});
    auto values = detail::checked_poses(problem, poses);

    if (!values) return std::unexpected(values.error());
    auto result = ceronoceres::linearize(problem, *values);
    const auto valid = validate_linearization(result, problem.active_node_count());
    if (!valid) return std::unexpected(valid.error());
    return result;
}
operation_result<ceronoceres::CostBreakdown> evaluate_graph(
    const ceronoceres::PreparedProblem& problem, std::span<const pose4> poses)
{
    if (!ceronoceres::cpu_fields_accessible(problem))
        return std::unexpected(operation_error{"host evaluation requires host field allocations"});
    auto values = detail::checked_poses(problem, poses);

    if (!values) return std::unexpected(values.error());
    const auto result = ceronoceres::evaluate(problem, *values);
    if (!detail::valid_cost(result))
        return std::unexpected(operation_error{"external graph evaluation produced an invalid cost"});
     return result;
}
}

namespace covoxslam::pgo
{
operation_status write_iteration_report(std::ostream& output, const ceronoceres::SolveResult& result)
{
    const auto summary = summarize_solve(result);
    if (!summary) return std::unexpected(summary.error());
    std::ostringstream text;
    text << std::setprecision(17) << "iteration accepted before after damping gradient update linear_iterations seconds linear_seconds\n";

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
operation_status write_solver_report(std::ostream& output, const ceronoceres::SolveResult& result)
{
    const auto residuals = report_residuals(result);
    if (!residuals) return std::unexpected(residuals.error());
    const auto summary = summarize_solve(result);
    if (!summary) return std::unexpected(summary.error());
    const auto& t = result.telemetry;
    std::ostringstream text;
    text << std::setprecision(17)
        << "solver " << ceronoceres::backend_name(result.backend) << '\n'
        << "termination " << ceronoceres::termination_name(result.termination) << '\n'
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
#endif

namespace covoxslam::pgo
{
operation_status validate_options(const options& value)
{
    if (!value.enabled) return {};
    if (!available(value.solver)) return std::unexpected(operation_error{"selected native PGO backend is unavailable"});
    if (!value.maximum_iterations || value.maximum_iterations > 1000000 ||
        !std::isfinite(value.maximum_seconds) || value.maximum_seconds <= 0 || value.device < 0)
        return std::unexpected(operation_error{"invalid PGO execution limits"});
    return {};
}
bool available(backend solver) noexcept
{
#if defined(covoxslam_has_pgo)
    switch (solver) {
    case backend::native_cpu: return ceronoceres::backend_available(ceronoceres::Backend::NativeCpu);
    case backend::cuda_cold: return ceronoceres::backend_available(ceronoceres::Backend::NativeCudaCold);
    }
#else
    static_cast<void>(solver);
#endif
    return false;
}
}
