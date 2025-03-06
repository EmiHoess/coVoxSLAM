#include "diagnostics_internal.hpp"
#include <algorithm>

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
