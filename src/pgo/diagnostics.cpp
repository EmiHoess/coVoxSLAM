#include "diagnostics_internal.hpp"
#include <ceronoceres/field_views.hpp>
#include <algorithm>
#include <limits>

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
