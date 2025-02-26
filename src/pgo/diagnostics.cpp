#include "diagnostics_internal.hpp"
#include <algorithm>
#include <limits>

namespace covoxslam::pgo
{
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
operation_result<ceronoceres::LinearizationSnapshot> linearize_graph(
    const ceronoceres::PreparedProblem& problem, std::span<const pose4> poses)
{
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
    auto values = detail::checked_poses(problem, poses);

    if (!values) return std::unexpected(values.error());
    const auto result = ceronoceres::evaluate(problem, *values);
    if (!detail::valid_cost(result))
        return std::unexpected(operation_error{"external graph evaluation produced an invalid cost"});
     return result;
}
}
