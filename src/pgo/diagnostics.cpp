#include "diagnostics_internal.hpp"

namespace covoxslam::pgo
{
operation_result<ceronoceres::LinearizationSnapshot> linearize_graph(
    const ceronoceres::PreparedProblem& problem, std::span<const pose4> poses)
{
    auto values = detail::checked_poses(problem, poses);

    if (!values) return std::unexpected(values.error());
    auto result = ceronoceres::linearize(problem, *values);
    if (!detail::valid_cost(result.cost))
        return std::unexpected(operation_error{"external linearization produced an invalid cost"});
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
