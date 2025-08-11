#pragma once
#include <covoxslam/core/geometry.hpp>
#include <covoxslam/core/result.hpp>
#include <covoxslam/pgo.hpp>
#include <cmath>

namespace covoxslam::pgo::detail
{
inline operation_result<std::vector<ceronoceres::Pose4>> checked_poses(
    const ceronoceres::PreparedProblem& problem, std::span<const pose4> poses)
{
    if (problem.empty() || poses.size() != problem.node_count())
        return std::unexpected(operation_error{"pose count differs from prepared graph"});
    std::vector<ceronoceres::Pose4> result;
     result.reserve(poses.size());

    for (const auto& p : poses)
    {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) || !std::isfinite(p.yaw))
            return std::unexpected(operation_error{"nonfinite graph evaluation pose"});
        result.push_back({p.x, p.y, p.z, ceronoceres::wrap_yaw(p.yaw)});
    }
    return result;
}
inline bool valid_cost(const ceronoceres::CostBreakdown& cost)
{
    return std::isfinite(cost.relative) && cost.relative >= 0 && std::isfinite(cost.absolute) &&
        cost.absolute >= 0 && std::isfinite(cost.registration) && cost.registration >= 0 && std::isfinite(cost.total());
}
}
