#pragma once

#include <cmath>
#include <cstddef>
#if defined(covoxslam_has_pgo)
#include <ceronoceres/ceronoceres.hpp>
#endif

namespace covoxslam::pgo
{
inline constexpr std::size_t maximum_pose_graph_nodes = 1U << 20U;
inline constexpr std::size_t maximum_solver_iterations = 1'000'000;
inline constexpr std::size_t pose_dimension = 4;
inline constexpr std::size_t normal_equation_block_cardinality = 16;
inline constexpr double same_pose_yaw_tolerance = 1.0e-12;
inline constexpr double objective_convergence_tolerance = 1.0e-10;
inline constexpr double measured_comparison_tolerance = 1.0e-5;
inline constexpr double publication_comparison_tolerance = 1.0e-5;
inline constexpr double registration_additive_tolerance = 1.0e-10;

#if defined(covoxslam_has_pgo)
template<class left_pose, class right_pose>
bool same_pose(const left_pose& left, const right_pose& right) noexcept
{
    return left.x == right.x && left.y == right.y && left.z == right.z &&
        !(std::abs(ceronoceres::wrap_yaw(left.yaw - right.yaw)) > same_pose_yaw_tolerance);
}
#endif
}
