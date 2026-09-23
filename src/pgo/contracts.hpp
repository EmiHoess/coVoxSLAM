#pragma once

#include <cstddef>

namespace covoxslam::pgo
{
inline constexpr std::size_t maximum_pose_graph_nodes = 1U << 20U;
inline constexpr std::size_t maximum_solver_iterations = 1'000'000;
inline constexpr std::size_t pose_dimension = 4;
inline constexpr std::size_t normal_equation_block_cardinality = 16;
}
