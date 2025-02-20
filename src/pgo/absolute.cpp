#include <covoxslam/pgo.hpp>
#include <algorithm>
#include <cmath>

namespace covoxslam::pgo
{
operation_status append_absolute(graph_snapshot& graph, std::uint64_t reference_id,
    pose4 reference_world, std::span<const absolute_measurement> measurements)
{
    const auto valid = validate_graph(graph);
    if (!valid) return valid;

    if (std::none_of(measurements.begin(), measurements.end(),
        [](const absolute_measurement& value) { return value.world_pose.has_value(); })) return {};
    const auto reference = std::find(graph.node_ids.begin(), graph.node_ids.end(), reference_id);

    if (reference == graph.node_ids.end() || !graph.problem.nodes[reference - graph.node_ids.begin()].constant ||
        !std::isfinite(reference_world.x) || !std::isfinite(reference_world.y) ||
        !std::isfinite(reference_world.z) || !std::isfinite(reference_world.yaw))
        return std::unexpected(operation_error{"invalid absolute measurement reference"});
    std::vector<relative_measurement> relative;
     relative.reserve(measurements.size());

    for (const auto& value : measurements) {

        if (!value.world_pose) continue;
        relative.push_back({reference_id, value.node_id, relative_pose(reference_world, *value.world_pose),
            value.sqrt_information, value.loss, value.diagnostic_id});
    }
    graph_snapshot staging;
    staging.node_ids = graph.node_ids;
    staging.problem.nodes = graph.problem.nodes;
    const auto converted = append_relative(staging, relative);
    if (!converted) return converted;
    std::vector<ceronoceres::AbsoluteFactor> factors;
    factors.reserve(relative.size());

    for (const auto& value : staging.problem.relative_factors)
        factors.push_back({value.node_i, value.node_j, value.measured,
            value.sqrt_information, value.loss, value.diagnostic_id});
    graph.problem.absolute_factors.insert(graph.problem.absolute_factors.end(), factors.begin(), factors.end());
    return {};
}
}
