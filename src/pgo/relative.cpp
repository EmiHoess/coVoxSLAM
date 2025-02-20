#include <covoxslam/pgo.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

namespace covoxslam::pgo
{
operation_status append_odometry(graph_snapshot& graph, std::span<const submap> maps)
{
    if (maps.size() != graph.node_ids.size())
        return std::unexpected(operation_error{"odometry submap count differs from graph"});
    std::vector<relative_measurement> measurements;
     measurements.reserve(maps.empty() ? 0 : maps.size() - 1);

    for (std::size_t index = 0; index < maps.size(); ++index)
    {
        const auto& frame = maps[index].frame;

        if (frame.id != graph.node_ids[index] || frame.end_stamp_ns < frame.start_stamp_ns ||
            (index && frame.start_stamp_ns <= maps[index - 1].frame.start_stamp_ns))
            return std::unexpected(operation_error{"odometry submap order is invalid"});
        if (!index) continue;
        relative_measurement value;
        value.source_id = maps[index - 1].frame.id;
        value.target_id = frame.id;
        value.measured = relative_pose(maps[index - 1].frame.anchor, frame.anchor);
        value.diagnostic_id = index;
        measurements.push_back(value);
    }
    return append_relative(graph, measurements);
}
operation_status append_relative(graph_snapshot& graph, std::span<const relative_measurement> measurements)
{
    const auto valid = validate_graph(graph);
    if (!valid) return valid;
    std::vector<ceronoceres::RelativeFactor> factors;
     factors.reserve(measurements.size());

    for (const auto& value : measurements) {
        const auto left = std::find(graph.node_ids.begin(), graph.node_ids.end(), value.source_id);
        const auto right = std::find(graph.node_ids.begin(), graph.node_ids.end(), value.target_id);
        const auto& p = value.measured;

        if (left == graph.node_ids.end() || right == graph.node_ids.end() || left == right ||
            !std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) || !std::isfinite(p.yaw) ||
            !std::isfinite(value.loss.scale) || value.loss.scale <= 0 ||
            unsigned(value.loss.kind) > unsigned(ceronoceres::LossKind::Cauchy) ||
            !std::all_of(value.sqrt_information.begin(), value.sqrt_information.end(),
                [](double x) { return std::isfinite(x) &&
                    std::abs(x) <= std::sqrt(std::numeric_limits<double>::max() / 16); }) ||
            std::none_of(value.sqrt_information.begin(), value.sqrt_information.end(),
                [](double x) { return x != 0; }))
            return std::unexpected(operation_error{"invalid relative pose measurement"});
        factors.push_back({static_cast<std::size_t>(left - graph.node_ids.begin()),
            static_cast<std::size_t>(right - graph.node_ids.begin()),
            {p.x, p.y, p.z, ceronoceres::wrap_yaw(p.yaw)}, value.sqrt_information, value.loss, value.diagnostic_id});
    }
    graph.problem.relative_factors.insert(graph.problem.relative_factors.end(), factors.begin(), factors.end());
    return {};
}
}
