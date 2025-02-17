#include <covoxslam/pgo.hpp>
#include <cmath>

namespace covoxslam::pgo
{
operation_result<graph_snapshot> make_graph(std::span<const graph_node> nodes)
{
    if (nodes.empty() || nodes.size() > 1048576)
        return std::unexpected(operation_error{"pose graph node count exceeds bounds"});
    graph_snapshot result;
    result.problem.nodes.reserve(nodes.size());
     result.node_ids.reserve(nodes.size());

    for (std::size_t index = 0; index < nodes.size(); ++index) {
        const auto& node = nodes[index];
        const auto& p = node.initial;
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) || !std::isfinite(p.yaw))
            return std::unexpected(operation_error{"pose graph has a nonfinite initial pose"});
        result.problem.nodes.push_back({node.id, {p.x, p.y, p.z, ceronoceres::wrap_yaw(p.yaw)},
            index == 0 || node.constant});
        result.node_ids.push_back(node.id);
    }
    return result;
}
}
