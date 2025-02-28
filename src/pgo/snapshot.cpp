#include <covoxslam/pgo_snapshot.hpp>
#include <algorithm>
#include <tuple>

namespace covoxslam::pgo
{
operation_result<snapshot_extents> measure_snapshot(const graph_snapshot& graph)
{
    const auto valid = validate_graph(graph);
    if (!valid) return std::unexpected(valid.error());
    const auto& problem = graph.problem;
    snapshot_extents result;
    result.nodes = problem.nodes.size();
    result.relative_factors = problem.relative_factors.size();
    result.absolute_factors = problem.absolute_factors.size();
    result.groups = problem.registration_groups.size();
    result.samples = problem.registration_samples.size();
    result.tsdf_tiles = problem.tsdf_tiles.size();
    result.esdf_tiles = problem.esdf_tiles.size();
    return result;
}
operation_result<solve_snapshot> freeze_graph(graph_snapshot graph)
{
    const auto extents = measure_snapshot(graph);
    if (!extents) return std::unexpected(extents.error());
    const auto less = [](const auto& left, const auto& right) {
        return std::tie(left.volume, left.x, left.y, left.z) < std::tie(right.volume, right.x, right.y, right.z);
    };
    std::sort(graph.problem.tsdf_tiles.begin(), graph.problem.tsdf_tiles.end(), less);
    std::sort(graph.problem.esdf_tiles.begin(), graph.problem.esdf_tiles.end(), less);
    auto prepared = prepare_graph(graph);

    if (!prepared) return std::unexpected(prepared.error());
    return solve_snapshot{std::make_shared<const graph_snapshot>(std::move(graph)), std::move(*prepared), *extents};
}
}
