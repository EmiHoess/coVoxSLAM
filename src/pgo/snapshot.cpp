#include <covoxslam/pgo_snapshot.hpp>
#include <algorithm>
#include <tuple>
#include <limits>

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
    const std::pair<std::size_t, std::size_t> arrays[]{
        {problem.nodes.size(), sizeof(ceronoceres::Node)},
        {graph.node_ids.size(), sizeof(std::uint64_t)},
        {problem.relative_factors.size(), sizeof(ceronoceres::RelativeFactor)},
        {problem.absolute_factors.size(), sizeof(ceronoceres::AbsoluteFactor)},
        {problem.registration_groups.size(), sizeof(ceronoceres::RegistrationGroup)},
        {problem.registration_samples.size(), sizeof(ceronoceres::RegistrationSample)},
        {problem.volumes.size(), sizeof(ceronoceres::FieldVolume)},
        {problem.tsdf_tiles.size(), sizeof(ceronoceres::TsdfTile)},
        {problem.esdf_tiles.size(), sizeof(ceronoceres::EsdfTile)}};

    for (const auto& [count, size] : arrays)
    {
        if (count > std::size_t(std::numeric_limits<std::int32_t>::max()) ||
            count > (std::numeric_limits<std::uint64_t>::max() - result.record_bytes) / size)
            return std::unexpected(operation_error{"graph snapshot exceeds index or byte bounds"});
        result.record_bytes += count * size;
    }
    const auto tile_extent_valid = [](const auto& tiles) {
        constexpr auto lower = std::numeric_limits<std::int32_t>::min() / tile_edge;
        constexpr auto upper = (std::numeric_limits<std::int32_t>::max() - tile_edge) / tile_edge;

        for (const auto& tile : tiles)
            if (tile.x < lower || tile.y < lower || tile.z < lower ||
                tile.x > upper || tile.y > upper || tile.z > upper) return false;
        return true;
    };

    if (!tile_extent_valid(problem.tsdf_tiles) || !tile_extent_valid(problem.esdf_tiles))
        return std::unexpected(operation_error{"graph tile coordinates exceed voxel index bounds"});
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
