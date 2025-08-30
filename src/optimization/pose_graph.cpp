#include <covoxslam/optimization/pose_graph.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <tuple>

namespace covoxslam::pgo
{
operation_result<graph_snapshot> make_submap_graph(std::span<const submap> maps,
    std::span<const absolute_measurement> measurements)
{
    std::vector<graph_node> nodes;
    nodes.reserve(maps.size());

    for (const auto& map : maps) nodes.push_back({map.frame.id, map.frame.anchor, false});
    auto graph = make_graph(nodes);
    if (!graph) return graph;
    const auto odometry = append_odometry(*graph, maps);
    if (!odometry) return std::unexpected(odometry.error());
    const auto absolute = append_absolute(*graph, maps.front().frame.id, maps.front().frame.anchor, measurements);

    if (!absolute) return std::unexpected(absolute.error());
    return graph;
}
operation_result<ceronoceres::PreparedProblem> prepare_graph(graph_snapshot graph)
{
    const auto valid = validate_graph(graph);

    if (!valid) return std::unexpected(valid.error());
    return ceronoceres::prepare(std::move(graph.problem));
}
operation_result<std::optional<graph_snapshot>> make_graph_if_enabled(
    std::span<const graph_node> nodes, const options& options)
{
    if (!options.enabled) return std::optional<graph_snapshot>{};
    const auto valid = validate_options(options);
    if (!valid) return std::unexpected(valid.error());
    auto graph = make_graph(nodes);
    if (!graph) return std::unexpected(graph.error());
     return std::optional<graph_snapshot>{std::move(*graph)};
}
operation_status validate_graph(const graph_snapshot& graph)
{
    const auto& problem = graph.problem;
    if (problem.nodes.empty() || problem.nodes.size() > 1048576 ||
        graph.node_ids.size() != problem.nodes.size() || !problem.nodes.front().constant)
        return std::unexpected(operation_error{"invalid anchored graph shape"});
    std::set<std::uint64_t> ids;
    const auto finite_pose = [](const ceronoceres::Pose4& pose) {
        return std::isfinite(pose.x) && std::isfinite(pose.y) && std::isfinite(pose.z) && std::isfinite(pose.yaw);
    };

    for (std::size_t index = 0; index < problem.nodes.size(); ++index) {
        const auto& node = problem.nodes[index];
        if (node.id != graph.node_ids[index] || !ids.insert(node.id).second || !finite_pose(node.initial))
            return std::unexpected(operation_error{"invalid or duplicate graph node"});
    }
    const auto valid_loss = [](const ceronoceres::Loss& loss) {
        return unsigned(loss.kind) <= unsigned(ceronoceres::LossKind::Cauchy) &&
            std::isfinite(loss.scale) && loss.scale > 0;
    };
    const auto valid_factor = [&](std::size_t left, std::size_t right, const ceronoceres::Pose4& measured,
        const ceronoceres::Matrix4& information, const ceronoceres::Loss& loss) {
        return left < problem.nodes.size() && right < problem.nodes.size() && left != right &&
            finite_pose(measured) && valid_loss(loss) && std::all_of(information.begin(), information.end(),
                [](double value) { return std::isfinite(value); });
    };
    for (const auto& factor : problem.relative_factors)
        if (!valid_factor(factor.node_i, factor.node_j, factor.measured, factor.sqrt_information, factor.loss))
            return std::unexpected(operation_error{"invalid relative graph factor"});

    for (const auto& factor : problem.absolute_factors)
        if (!valid_factor(factor.reference_node, factor.measured_node, factor.measured_relative,
            factor.sqrt_information, factor.loss))
            return std::unexpected(operation_error{"invalid absolute graph factor"});
    const auto& model = problem.registration;
    if (!valid_loss(model.loss) || !std::isfinite(model.minimum_tsdf_weight) || model.minimum_tsdf_weight < 0 ||
        !std::isfinite(model.maximum_confidence_weight) || model.maximum_confidence_weight <= 0 ||
        !std::isfinite(model.minimum_gradient_norm) || model.minimum_gradient_norm < 0 ||
        !std::isfinite(model.maximum_absolute_residual_m) || model.maximum_absolute_residual_m < 0)
        return std::unexpected(operation_error{"invalid registration model"});

    for (const auto& volume : problem.volumes)
        if (volume.pose_node >= problem.nodes.size() || !std::isfinite(volume.voxel_size) || volume.voxel_size <= 0)
            return std::unexpected(operation_error{"invalid graph field volume"});
    const auto valid_tiles = [&](const auto& tiles) {
        std::set<std::tuple<std::size_t, int, int, int>> keys;
        for (const auto& tile : tiles) {
            if (tile.volume >= problem.volumes.size() || !keys.emplace(tile.volume, tile.x, tile.y, tile.z).second)
                return false;
        }
        return true;
    };
    if (!valid_tiles(problem.tsdf_tiles) || !valid_tiles(problem.esdf_tiles))
        return std::unexpected(operation_error{"invalid or duplicate graph tile"});
    std::vector<bool> used(problem.registration_samples.size());

    for (std::size_t index = 0; index < problem.registration_groups.size(); ++index) {
        const auto& group = problem.registration_groups[index];
        if (group.node_i >= problem.nodes.size() || group.node_j >= problem.nodes.size() || group.node_i == group.node_j ||
            group.volume_i >= problem.volumes.size() || group.volume_j >= problem.volumes.size() ||
            problem.volumes[group.volume_i].pose_node != group.node_i || problem.volumes[group.volume_j].pose_node != group.node_j ||
            group.sample_begin > used.size() || group.sample_count > used.size() - group.sample_begin ||
            !std::isfinite(group.direction_weight) || group.direction_weight < 0 || unsigned(group.field) > 1)
            return std::unexpected(operation_error{"invalid registration group"});
        for (std::size_t offset = 0; offset < group.sample_count; ++offset) {
            const auto sample_index = group.sample_begin + offset;
            const auto& sample = problem.registration_samples[sample_index];
            const auto& p = sample.reference_local_point;
            if (used[sample_index] || sample.group != index || !std::isfinite(p.x) || !std::isfinite(p.y) ||
                !std::isfinite(p.z) || !std::isfinite(sample.reference_distance) ||
                !std::isfinite(sample.reference_tsdf_weight) || sample.reference_tsdf_weight < 0)
                return std::unexpected(operation_error{"invalid or shared registration sample"});
            used[sample_index] = true;
        }
    }
    if (std::find(used.begin(), used.end(), false) != used.end())
        return std::unexpected(operation_error{"unowned registration sample"});
    return {};
}
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
    auto valid = validate_graph(result);
    if (!valid) return std::unexpected(valid.error());
    return result;
}
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
