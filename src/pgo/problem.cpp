#include <covoxslam/pgo.hpp>
#include <algorithm>
#include <cmath>
#include <set>
#include <tuple>

namespace covoxslam::pgo
{
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
}
