#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/optimization/pose_graph.hpp>
#include <bit>
#if defined(covoxslam_has_pgo)
#include <ceronoceres/field_views.hpp>
namespace covoxslam::store
{
struct preparation_key
{
    std::vector<std::byte> bytes;
    bool operator==(const preparation_key&) const = default;
};
inline operation_result<preparation_key> make_preparation_key(const pgo::graph_snapshot& graph,
    const ceronoceres::field_snapshot& fields, std::size_t budget)
{
    if (!graph.problem.tsdf_tiles.empty() || !graph.problem.esdf_tiles.empty())
        return std::unexpected(operation_error{"field cache requires uncopied graph payloads"});
    preparation_key key;
    bool overflow{};
    const auto add = [&]<class t>(t value) {
        if (overflow || sizeof(t) > budget - key.bytes.size()) { overflow = true; return; }
        const auto bytes = std::bit_cast<std::array<std::byte, sizeof(t)>>(value);
        key.bytes.insert(key.bytes.end(), bytes.begin(), bytes.end());
    };
    const auto pose = [&](const auto& p) { add(p.x); add(p.y); add(p.z); add(p.yaw); };
    const auto loss = [&](const auto& l) { add(l.kind); add(l.scale); };
    const auto& p = graph.problem;
    add(ceronoceres::field_view_api_version);
    add(fields.version); add(fields.device);
    add(fields.dependencies.size());
    for (auto dependency : fields.dependencies) add(dependency);
    add(graph.node_ids.size());

    for (auto id : graph.node_ids) add(id);
    add(p.nodes.size());
    for (const auto& node : p.nodes) { add(node.id); pose(node.initial); add(node.constant); }
    add(p.relative_factors.size());

    for (const auto& factor : p.relative_factors) {
        add(factor.node_i); add(factor.node_j); pose(factor.measured);
        for (auto value : factor.sqrt_information) add(value);
        loss(factor.loss); add(factor.diagnostic_id);
    }
    add(p.absolute_factors.size());

    for (const auto& factor : p.absolute_factors) {
        add(factor.reference_node); add(factor.measured_node); pose(factor.measured_relative);
        for (auto value : factor.sqrt_information) add(value);
        loss(factor.loss); add(factor.diagnostic_id);
    }
    add(p.volumes.size());
    for (const auto& volume : p.volumes) {
        add(volume.pose_node); add(volume.voxel_size); add(volume.has_tsdf); add(volume.has_esdf);
    }
    add(p.registration_groups.size());

    for (const auto& group : p.registration_groups) {
        add(group.node_i); add(group.node_j); add(group.volume_i); add(group.volume_j); add(group.field);
        add(group.sample_begin); add(group.sample_count); add(group.direction_weight); add(group.diagnostic_id);
    }
    add(p.registration_samples.size());
    for (const auto& sample : p.registration_samples) {
        add(sample.group); add(sample.reference_local_point.x); add(sample.reference_local_point.y);
        add(sample.reference_local_point.z); add(sample.reference_distance); add(sample.reference_tsdf_weight); add(sample.ordinal);
    }
    const auto& model = p.registration;
    add(model.evaluator); loss(model.loss); add(model.minimum_tsdf_weight); add(model.maximum_confidence_weight);
    add(model.minimum_gradient_norm); add(model.maximum_absolute_residual_m); add(model.allow_nearest_fallback);
    add(fields.tiles.size());

    for (const auto& tile : fields.tiles) {
        add(tile.volume); add(tile.kind); add(tile.x); add(tile.y); add(tile.z);
        add(std::uintptr_t(tile.distance)); add(std::uintptr_t(tile.weight)); add(std::uintptr_t(tile.flags));
        add(tile.voxel_stride); add(tile.encoding); add(tile.voxel_size); add(tile.maximum_distance);
        add(tile.content_epoch); add(tile.allocation_generation); add(tile.slot_generation);
    }
    if (overflow) return std::unexpected(operation_error{"preparation key exceeds byte budget"});
    return key;
}
}
#endif
