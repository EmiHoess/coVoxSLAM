#include <covoxslam/pgo/solver.hpp>
#include <covoxslam/pgo/publication.hpp>
#include <covoxslam/core/hash.hpp>
#include "contracts.hpp"
#include <algorithm>
#include <cmath>

namespace covoxslam::pgo
{
operation_result<std::optional<optimization_result>> optimize_submaps(std::span<const submap> maps,
    const optimization_options& options, std::span<const absolute_measurement> measurements)
{
    if (!options.execution.enabled) return std::optional<optimization_result>{};
    const auto configured = validate_options(options.execution);
    if (!configured) return std::unexpected(configured.error());
    const auto input_hash = publication_hash(maps);
    auto graph = make_submap_graph(maps, measurements);
    if (!graph) return std::unexpected(graph.error());
    auto candidates = find_candidates(maps, options.candidates);
    if (!candidates) return std::unexpected(candidates.error());
    const auto counts = summarize_candidates(*candidates);
    if (!counts) return std::unexpected(counts.error());
    const auto registration = append_registration(*graph, maps, *candidates, options.registration);
    if (!registration) return std::unexpected(registration.error());
    auto snapshot = freeze_graph(std::move(*graph));
    if (!snapshot) return std::unexpected(snapshot.error());
    auto solution = solve_native(*snapshot, options.execution, options.lm);

    if (!solution) return std::unexpected(solution.error());
    return std::optional<optimization_result>{{std::move(*snapshot), std::move(*solution), *counts, input_hash}};
}
std::uint64_t publication_hash(std::span<const submap> maps)
{
    std::uint64_t hash = fnv1a_offset_basis;
    const auto scalar = [&]<class t>(const t& value) {
        hash = fnv1a(std::as_bytes(std::span(&value, 1)), hash);
    };
    const auto pose_hash = [&](const pose4& p) { scalar(p.x); scalar(p.y); scalar(p.z); scalar(p.yaw); };
    scalar(maps.size());

    for (const auto& map : maps)
    {
        scalar(map.frame.id); scalar(map.frame.start_stamp_ns); scalar(map.frame.end_stamp_ns);
        pose_hash(map.frame.anchor); pose_hash(map.optimized_world_pose);
        scalar(map.frame.frame_ids.size());

        for (int id : map.frame.frame_ids) scalar(id);
        scalar(map.config.voxel_size); scalar(map.config.truncation_distance);
        scalar(map.config.esdf_generator); scalar(map.config.esdf_minimum_weight);
        scalar(map.config.extraction_distance); scalar(map.config.maximum_weight);
        scalar(map.config.minimum_ray_length_m); scalar(map.config.maximum_ray_length_m);
        scalar(map.config.maximum_tiles); scalar(map.config.esdf_site_band_m); scalar(map.config.esdf_maximum_distance_m);
        const auto& initial = map.frame.initial_pose;
        scalar(initial.translation.x); scalar(initial.translation.y); scalar(initial.translation.z);
        scalar(initial.rotation.x); scalar(initial.rotation.y); scalar(initial.rotation.z); scalar(initial.rotation.w);
        const auto tiles_hash = [&](const auto& tiles, const auto& voxel_hash) {
            std::vector<int3> keys;
            keys.reserve(tiles.size());

            for (const auto& [key, tile] : tiles) keys.push_back(key);
            std::sort(keys.begin(), keys.end(), block_index_less);
            scalar(keys.size());

            for (const auto key : keys) {
                scalar(key.x); scalar(key.y); scalar(key.z);

                for (const auto& voxel : tiles.find(key)->second.voxels) voxel_hash(voxel);
            }
        };
        tiles_hash(map.tsdf, [&](const tsdf_voxel& v) {
            scalar(v.distance); scalar(v.weight); scalar(v.sample_count); scalar(v.flags);
        });
        tiles_hash(map.esdf, [&](const esdf_voxel& v) {
            scalar(v.signed_distance_m); scalar(v.parent_dx); scalar(v.parent_dy); scalar(v.parent_dz); scalar(v.flags);
        });
    }
    return hash;
}
operation_result<publication_update> prepare_publication(std::span<const submap> maps,
    const optimization_result& result, const publication_limits& limits)
{
    const auto valid = validate_snapshot_result(result.snapshot, result.solution);
    if (!valid) return std::unexpected(valid.error());
    auto poses = solution_poses(result.solution, maps.size());

    if (!poses) return std::unexpected(poses.error());
    if (maps.size() != result.snapshot.extents().nodes || publication_hash(maps) != result.map_hash)
        return std::unexpected(operation_error{"map data changed since graph construction"});
    const double bounds[]{limits.maximum_translation_m, limits.maximum_yaw_rad, limits.maximum_registration_growth};

    for (double bound : bounds)
        if (!std::isfinite(bound) || bound < 0)
            return std::unexpected(operation_error{"invalid pose publication limits"});
    if (!std::isfinite(limits.minimum_registration_retention) || limits.minimum_registration_retention < 0 ||
        limits.minimum_registration_retention > 1)
        return std::unexpected(operation_error{"invalid registration retention limit"});

    for (std::size_t i = 0; i < maps.size(); ++i) {
        const auto& node = result.snapshot.graph().problem.nodes[i];
        const auto& initial = maps[i].frame.anchor;
        const auto& proposed = (*poses)[i];
        if (node.id != maps[i].frame.id || !same_pose(node.initial, initial))
            return std::unexpected(operation_error{"publication graph does not match map anchors"});
        const auto translation = std::hypot(proposed.x - initial.x, proposed.y - initial.y, proposed.z - initial.z);

        if (translation > limits.maximum_translation_m ||
            std::abs(abseil::wrap_yaw(proposed.yaw - initial.yaw)) > limits.maximum_yaw_rad)
            return std::unexpected(operation_error{"pose update exceeds publication limits"});
    }
    std::vector<pose4> initial_poses;
    initial_poses.reserve(maps.size());

    for (const auto& map : maps) initial_poses.push_back(map.frame.anchor);
    const auto before = linearize_graph(result.snapshot.prepared(), initial_poses);
    const auto after = linearize_graph(result.snapshot.prepared(), *poses);

    if (!before) return std::unexpected(before.error());
    if (!after) return std::unexpected(after.error());
    const auto agrees = [](const auto& reported, const auto& measured) {
        const auto close = [](double a, double b) { return std::abs(a - b) <= measured_comparison_tolerance *
            std::max({1.0, std::abs(a), std::abs(b)}); };
        return close(reported.relative, measured.relative) && close(reported.absolute, measured.absolute) &&
            close(reported.registration, measured.registration);
    };
    if (!agrees(result.solution.initial_cost, before->cost) || !agrees(result.solution.final_cost, after->cost))
        return std::unexpected(operation_error{"solver cost does not match frozen graph evaluation"});
    if (after->cost.total() > before->cost.total() + publication_comparison_tolerance *
        std::max(1.0, before->cost.total()))
        return std::unexpected(operation_error{"verified objective increased"});
    if (double(after->registration.accepted_rows) <
        double(before->registration.accepted_rows) * limits.minimum_registration_retention)
        return std::unexpected(operation_error{"registration support decreased beyond publication limit"});
    if (after->cost.registration > before->cost.registration * (1 + limits.maximum_registration_growth) +
        registration_additive_tolerance)
        return std::unexpected(operation_error{"registration cost worsened beyond publication limit"});
    return publication_update{result.map_hash, std::move(*poses)};
}
operation_status publish_poses(std::span<submap> maps, const publication_update& update, publication_mode mode)
{
    if (mode != publication_mode::apply && mode != publication_mode::shadow)
        return std::unexpected(operation_error{"invalid pose publication mode"});
    if (maps.size() != update.poses_.size() || publication_hash(maps) != update.source_hash_)
        return std::unexpected(operation_error{"pose publication source is stale"});

    if (mode == publication_mode::shadow) return {};

    for (std::size_t i = 0; i < maps.size(); ++i)
        maps[i].optimized_world_pose = update.poses_[i];
    return {};
}
}
