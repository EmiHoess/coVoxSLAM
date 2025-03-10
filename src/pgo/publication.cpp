#include <covoxslam/pgo_publication.hpp>
#include <algorithm>
#include <cmath>

namespace covoxslam::pgo
{
std::uint64_t publication_hash(std::span<const submap> maps)
{
    std::uint64_t hash = 14695981039346656037ULL;
    const auto scalar = [&]<class t>(const t& value) {
        for (auto byte : std::as_bytes(std::span(&value, 1))) {
            hash ^= std::to_integer<unsigned char>(byte);
            hash *= 1099511628211ULL;
        }
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

    for (std::size_t i = 0; i < maps.size(); ++i) {
        const auto& node = result.snapshot.graph().problem.nodes[i];
        const auto& initial = maps[i].frame.anchor;
        const auto& proposed = (*poses)[i];
        if (node.id != maps[i].frame.id || node.initial.x != initial.x || node.initial.y != initial.y ||
            node.initial.z != initial.z || std::abs(ceronoceres::wrap_yaw(node.initial.yaw - initial.yaw)) > 1e-12)
            return std::unexpected(operation_error{"publication graph does not match map anchors"});
        const auto translation = std::hypot(proposed.x - initial.x, proposed.y - initial.y, proposed.z - initial.z);

        if (translation > limits.maximum_translation_m ||
            std::abs(ceronoceres::wrap_yaw(proposed.yaw - initial.yaw)) > limits.maximum_yaw_rad)
            return std::unexpected(operation_error{"pose update exceeds publication limits"});
    }
    const auto& before = result.solution.initial_cost;
    const auto& after = result.solution.final_cost;
    if (after.registration > before.registration * (1 + limits.maximum_registration_growth) + 1e-10)
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
