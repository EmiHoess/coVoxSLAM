#pragma once
#include <covoxslam/store/coordinator.hpp>
#include <covoxslam/mapping.hpp>

namespace covoxslam::store
{
template<storage::layout l> operation_status import_submap(coordinated_store<l>& maps, const submap& source)
{
    const auto id = source.frame.id;
    const auto& initial = source.frame.anchor;
    auto checked = maps.create(id, {initial.x, initial.y, initial.z, initial.yaw});
    if (!checked) return checked;
    const auto load = [&]<storage::schema s>(const auto& tiles) -> operation_status {
        return maps.template edit<s>(id, [&](auto& writer) -> operation_status {

            for (const auto& [index, tile] : tiles) {
                auto handle = writer.allocate({id, index.x, index.y, index.z});
                if (!handle) return std::unexpected(handle.error());

                for (std::size_t i = 0; i < 512; ++i) {
                    auto target = writer.voxel(*handle, i);
                    if (!target) return std::unexpected(target.error());
                    const auto& value = tile.voxels[i];
                    if constexpr (s == storage::schema::tsdf)
                        storage::scatter_record<s>(*target, {value.distance, value.weight, value.sample_count, value.flags});
                    else storage::scatter_record<s>(*target, {value.signed_distance_m,
                        value.parent_dx, value.parent_dy, value.parent_dz, value.flags});
                }
            }
            return {};
        }, s == storage::schema::tsdf ? 0 : 1);
    };
    checked = load.template operator()<storage::schema::tsdf>(source.tsdf);
    if (!checked) { maps.retire(id); return checked; }
    if (!source.esdf.empty()) {
        checked = load.template operator()<storage::schema::esdf>(source.esdf);
        if (!checked) { maps.retire(id); return checked; }
    }
    auto current = maps.snapshot(id, 1, std::numeric_limits<std::size_t>::max());
    if (!current) return std::unexpected(current.error());
    const auto& optimized = source.optimized_world_pose;
    const pose_publication update{id, current->record.versions, {optimized.x, optimized.y, optimized.z, optimized.yaw}};
    checked = maps.publish_poses({&update, 1});
    if (!checked) return checked;
    return maps.finalize(id, source.esdf.empty() ? 1U : 3U);
}
}
