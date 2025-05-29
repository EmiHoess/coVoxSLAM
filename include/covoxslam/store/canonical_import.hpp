#pragma once
#include <covoxslam/store/coordinator.hpp>
#include <covoxslam/mapping.hpp>

namespace covoxslam::store
{
template<storage::layout l> operation_status import_band_topology(coordinated_store<l>& maps, const submap& source)
{
    const auto id = source.frame.id;
    const auto& pose = source.frame.anchor;
    auto checked = maps.create(id, {pose.x, pose.y, pose.z, pose.yaw});

    if (!checked) return checked;
    const auto load = [&]() -> operation_status {
        if (source.tsdf.size() > source.config.maximum_tiles ||
            source.tsdf.size() > std::numeric_limits<std::size_t>::max() / 512 / 20)
            return std::unexpected(operation_error{"band topology exceeds import bounds"});
        std::vector<block_key> keys;
        std::vector<storage::tsdf_value> values;
        keys.reserve(source.tsdf.size()); values.reserve(source.tsdf.size() * 512);

        for (const auto& [index, tile] : source.tsdf) {
            keys.push_back({id, index.x, index.y, index.z});

            for (const auto& value : tile.voxels)
                values.push_back({value.distance, value.weight, value.sample_count, value.flags});
        }
        auto imported = maps.template import_field<storage::schema::tsdf>(id, keys, values);
        if (!imported) return imported;
        std::vector<storage::esdf_value> empty(values.size());
        return maps.template import_field<storage::schema::esdf>(id, keys, empty, 1);
    };
    checked = load();

    if (!checked) static_cast<void>(maps.retire(id));
    return checked;
}
template<storage::layout l> operation_status import_submap(coordinated_store<l>& maps, const submap& source)
{
    const auto id = source.frame.id;
    const auto& initial = source.frame.anchor;
    auto checked = maps.create(id, {initial.x, initial.y, initial.z, initial.yaw});
    if (!checked) return checked;
    const auto load = [&]<storage::schema s>(const auto& tiles) -> operation_status {
        using value_type = storage::value<s>;
        if (tiles.size() > source.config.maximum_tiles ||
            tiles.size() > std::numeric_limits<std::size_t>::max() / 512 / sizeof(value_type))
            return std::unexpected(operation_error{"canonical field exceeds import bounds"});
        std::vector<block_key> keys;
        std::vector<value_type> values;
        keys.reserve(tiles.size());
        values.reserve(tiles.size() * 512);

        for (const auto& [index, tile] : tiles) {
            keys.push_back({id, index.x, index.y, index.z});

            for (const auto& value : tile.voxels) {
                if constexpr (s == storage::schema::tsdf)
                    values.push_back({value.distance, value.weight, value.sample_count, value.flags});
                else values.push_back({value.signed_distance_m,
                    value.parent_dx, value.parent_dy, value.parent_dz, value.flags});
            }
        }
        return maps.template import_field<s>(id, keys, values, s == storage::schema::tsdf ? 0 : 1);
    };
    checked = load.template operator()<storage::schema::tsdf>(source.tsdf);
    if (!checked) { static_cast<void>(maps.retire(id)); return checked; }
    if (!source.esdf.empty()) {
        checked = load.template operator()<storage::schema::esdf>(source.esdf);
        if (!checked) { static_cast<void>(maps.retire(id)); return checked; }
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
