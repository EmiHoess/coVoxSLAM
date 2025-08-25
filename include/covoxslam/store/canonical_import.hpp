#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/pipeline/submap.hpp>
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/store/coordinator.hpp>

namespace covoxslam::store
{
template<storage::layout l> operation_status import_active_submap(coordinated_store<l>& maps,
    const submap& source, bool band_topology = false)
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

            for (const auto& value : tile.voxels)
                values.push_back(value);
        }
        return maps.template import_field<s>(id, keys, values, s == storage::schema::tsdf ? 0 : 1);
    };
    checked = load.template operator()<storage::schema::tsdf>(source.tsdf);
    if (!checked) { static_cast<void>(maps.retire(id)); return checked; }
    if (band_topology) {
        std::vector<block_key> keys;
        keys.reserve(source.tsdf.size());

        for (const auto& [index, tile] : source.tsdf)
            keys.push_back({id, index.x, index.y, index.z});
        std::vector<storage::esdf_value> empty(keys.size() * 512);
        checked = maps.template import_field<storage::schema::esdf>(id, keys, empty, 1);
        if (!checked) { static_cast<void>(maps.retire(id)); return checked; }
    }
    else if (!source.esdf.empty()) {
        checked = load.template operator()<storage::schema::esdf>(source.esdf);
        if (!checked) { static_cast<void>(maps.retire(id)); return checked; }
    }
    auto current = maps.snapshot(id, 1, std::numeric_limits<std::size_t>::max());
    if (!current) return std::unexpected(current.error());
    const auto& optimized = source.optimized_world_pose;
    const pose_publication update{id, current->record.versions, {optimized.x, optimized.y, optimized.z, optimized.yaw}};
    return maps.publish_poses({&update, 1});
}
template<storage::layout l> operation_status import_band_topology(coordinated_store<l>& maps, const submap& source)
{
    return import_active_submap(maps, source, true);
}
template<storage::layout l> operation_status import_submap(coordinated_store<l>& maps, const submap& source)
{
    auto checked = import_active_submap(maps, source);
    if (!checked) return checked;
    return maps.finalize(source.frame.id, source.esdf.empty() ? 1U : 3U);
}
}
