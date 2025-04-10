#pragma once
#include <covoxslam/store/handle.hpp>
#include <covoxslam/store/cpu_index.hpp>
#include <covoxslam/storage/draft.hpp>
#include <covoxslam/storage/record_pack.hpp>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace covoxslam::store
{
template<storage::schema s, storage::layout l> struct field_pool
{
    storage::draft<s, l> payload;
    cpu_block_index index;
    std::uint64_t content_epoch{};
};
template<storage::layout l, storage::schema es = storage::schema::esdf> struct map_entry
{
    submap_record record;
    std::shared_ptr<const field_pool<storage::schema::tsdf, l>> tsdf;
    std::shared_ptr<const field_pool<es, l>> esdf;
};
inline bool finite_pose(const pose_record& pose) noexcept
{
    return std::isfinite(pose.x) && std::isfinite(pose.y) && std::isfinite(pose.z) && std::isfinite(pose.yaw);
}
template<storage::layout l, storage::schema es = storage::schema::esdf> class map_store
{
    std::unordered_map<std::uint64_t, map_entry<l, es>> maps_;
    std::size_t maximum_tiles_{1000000};
public:
    explicit map_store(std::size_t maximum_tiles = 1000000) : maximum_tiles_(maximum_tiles) {}
    std::size_t size() const noexcept { return maps_.size(); }
    const map_entry<l, es>* find(std::uint64_t id) const noexcept
    {
        const auto found = maps_.find(id);
        return found == maps_.end() ? nullptr : &found->second;
    }
    operation_status create(std::uint64_t id, pose_record pose = {})
    {
        if (maps_.contains(id)) return std::unexpected(operation_error{"submap identity already exists"});
        if (!finite_pose(pose)) return std::unexpected(operation_error{"submap pose is not finite"});
        map_entry<l, es> entry;
        entry.record.id = id;
        entry.record.initial_pose = entry.record.optimized_pose = pose;
        maps_.emplace(id, std::move(entry));
        return {};
    }
    operation_status update_pose(std::uint64_t id, std::uint64_t expected, pose_record pose)
    {
        auto found = maps_.find(id);
        if (found == maps_.end() || found->second.record.state == lifecycle::retired || !finite_pose(pose))
            return std::unexpected(operation_error{"pose publication has invalid identity or value"});
        auto& record = found->second.record;
        if (record.versions.pose != expected)
            return std::unexpected(operation_error{"pose publication is stale"});
        auto next = next_epoch(expected);
        if (!next) return std::unexpected(next.error());
        record.optimized_pose = pose;
        record.versions.pose = *next;
        return {};
    }
    template<storage::schema s> operation_result<std::shared_ptr<const field_pool<s, l>>> field(std::uint64_t id) const
    {
        const auto* entry = find(id);
        if (!entry || entry->record.state == lifecycle::retired)
            return std::unexpected(operation_error{"submap identity is unavailable"});
        auto result = [&]() { if constexpr (s == storage::schema::tsdf) return entry->tsdf; else return entry->esdf; }();
        if (!result) return std::unexpected(operation_error{"submap field is absent"});
        return result;
    }
    operation_result<block_key> resolve(tile_handle handle) const
    {
        const auto resolve_field = [&]<storage::schema s>() -> operation_result<block_key> {
            auto selected = field<s>(handle.submap);
            if (!selected) return std::unexpected(selected.error());
            return (*selected)->index.resolve(handle);
        };
        switch (handle.field) {
        case field_kind::tsdf: return resolve_field.template operator()<storage::schema::tsdf>();
        case field_kind::esdf: return resolve_field.template operator()<es>();
        default: return std::unexpected(operation_error{"tile field is invalid"});
        }
    }
    template<storage::schema s> operation_status import_field(std::uint64_t id,
        std::span<const block_key> keys, std::span<const storage::value<s>> records)
    {
        auto found = maps_.find(id);
        if (found == maps_.end() || found->second.record.state != lifecycle::active)
            return std::unexpected(operation_error{"field import requires an active submap"});
        auto& entry = found->second;
        auto& field = [&]() -> auto& { if constexpr (s == storage::schema::tsdf) return entry.tsdf; else return entry.esdf; }();

        if (field) return std::unexpected(operation_error{"field import cannot replace published storage"});
        if (keys.size() > maximum_tiles_ || keys.size() > std::numeric_limits<std::uint32_t>::max() ||
            records.size() / 512 != keys.size() || records.size() % 512)
            return std::unexpected(operation_error{"field payload size is invalid"});
        std::unordered_set<block_key, key_hash> unique;

        for (const auto& key : keys)
            if (key.submap != id || !unique.insert(key).second)
                return std::unexpected(operation_error{"field keys conflict"});
        auto replacement = std::make_shared<field_pool<s, l>>();
        auto checked = replacement->payload.initialize(keys.size());
        if (!checked) return checked;
        checked = replacement->index.initialize(id, s == storage::schema::tsdf ? field_kind::tsdf : field_kind::esdf, keys);
        if (!checked) return checked;
        auto view = replacement->payload.view();

        for (std::size_t i = 0; i < records.size(); ++i)
            storage::scatter_record<s>(view[i / 512].voxels[i % 512], records[i]);
        replacement->content_epoch = 1;
        field = std::move(replacement);
        entry.record.cpu = residency::resident;
        if constexpr (s == storage::schema::tsdf) entry.record.versions.tsdf = 1;
        else entry.record.versions.esdf = 1;
        return {};
    }
};
}
