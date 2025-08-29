#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/map_store/authority.hpp>
#include <covoxslam/map_store/coordinator.hpp>
#include <covoxslam/map_store/cpu_index.hpp>
#include <covoxslam/map_store/residency.hpp>
#include <covoxslam/store/field_access.hpp>
#include <covoxslam/storage/host_buffer.hpp>
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/storage/record_pack.hpp>
#include <cmath>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace covoxslam::store
{
template<storage::layout l, storage::schema es = storage::schema::esdf> class map_store
{
#include <covoxslam/map_store/detail/cpu_store_state.inl>
    operation_result<std::vector<map_snapshot<l, es>>> snapshot(std::span<const std::uint64_t> ids,
        std::uint32_t fields, std::size_t maximum_payload_bytes) const
    {
        if (!fields || (fields & ~3U)) return std::unexpected(operation_error{"snapshot field mask is invalid"});
        std::unordered_set<std::uint64_t> unique;
        std::vector<map_snapshot<l, es>> result;
        std::size_t remaining = maximum_payload_bytes;

        for (auto id : ids) {
            const auto* entry = find(id);
            if (!entry || entry->record.state == lifecycle::retired || !unique.insert(id).second)
                return std::unexpected(operation_error{"snapshot submap is absent retired or duplicated"});
            map_snapshot<l, es> copy;
            copy.record = entry->record;
            const auto bind = [&]<storage::schema s>(auto& target) -> operation_status {
                auto selected = field<s>(id);
                if (!selected) return std::unexpected(selected.error());
                const auto capacity = (*selected)->payload.capacity();
                constexpr auto bytes_per_tile = storage::payload_tile_bytes<s, l>;
                if (capacity > remaining / bytes_per_tile)
                    return std::unexpected(operation_error{"snapshot payload budget is exhausted"});
                remaining -= capacity * bytes_per_tile;
                target = *selected;
                return {};
            };
            if (fields & 1U) {
                auto checked = bind.template operator()<storage::schema::tsdf>(copy.tsdf);
                if (!checked) return std::unexpected(checked.error());
            }
            if (fields & 2U) {
                auto checked = bind.template operator()<es>(copy.esdf);
                if (!checked) return std::unexpected(checked.error());
            }
            result.push_back(std::move(copy));
        }
        return result;
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
    operation_status retire(std::uint64_t id)
    {
        auto found = maps_.find(id);
        if (found == maps_.end() || found->second.record.state == lifecycle::retired)
            return std::unexpected(operation_error{"submap is already absent or retired"});
        auto& entry = found->second;
        if (entry.record.writer != writer_kind::none)
            return std::unexpected(operation_error{"cannot retire a submap with a writer"});
        auto next = next_epoch(entry.record.versions.descriptor);
        if (!next) return std::unexpected(next.error());
        entry.record.state = lifecycle::retired;
        entry.record.versions.descriptor = *next;
        if (entry.tsdf && entry.tsdf->memory) entry.tsdf->memory->retire();
        if (entry.esdf && entry.esdf->memory) entry.esdf->memory->retire();
        entry.tsdf.reset();
        entry.esdf.reset();
        entry.record.cpu = entry.record.gpu = residency::absent;
        return {};
    }
    operation_status seal(std::uint64_t id, std::uint32_t fields, std::uint64_t esdf_source)
    {
        auto found = maps_.find(id);
        if (found == maps_.end() || !fields || (fields & ~3U) || found->second.record.state != lifecycle::active)
            return std::unexpected(operation_error{"invalid submap seal"});
        auto& entry = found->second;
        if (((fields & 1U) && !entry.tsdf) || ((fields & 2U) && (!entry.esdf || !entry.tsdf ||
            esdf_source != entry.record.versions.tsdf)) || entry.record.writer != writer_kind::none)
            return std::unexpected(operation_error{"submap seal requires current fields"});
        entry.record.versions.esdf_source = esdf_source;
        entry.record.state = lifecycle::finalized;
        return {};
    }
    template<storage::schema s> operation_result<tile_handle> lookup(block_key key) const
    {
        auto selected = field<s>(key.submap);
        if (!selected) return std::unexpected(selected.error());
        return (*selected)->index.find(key);
    }
    template<storage::schema s> operation_result<std::shared_ptr<const field_pool<s, l>>> field(std::uint64_t id) const
    {
        const auto* entry = find(id);
        if (!entry || entry->record.state == lifecycle::retired)
            return std::unexpected(operation_error{"submap identity is unavailable"});
        auto result = select_field<s, es>(*entry);
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
    template<storage::schema s> operation_result<tile_handle> insert(block_key key)
    {
        auto found = maps_.find(key.submap);
        if (found == maps_.end() || found->second.record.state != lifecycle::active)
            return std::unexpected(operation_error{"insertion requires an active submap"});
        auto& entry = found->second;
        auto& source = select_field<s, es>(entry);
        if (source) {
            auto existing = source->index.find(key);
            if (existing) return existing;
        }
        auto index = source ? source->index : cpu_block_index{};
        if (!source) {
            auto initialized = index.initialize(key.submap, s == storage::schema::tsdf ? field_kind::tsdf : field_kind::esdf, {});
            if (!initialized) return std::unexpected(initialized.error());
        }
        auto handle = index.insert(key, maximum_tiles_);
        if (!handle) return std::unexpected(handle.error());
        const auto epoch = next_epoch(source ? source->content_epoch : 0);
        const auto allocation = next_epoch(entry.record.versions.allocation), descriptor = next_epoch(entry.record.versions.descriptor);
        if (!epoch || !allocation || !descriptor) return std::unexpected(operation_error{"store epoch exhausted"});
        auto replacement = clone_field<s>(key.submap, source, index.slot_count());
        if (!replacement) return std::unexpected(replacement.error());
        (*replacement)->index = std::move(index);
        auto view = (*replacement)->payload.view();

        for (std::size_t voxel = 0; voxel < 512; ++voxel)
            storage::scatter_record(view[handle->slot].voxels[voxel], storage::value<s>{});
        (*replacement)->content_epoch = *epoch;
        if (source && source->memory) source->memory->retire();
        source = std::move(*replacement);
        entry.record.versions.allocation = *allocation;
        entry.record.versions.descriptor = *descriptor;
        (s == storage::schema::tsdf ? entry.record.versions.tsdf : entry.record.versions.esdf) = *epoch;
        entry.record.cpu = residency::resident;
        return handle;
    }
    operation_status erase(tile_handle handle)
    {
        auto key = resolve(handle);
        if (!key) return std::unexpected(key.error());
        auto& entry = maps_.at(handle.submap);
        if (entry.record.state != lifecycle::active) return std::unexpected(operation_error{"removal requires an active submap"});
        auto remove = [&]<storage::schema s>() -> operation_status {
            auto& source = select_field<s, es>(entry);
            auto epoch = next_epoch(source->content_epoch), allocation = next_epoch(entry.record.versions.allocation);
            auto descriptor = next_epoch(entry.record.versions.descriptor);
            if (!epoch || !allocation || !descriptor) return std::unexpected(operation_error{"store epoch exhausted"});
            auto replacement = clone_field<s>(handle.submap, source, source->payload.size());
            if (!replacement) return std::unexpected(replacement.error());
            auto checked = (*replacement)->index.erase(handle);
            if (!checked) return checked;
            auto view = (*replacement)->payload.view();

            for (std::size_t voxel = 0; voxel < 512; ++voxel)
                storage::scatter_record(view[handle.slot].voxels[voxel], storage::value<s>{});
            (*replacement)->content_epoch = *epoch;
            if (source->memory) source->memory->retire();
            source = std::move(*replacement);
            entry.record.versions.allocation = *allocation;
            entry.record.versions.descriptor = *descriptor;
            (s == storage::schema::tsdf ? entry.record.versions.tsdf : entry.record.versions.esdf) = *epoch;
            return {};
        };
        return handle.field == field_kind::tsdf ? remove.template operator()<storage::schema::tsdf>() : remove.template operator()<es>();
    }
    template<storage::schema s> operation_status import_field(std::uint64_t id,
        std::span<const block_key> keys, std::span<const storage::value<s>> records)
    {
        auto found = maps_.find(id);
        if (found == maps_.end() || found->second.record.state != lifecycle::active)
            return std::unexpected(operation_error{"field import requires an active submap"});
        auto& entry = found->second;
        auto& field = select_field<s, es>(entry);
        const auto allocation = next_epoch(entry.record.versions.allocation);
        const auto descriptor = next_epoch(entry.record.versions.descriptor);
        if (!allocation || !descriptor) return std::unexpected(operation_error{"store epoch exhausted"});

        if (field) return std::unexpected(operation_error{"field import cannot replace published storage"});
        if (keys.size() > maximum_tiles_ || keys.size() > std::numeric_limits<std::uint32_t>::max() ||
            records.size() / 512 != keys.size() || records.size() % 512)
            return std::unexpected(operation_error{"field payload size is invalid"});
        std::unordered_set<block_key, key_hash> unique;

        for (const auto& key : keys)
            if (key.submap != id || !unique.insert(key).second)
                return std::unexpected(operation_error{"field keys conflict"});
        auto allocated = allocate_field<s>(keys.size());
        if (!allocated) return std::unexpected(allocated.error());
        auto replacement = std::move(*allocated);
        auto checked = replacement->index.initialize(id, s == storage::schema::tsdf ? field_kind::tsdf : field_kind::esdf, keys);
        if (!checked) return checked;
        auto view = replacement->payload.view();

        for (std::size_t i = 0; i < records.size(); ++i)
            storage::scatter_record<s>(view[i / 512].voxels[i % 512], records[i]);
        replacement->content_epoch = 1;
        field = std::move(replacement);
        entry.record.versions.allocation = *allocation;
        entry.record.versions.descriptor = *descriptor;
        entry.record.cpu = residency::resident;
        if constexpr (s == storage::schema::tsdf) entry.record.versions.tsdf = 1;
        else entry.record.versions.esdf = 1;
        return {};
    }
};
}
