#pragma once
#include <covoxslam/store/handle.hpp>
#include <covoxslam/store/cpu_index.hpp>
#include <covoxslam/store/budget.hpp>
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
    std::shared_ptr<budget_claim> memory;
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
template<storage::layout l, storage::schema es = storage::schema::esdf> struct map_snapshot
{
    submap_record record;
    std::shared_ptr<const field_pool<storage::schema::tsdf, l>> tsdf;
    std::shared_ptr<const field_pool<es, l>> esdf;
};
struct store_statistics
{
    std::size_t submaps{}, retired{}, live_tiles{}, allocated_slots{}, payload_capacity_bytes{}, index_entries{};
};
template<storage::layout l, storage::schema es = storage::schema::esdf> class map_store
{
    std::unordered_map<std::uint64_t, map_entry<l, es>> maps_;
    std::size_t maximum_tiles_{1000000};
    std::shared_ptr<memory_budget> budget_;
    template<storage::schema s> operation_result<std::shared_ptr<field_pool<s, l>>> allocate_field(std::size_t count)
    {
        constexpr auto stride = storage::storage_traits<s, l>::tile_stride *
            (l == storage::layout::pool_soa ? storage::field_count<s> : 1);
        if (count > maximum_tiles_ || count > std::numeric_limits<std::size_t>::max() / stride)
            return std::unexpected(operation_error{"field allocation exceeds capacity"});
        auto claim = budget_->reserve({count * stride, 0, 0});
        if (!claim) return std::unexpected(claim.error());
        auto field = std::make_shared<field_pool<s, l>>();
        field->memory = std::move(*claim);
        auto checked = field->payload.initialize(count);
        if (!checked) return std::unexpected(checked.error());
        return field;
    }
    template<storage::schema s> auto& select_field(map_entry<l, es>& entry)
    {
        if constexpr (s == storage::schema::tsdf) return entry.tsdf;
        else return entry.esdf;
    }
    template<storage::schema s> operation_result<std::shared_ptr<field_pool<s, l>>> clone_field(
        std::uint64_t id, const std::shared_ptr<const field_pool<s, l>>& source, std::size_t count)
    {
        auto allocated = allocate_field<s>(count);
        if (!allocated) return std::unexpected(allocated.error());
        auto target = std::move(*allocated);
        if (source) {
            target->index = source->index;
            auto in = source->payload.view();
            auto out = target->payload.view();

            for (std::size_t tile = 0; tile < std::min(count, in.size()); ++tile)
                for (std::size_t voxel = 0; voxel < 512; ++voxel)
                    storage::scatter_record(out[tile].voxels[voxel], storage::gather_record(in[tile].voxels[voxel]));
        }
        else {
            auto checked = target->index.initialize(id, s == storage::schema::tsdf ? field_kind::tsdf : field_kind::esdf, {});
            if (!checked) return std::unexpected(checked.error());
        }
        return target;
    }
public:
    explicit map_store(std::size_t maximum_tiles = 1000000, std::shared_ptr<memory_budget> budget = {})
        : maximum_tiles_(maximum_tiles), budget_(budget ? std::move(budget) : std::make_shared<memory_budget>(residency_policy{}.limits)) {}
    budget_snapshot memory() const { return budget_->snapshot(); }
    std::size_t size() const noexcept { return maps_.size(); }
    const map_entry<l, es>* find(std::uint64_t id) const noexcept
    {
        const auto found = maps_.find(id);
        return found == maps_.end() ? nullptr : &found->second;
    }
    operation_result<store_statistics> statistics() const
    {
        store_statistics result;
        result.submaps = maps_.size();

        for (const auto& [id, map] : maps_) {
            result.retired += map.record.state == lifecycle::retired;
            auto count = [&](const auto& field) -> operation_status {
                if (!field) return {};
                using pool = std::remove_cvref_t<decltype(*field)>;
                constexpr auto selected = std::is_same_v<pool, field_pool<storage::schema::tsdf, l>> ? storage::schema::tsdf : es;
                constexpr auto bytes_per_tile = storage::storage_traits<selected, l>::tile_stride *
                    (l == storage::layout::pool_soa ? storage::field_count<selected> : 1);
                const auto descriptor = field->payload.view();
                const auto bytes = field->payload.capacity() * bytes_per_tile;
                if (bytes > std::numeric_limits<std::size_t>::max() - result.payload_capacity_bytes)
                    return std::unexpected(operation_error{"store statistics overflow"});
                result.payload_capacity_bytes += bytes;
                result.live_tiles += field->index.size();
                result.allocated_slots += descriptor.size();
                result.index_entries += field->index.size();
                return {};
            };
            auto checked = count(map.tsdf);
            if (!checked) return std::unexpected(checked.error());
            checked = count(map.esdf);
            if (!checked) return std::unexpected(checked.error());
        }
        return result;
    }
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
                constexpr auto bytes_per_tile = storage::storage_traits<s, l>::tile_stride *
                    (l == storage::layout::pool_soa ? storage::field_count<s> : 1);
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
    template<storage::schema s> operation_result<tile_handle> insert(block_key key)
    {
        auto found = maps_.find(key.submap);
        if (found == maps_.end() || found->second.record.state != lifecycle::active)
            return std::unexpected(operation_error{"insertion requires an active submap"});
        auto& entry = found->second;
        auto& source = select_field<s>(entry);
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
            auto& source = select_field<s>(entry);
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
        auto& field = [&]() -> auto& { if constexpr (s == storage::schema::tsdf) return entry.tsdf; else return entry.esdf; }();
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
