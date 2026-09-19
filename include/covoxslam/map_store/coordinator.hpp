#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/map_store/authority.hpp>
#include <covoxslam/map_store/cpu_index.hpp>
#include <covoxslam/map_store/residency.hpp>
#include <covoxslam/map_store/snapshot.hpp>
#include <covoxslam/map_store/checkpoint.hpp>
#include <covoxslam/map_store/detail/field_access.hpp>
#include <covoxslam/storage/host_buffer.hpp>
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/storage/transfers.hpp>
#if defined(covoxslam_has_cuda)
#include <covoxslam/storage/device_buffer.hpp>
#include <covoxslam/map_store/detail/gpu_store.hpp>
#endif
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace covoxslam::store
{
template<storage::schema s, storage::layout l> class field_editor
{
    field_pool<s, l>& field_;
    memory_budget& budget_;
    std::size_t maximum_tiles_;
    std::optional<operation_error> failure_;
public:
    struct cached_tile
    {
        tile_handle handle;
        const void* owner{};
        std::uint64_t generation{};
    };
    using voxel_reference = decltype(std::declval<storage::host_buffer<s, l>&>().view()[0].voxels[0]);
    field_editor(field_pool<s, l>& field, memory_budget& budget, std::size_t maximum_tiles)
        : field_(field), budget_(budget), maximum_tiles_(maximum_tiles) {}
    field_editor(const field_editor&) = delete;
    field_editor& operator=(const field_editor&) = delete;
    std::size_t size() const { return field_.index.size(); }
    std::span<const slot_record> slots() const { return field_.index.records(); }
    operation_result<tile_handle> find(block_key key) const { return field_.index.find(key); }
private:
    operation_result<tile_handle> allocate_impl(block_key key)
    {
        auto present = find(key);
        if (present) return present;
        const auto count = field_.payload.size();
        auto handle = field_.index.insert(key, maximum_tiles_);
        if (!handle) return std::unexpected(handle.error());
        const auto capacity = field_.index.slot_count();
        std::shared_ptr<budget_claim> replacement;
        if (capacity > field_.payload.capacity()) {
            constexpr auto stride = storage::payload_tile_bytes<s, l>;
            if (capacity > std::numeric_limits<std::size_t>::max() / stride)
                return std::unexpected(operation_error{"mapping field growth overflows"});
            auto memory = budget_.reserve({capacity * stride, 0, 0});
            if (!memory) return std::unexpected(memory.error());
            replacement = std::move(*memory);
        }
        auto checked = field_.payload.resize(capacity);
        if (!checked) return std::unexpected(checked.error());
        if (replacement) field_.memory = std::move(replacement);
        if (handle->slot < count) {
            auto view = field_.payload.view();

            for (std::size_t voxel = 0; voxel < tile_voxels; ++voxel)
                storage::scatter_record(view[handle->slot].voxels[voxel], storage::value<s>{});
        }
        return handle;
    }
public:
    operation_status status() const
    {
        if (failure_) return std::unexpected(*failure_);
        if (field_.payload.size() != field_.index.slot_count())
            return std::unexpected(operation_error{"mapping payload and index sizes differ"});
        return {};
    }
    operation_result<tile_handle> allocate(block_key key)
    {
        if (failure_) return std::unexpected(*failure_);
        auto result = allocate_impl(key);
        if (!result) failure_ = result.error();
        return result;
    }
    operation_result<cached_tile> cache(block_key key)
    {
        auto handle = allocate(key);
        if (!handle) return std::unexpected(handle.error());
        return cached_tile{*handle, &field_, field_.payload.generation()};
    }
    operation_result<voxel_reference> voxel(tile_handle handle, std::size_t local)
    {
        if (failure_) return std::unexpected(*failure_);
        auto checked = field_.index.resolve(handle);
        if (!checked) { failure_ = checked.error(); return std::unexpected(*failure_); }
        if (local >= tile_voxels) {
            failure_ = operation_error{"local voxel index is invalid"};
            return std::unexpected(*failure_);
        }
        return field_.payload.view()[handle.slot].voxels[local];
    }
    operation_result<voxel_reference> voxel(cached_tile cached, std::size_t local)
    {
        if (cached.owner != &field_ || cached.generation != field_.payload.generation()) {
            failure_ = operation_error{"cached mapping view is stale"};
            return std::unexpected(*failure_);
        }
        return voxel(cached.handle, local);
    }
};
template<storage::layout l, storage::schema es = storage::schema::esdf> struct map_entry
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
        constexpr auto stride = storage::payload_tile_bytes<s, l>;
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
                for (std::size_t voxel = 0; voxel < tile_voxels; ++voxel)
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
    template<storage::schema s> operation_status initialize_topology(std::uint64_t id,
        std::span<const block_key> keys, std::uint64_t epoch, std::uint64_t esdf_source = 0)
    {
        auto found = maps_.find(id);

        if (found == maps_.end() || found->second.record.state != lifecycle::active || !epoch ||
            select_field<s, es>(found->second))
            return std::unexpected(operation_error{"topology requires an empty active field"});
        cpu_block_index index;
        auto checked = index.initialize(id, s == storage::schema::tsdf ? field_kind::tsdf : field_kind::esdf, keys);
        if (!checked) return checked;
        auto field = allocate_field<s>(keys.size());

        if (!field) return std::unexpected(field.error());
        (*field)->index = std::move(index);
         (*field)->content_epoch = epoch;
        return install_mirror<s>(id, std::move(*field), esdf_source);
    }
    template<storage::schema s> operation_status install_mirror(std::uint64_t id,
        std::shared_ptr<const field_pool<s, l>> replacement, std::uint64_t esdf_source)
    {
        auto found = maps_.find(id);
        if (found == maps_.end() || found->second.record.state == lifecycle::retired || !replacement)
            return std::unexpected(operation_error{"mirror destination is unavailable"});
        auto& entry = found->second;
        auto& field = select_field<s, es>(entry);
        const auto kind = s == storage::schema::tsdf ? field_kind::tsdf : field_kind::esdf;
        auto generation = next_epoch(entry.record.versions.allocation), descriptor = next_epoch(entry.record.versions.descriptor);
        if (!generation || !descriptor || !budget_->owns(replacement->memory) || replacement->index.submap() != id ||
            replacement->index.field() != kind || replacement->payload.size() > maximum_tiles_ ||
            replacement->payload.size() != replacement->index.slot_count() || !replacement->content_epoch ||
            (field && replacement->content_epoch < field->content_epoch))
            return std::unexpected(operation_error{"mirror metadata or admission is invalid"});
        if (field && field->memory) field->memory->retire();
        field = std::move(replacement);
        entry.record.versions.allocation = *generation; entry.record.versions.descriptor = *descriptor;
        (s == storage::schema::tsdf ? entry.record.versions.tsdf : entry.record.versions.esdf) = field->content_epoch;
        if constexpr (s != storage::schema::tsdf) entry.record.versions.esdf_source = esdf_source;
        entry.record.cpu = residency::resident;
        return {};
    }
    template<storage::schema s, class function> operation_status edit_field(std::uint64_t id, function&& edit)
    {
        auto found = maps_.find(id);
        if (found == maps_.end() || found->second.record.state != lifecycle::active)
            return std::unexpected(operation_error{"field editing requires an active submap"});
        auto& entry = found->second;
        auto& source = select_field<s, es>(entry);
        auto epoch = next_epoch(source ? source->content_epoch : 0);
        auto allocation = next_epoch(entry.record.versions.allocation), descriptor = next_epoch(entry.record.versions.descriptor);
        if (!epoch || !allocation || !descriptor) return std::unexpected(operation_error{"field editing epoch exhausted"});
        auto replacement = clone_field<s>(id, source, source ? source->payload.size() : 0);
        if (!replacement) return std::unexpected(replacement.error());
        field_editor<s, l> editor(**replacement, *budget_, maximum_tiles_);
        auto checked = edit(editor);
        if (!checked) return checked;
        checked = editor.status();
        if (!checked) return checked;
        (*replacement)->content_epoch = *epoch;
        if (source && source->memory) source->memory->retire();
        source = std::move(*replacement);
        entry.record.versions.allocation = *allocation; entry.record.versions.descriptor = *descriptor;
        (s == storage::schema::tsdf ? entry.record.versions.tsdf : entry.record.versions.esdf) = *epoch;
        entry.record.cpu = residency::resident;
        return {};
    }
    operation_status restore(map_snapshot<l, es> snapshot)
    {
        auto& record = snapshot.record;
        if (maps_.contains(record.id) || record.state != lifecycle::finalized || record.writer != writer_kind::none ||
            !finite_pose(record.initial_pose) || !finite_pose(record.optimized_pose) || !record.versions.pose ||
            (!snapshot.tsdf && !snapshot.esdf) || bool(record.versions.tsdf) != bool(snapshot.tsdf) ||
            bool(record.versions.esdf) != bool(snapshot.esdf) ||
            (snapshot.esdf && (!snapshot.tsdf || record.versions.esdf_source != record.versions.tsdf)))
            return std::unexpected(operation_error{"restored submap identity or metadata conflicts"});
        auto admit = [&]<storage::schema s>(const std::shared_ptr<const field_pool<s, l>>& field) -> operation_status {
            if (!field) return {};
            const auto version = s == storage::schema::tsdf ? record.versions.tsdf : record.versions.esdf;
            const auto kind = s == storage::schema::tsdf ? field_kind::tsdf : field_kind::esdf;
            if (!budget_->owns(field->memory) || !field->index.initialized() || field->index.submap() != record.id ||
                field->index.field() != kind || field->payload.size() != field->index.slot_count() ||
                field->payload.size() > maximum_tiles_ || field->content_epoch != version)
                return std::unexpected(operation_error{"restored field identity or admission differs"});
            return {};
        };
        auto checked = admit(snapshot.tsdf);
        if (!checked) return checked;
        checked = admit(snapshot.esdf);
        if (!checked) return checked;
        auto allocation = next_epoch(record.versions.allocation), descriptor = next_epoch(record.versions.descriptor);
        if (!allocation || !descriptor) return std::unexpected(operation_error{"restored allocation epochs are exhausted"});
        record.versions.allocation = *allocation; record.versions.descriptor = *descriptor;
        record.cpu = residency::resident; record.gpu = residency::absent;
        maps_.emplace(record.id, map_entry<l, es>{record, std::move(snapshot.tsdf), std::move(snapshot.esdf)});
        return {};
    }
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
                constexpr auto bytes_per_tile = storage::payload_tile_bytes<selected, l>;
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

        for (std::size_t voxel = 0; voxel < tile_voxels; ++voxel)
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

            for (std::size_t voxel = 0; voxel < tile_voxels; ++voxel)
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
            records.size() / tile_voxels != keys.size() || records.size() % tile_voxels)
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
            storage::scatter_record<s>(view[i / tile_voxels].voxels[i % tile_voxels], records[i]);
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
struct residency_diagnostics
{
    std::size_t active{}, finalized{}, retired{}, cpu_stale_fields{}, gpu_stale_fields{};
    std::size_t pending_reclamation_bytes{}, current_cpu_payload_bytes{};
};
struct pose_publication
{
    std::uint64_t submap{};
    epochs expected;
    pose_record optimized;
};
template<storage::layout l, storage::schema es = storage::schema::esdf> class coordinated_store
{
    struct entry
    {
        map_store<l, es> cpu;
        map_authority authority;
        entry(std::size_t slots, std::shared_ptr<memory_budget> budget) : cpu(slots, std::move(budget)) {}
    };
    mutable std::mutex mutex_;
    residency_policy policy_;
    std::shared_ptr<memory_budget> budget_;
    std::unordered_map<std::uint64_t, std::shared_ptr<entry>> maps_;
    retirement_queue retired_;
#if defined(covoxslam_has_cuda)
    gpu_map_store<l, es> gpu_;
#endif
public:
    operation_status publish_poses(std::span<const pose_publication> poses)
    {
        std::lock_guard lock(mutex_);
        std::unordered_set<std::uint64_t> identities;

        for (const auto& update : poses) {
            auto found = maps_.find(update.submap);
            if (found == maps_.end() || !identities.insert(update.submap).second || !finite_pose(update.optimized))
                return std::unexpected(operation_error{"pose batch has invalid identities or values"});
            const auto& map = *found->second;
            const auto state = map.authority.snapshot();
            const auto* current = map.cpu.find(update.submap);
            if (!current || state.state == lifecycle::retired || state.writer != writer_kind::none ||
                current->record.versions.pose != update.expected.pose || !next_epoch(update.expected.pose) ||
                state.content[0] != update.expected.tsdf || state.content[1] != update.expected.esdf ||
                state.esdf_source != update.expected.esdf_source)
                return std::unexpected(operation_error{"pose batch source snapshot is stale"});
        }

        for (const auto& update : poses) {
            auto& map = *maps_.find(update.submap)->second;
            auto checked = map.cpu.update_pose(update.submap, update.expected.pose, update.optimized);
            if (!checked) return checked;
#if defined(covoxslam_has_cuda)
            auto resident = gpu_.maps_.find(update.submap);
            if (resident != gpu_.maps_.end()) {
                resident->second.record.optimized_pose = update.optimized;
                resident->second.record.versions.pose = update.expected.pose + 1;
            }
#endif
        }
        return {};
    }
    explicit coordinated_store(residency_policy policy = {}) : policy_(policy), budget_(std::make_shared<memory_budget>(policy.limits))
#if defined(covoxslam_has_cuda)
        , gpu_(budget_)
#endif
    {}
    operation_status configure_budget(memory_bytes limits)
    {
        std::lock_guard lock(mutex_);
        auto checked = budget_->configure(limits);
        if (checked) policy_.limits = limits;
        return checked;
    }
    budget_snapshot memory() const { return budget_->snapshot(); }
    template<storage::schema s, class function> operation_status edit(std::uint64_t id, function&& function_body,
        std::uint64_t esdf_source = 0)
    {
        std::lock_guard lock(mutex_);
        auto found = maps_.find(id);
        if (found == maps_.end()) return std::unexpected(operation_error{"mapping submap is absent"});
        auto& map = *found->second;
        auto writer = map.authority.acquire(writer_kind::cpu, s == storage::schema::tsdf ? 1U : 2U);
        if (!writer) return std::unexpected(writer.error());
        return writer->publish(esdf_source, [&](const auto&) {
            return map.cpu.template edit_field<s>(id, std::forward<function>(function_body));
        });
    }
    operation_status save(std::uint64_t id, const std::filesystem::path& destination,
        checkpoint_parameters parameters, std::size_t maximum_bytes) const
    {
        map_snapshot<l, es> selected;
        {
            std::lock_guard lock(mutex_);
            auto found = maps_.find(id);
            if (found == maps_.end()) return std::unexpected(operation_error{"checkpoint submap is absent"});
            const auto state = found->second->authority.snapshot();
            const auto fields = std::uint32_t(bool(state.content[0])) | (std::uint32_t(bool(state.content[1])) << 1);
            if (state.state != lifecycle::finalized || state.cpu != state.content)
                return std::unexpected(operation_error{"checkpoint requires complete finalized CPU storage"});
            const std::array<std::uint64_t, 1> ids{id};
            auto snapshot = found->second->cpu.snapshot(ids, fields, maximum_bytes);
            if (!snapshot) return std::unexpected(snapshot.error());
            selected = std::move(snapshot->front());
            selected.record.versions.esdf_source = state.esdf_source;
        }
        auto bytes = encode_checkpoint(selected, parameters, maximum_bytes);
        if (!bytes) return std::unexpected(bytes.error());
        return save_checkpoint_bytes(destination, *bytes);
    }
    operation_result<std::uint64_t> restore(const std::filesystem::path& source,
        checkpoint_parameters parameters, std::size_t maximum_bytes)
    {
        auto bytes = read_checkpoint_bytes(source, maximum_bytes);
        if (!bytes) return std::unexpected(bytes.error());
        std::lock_guard lock(mutex_);
        auto decoded = decode_checkpoint<l, es>(*bytes, parameters, policy_.maximum_tiles, *budget_);
        if (!decoded) return std::unexpected(decoded.error());
        const auto id = decoded->record.id;
        if (maps_.contains(id)) return std::unexpected(operation_error{"checkpoint identity already exists"});
        auto replacement = std::make_shared<entry>(policy_.maximum_tiles, budget_);
        auto checked = replacement->cpu.restore(std::move(*decoded));
        if (!checked) return std::unexpected(checked.error());
        checked = replacement->authority.restore(replacement->cpu.find(id)->record);
        if (!checked) return std::unexpected(checked.error());
        maps_.emplace(id, std::move(replacement));
        return id;
    }
    operation_result<residency_diagnostics> diagnostics() const
    {
        std::lock_guard lock(mutex_);
        residency_diagnostics result;
        result.pending_reclamation_bytes = retired_.pending_bytes();
#if defined(covoxslam_has_cuda)
        if (gpu_.pending_bytes() > std::numeric_limits<std::size_t>::max() - result.pending_reclamation_bytes)
            return std::unexpected(operation_error{"pending retirement count overflows"});
        result.pending_reclamation_bytes += gpu_.pending_bytes();
#endif

        for (const auto& [id, map] : maps_) {
            const auto state = map->authority.snapshot();
            result.active += state.state == lifecycle::active;
            result.finalized += state.state == lifecycle::finalized;
            result.retired += state.state == lifecycle::retired;
            if (state.state == lifecycle::retired) continue;

            for (unsigned field = 0; field < 2; ++field) {
                result.cpu_stale_fields += state.content[field] && state.cpu[field] != state.content[field];
                result.gpu_stale_fields += state.content[field] && state.gpu[field] != state.content[field];
            }
            auto current = map->cpu.statistics();
            if (!current) return std::unexpected(current.error());
            if (current->payload_capacity_bytes > std::numeric_limits<std::size_t>::max() - result.current_cpu_payload_bytes)
                return std::unexpected(operation_error{"current payload count overflows"});
            result.current_cpu_payload_bytes += current->payload_capacity_bytes;
        }
        return result;
    }
private:
    operation_result<immutable_snapshot<l, es>> lease_locked(std::span<const std::uint64_t> ids, std::uint32_t fields, std::size_t budget) const
    {
        std::vector<map_snapshot<l, es>> selected;

        for (auto id : ids) {
            auto found = maps_.find(id);
            if (found == maps_.end()) return std::unexpected(operation_error{"snapshot submap is absent"});
            const auto state = found->second->authority.snapshot();
            auto dependency = require_current_esdf(state, fields);
            if (!dependency) return std::unexpected(dependency.error());
            for (unsigned field = 0; field < 2; ++field)
                if ((fields & (1U << field)) && state.cpu[field] != state.content[field])
                    return std::unexpected(operation_error{"CPU snapshot requires mirror synchronization"});
            const std::array<std::uint64_t, 1> single{id};
            auto copied = found->second->cpu.snapshot(single, fields, budget);
            if (!copied) return std::unexpected(copied.error());
            copied->front().record.versions.esdf_source = found->second->authority.snapshot().esdf_source;
            selected.push_back(std::move(copied->front()));
        }
        return immutable_snapshot<l, es>::create(std::move(selected), budget);
    }
public:
    operation_result<immutable_snapshot<l, es>> lease(std::span<const std::uint64_t> ids, std::uint32_t fields, std::size_t budget) const
    {
        std::lock_guard lock(mutex_);
        return lease_locked(ids, fields, budget);
    }
    operation_status retire(std::uint64_t id)
    {
        std::lock_guard lock(mutex_);
        auto found = maps_.find(id);
        if (found == maps_.end()) return std::unexpected(operation_error{"submap identity is absent"});
        auto& map = *found->second;
        const auto state = map.authority.snapshot();
        if (state.writer != writer_kind::none || state.state == lifecycle::retired)
            return std::unexpected(operation_error{"retirement conflicts"});
        const auto* entry = map.cpu.find(id);
        if (!entry || !next_epoch(entry->record.versions.descriptor))
            return std::unexpected(operation_error{"retirement descriptor is exhausted"});
        std::vector<retired_allocation> retained;
        auto append = [&]<storage::schema s>(const std::shared_ptr<const field_pool<s, l>>& field) {
            if (!field) return;
            constexpr auto stride = storage::payload_tile_bytes<s, l>;
            retained.push_back({field, field->payload.capacity() * stride});
        };
        append(entry->tsdf); append(entry->esdf);
#if defined(covoxslam_has_cuda)
        auto gpu_retained = gpu_.prepare_retirement(id);
        if (!gpu_retained) return std::unexpected(gpu_retained.error());
        retained.insert(retained.end(), gpu_retained->begin(), gpu_retained->end());
#endif
        auto checked = retired_.retain_many(retained);
        if (!checked) return checked;
#if defined(covoxslam_has_cuda)
        gpu_.finish_retirement(id);
#endif
        checked = map.cpu.retire(id);
        if (!checked) return checked;
        return map.authority.retire();
    }
    std::size_t collect()
    {
        std::lock_guard lock(mutex_);
        auto bytes = retired_.collect();
#if defined(covoxslam_has_cuda)
        bytes += gpu_.collect();
#endif
        return bytes;
    }
    operation_status create(std::uint64_t id, pose_record pose = {})
    {
        std::lock_guard lock(mutex_);
        if (maps_.contains(id)) return std::unexpected(operation_error{"submap identity already exists"});
        auto replacement = std::make_shared<entry>(policy_.maximum_tiles, budget_);
        auto checked = replacement->cpu.create(id, pose);
        if (!checked) return checked;
        maps_.emplace(id, std::move(replacement));
        return {};
    }
    operation_result<authority_snapshot> status(std::uint64_t id) const
    {
        std::lock_guard lock(mutex_);
        auto found = maps_.find(id);
        if (found == maps_.end()) return std::unexpected(operation_error{"submap identity is absent"});
        return found->second->authority.snapshot();
    }
    template<storage::schema s> operation_status import_field(std::uint64_t id, std::span<const block_key> keys,
        std::span<const storage::value<s>> values, std::uint64_t esdf_source = 0)
    {
        std::lock_guard lock(mutex_);
        auto found = maps_.find(id);
        if (found == maps_.end()) return std::unexpected(operation_error{"submap identity is absent"});
        auto& map = *found->second;
        auto writer = map.authority.acquire(writer_kind::cpu, s == storage::schema::tsdf ? 1U : 2U);
        if (!writer) return std::unexpected(writer.error());
        return writer->publish(esdf_source, [&](const auto&) {
            return map.cpu.template import_field<s>(id, keys, values);
        });
    }
    template<storage::schema s> operation_result<tile_handle> insert(block_key key)
    {
        std::lock_guard lock(mutex_);
        auto found = maps_.find(key.submap);
        if (found == maps_.end()) return std::unexpected(operation_error{"submap identity is absent"});
        auto& map = *found->second;
        auto state = map.authority.snapshot();
        if (state.state != lifecycle::active) return std::unexpected(operation_error{"insertion requires an active submap"});
        auto existing = map.cpu.template lookup<s>(key);
        if (existing) return existing;
        auto writer = map.authority.acquire(writer_kind::cpu, s == storage::schema::tsdf ? 1U : 2U);
        if (!writer) return std::unexpected(writer.error());
        operation_result<tile_handle> result = std::unexpected(operation_error{"insertion did not publish"});
        auto checked = writer->publish(state.esdf_source, [&](const auto&) -> operation_status {
            result = map.cpu.template insert<s>(key);
            if (!result) return std::unexpected(result.error());
            return {};
        });
        if (!checked) return std::unexpected(checked.error());
        return result;
    }
    operation_status erase(tile_handle handle)
    {
        std::lock_guard lock(mutex_);
        auto found = maps_.find(handle.submap);
        if (found == maps_.end()) return std::unexpected(operation_error{"submap identity is absent"});
        auto& map = *found->second;
        const auto source = map.authority.snapshot().esdf_source;
        auto writer = map.authority.acquire(writer_kind::cpu, handle.field == field_kind::tsdf ? 1U : 2U);
        if (!writer) return std::unexpected(writer.error());
        return writer->publish(source, [&](const auto&) { return map.cpu.erase(handle); });
    }
    operation_result<map_snapshot<l, es>> snapshot(std::uint64_t id, std::uint32_t fields, std::size_t budget) const
    {
        std::lock_guard lock(mutex_);
        auto found = maps_.find(id);
        if (found == maps_.end()) return std::unexpected(operation_error{"submap identity is absent"});
        const std::array<std::uint64_t, 1> ids{id};
        const auto state = found->second->authority.snapshot();
        auto dependency = require_current_esdf(state, fields);
        if (!dependency) return std::unexpected(dependency.error());

        for (unsigned field = 0; field < 2; ++field)
            if ((fields & (1U << field)) && state.cpu[field] != state.content[field])
                return std::unexpected(operation_error{"CPU snapshot requires mirror synchronization"});
        auto result = found->second->cpu.snapshot(ids, fields, budget);
        if (!result) return std::unexpected(result.error());
        result->front().record.versions.esdf_source = found->second->authority.snapshot().esdf_source;
        return std::move(result->front());
    }
    operation_status finalize(std::uint64_t id, std::uint32_t fields)
    {
        std::lock_guard lock(mutex_);
        auto found = maps_.find(id);
        if (found == maps_.end()) return std::unexpected(operation_error{"submap identity is absent"});
        auto& map = *found->second;
        const auto source = map.authority.snapshot().esdf_source;
        return map.authority.finalize(fields, [&] { return map.cpu.seal(id, fields, source); });
    }
#if defined(covoxslam_has_cuda)
    operation_status import_device_tsdf(std::uint64_t id, pose_record pose,
        std::span<const block_key> keys, const storage::device_buffer<storage::schema::tsdf, l>& source)
    {
        std::lock_guard lock(mutex_);

        if (maps_.contains(id) || gpu_.maps_.contains(id) || keys.size() > policy_.maximum_tiles ||
            keys.size() > source.size() || !finite_pose(pose))
            return std::unexpected(operation_error{"resident submap admission is invalid"});
        auto descriptor = source.descriptor();
        if (!descriptor) return std::unexpected(descriptor.error());
        auto replacement = std::make_shared<entry>(policy_.maximum_tiles, budget_);
        auto checked = replacement->cpu.create(id, pose);
        if (!checked) return checked;
        checked = replacement->cpu.template initialize_topology<storage::schema::tsdf>(id, keys, 1);
        if (!checked) return checked;
        checked = replacement->cpu.template initialize_topology<es>(id, keys, 1, 1);
        if (!checked) return checked;
        const auto* metadata = replacement->cpu.find(id);
        auto tsdf = gpu_.template upload<storage::schema::tsdf>(metadata->tsdf,
            metadata->record.versions.allocation, false, &source);
        if (!tsdf) return std::unexpected(tsdf.error());
        auto esdf = gpu_.template upload<es>(metadata->esdf, metadata->record.versions.allocation, false);

        if (!esdf) return std::unexpected(esdf.error());
        auto writer = replacement->authority.acquire(writer_kind::gpu, 3);
        if (!writer) return std::unexpected(writer.error());
        checked = writer->publish(1, [](const auto&) -> operation_status { return {}; });
        if (!checked) return checked;
        resident_entry<l, es> resident{metadata->record, std::move(*tsdf), std::move(*esdf)};
         resident.record.cpu = residency::absent;
        resident.record.gpu = residency::resident;
        maps_.emplace(id, std::move(replacement));
        gpu_.maps_.emplace(id, std::move(resident));
        return {};
    }
    operation_result<std::vector<resident_entry<l, es>>> lease_resident(
        std::span<const std::uint64_t> ids, std::uint32_t fields, std::size_t budget) const
    {
        std::lock_guard lock(mutex_);
        std::vector<resident_entry<l, es>> result;
        std::unordered_set<std::uint64_t> unique;
        if (!fields || (fields & ~3U)) return std::unexpected(operation_error{"invalid resident field selection"});

        for (auto id : ids) {
            auto found = maps_.find(id);
            if (found == maps_.end() || !unique.insert(id).second)
                return std::unexpected(operation_error{"resident snapshot identity is invalid"});
            const auto state = found->second->authority.snapshot();
            auto valid = require_current_esdf(state, fields);
            if (!valid) return std::unexpected(valid.error());
            auto resident = gpu_.maps_.find(id);
            if (state.state != lifecycle::finalized || state.writer != writer_kind::none || resident == gpu_.maps_.end())
                return std::unexpected(operation_error{"resident PGO requires finalized maps"});
            auto selected = resident->second;
            selected.record = found->second->cpu.find(id)->record;
            selected.record.versions.tsdf = state.content[0];
            selected.record.versions.esdf = state.content[1];
            selected.record.versions.esdf_source = state.esdf_source;
            const auto admit = [&]<storage::schema s>(const std::shared_ptr<const gpu_field<s, l>>& field) -> operation_status {
                constexpr auto ordinal = s == storage::schema::tsdf ? 0 : 1;
                if (!(fields & (1U << ordinal))) return {};
                if (!field || !state.content[ordinal] || state.gpu[ordinal] != state.content[ordinal] ||
                    field->content_epoch != state.content[ordinal])
                    return std::unexpected(operation_error{"resident snapshot requires current GPU fields"});
                auto bytes = gpu_.retained_field(field);
                if (!bytes) return std::unexpected(bytes.error());
                if (bytes->bytes > budget) return std::unexpected(operation_error{"resident snapshot budget exceeded"});
                budget -= bytes->bytes;
                return {};
            };
            valid = admit(selected.tsdf);
            if (!valid) return std::unexpected(valid.error());
            valid = admit(selected.esdf);
            if (!valid) return std::unexpected(valid.error());
            if (!(fields & 1U)) selected.tsdf.reset();
            if (!(fields & 2U)) selected.esdf.reset();
            result.push_back(std::move(selected));
        }
        return result;
    }
    template<class function> operation_status derive_gpu(std::uint64_t id, function&& consumer, bool preserve_values = true)
    {
        std::lock_guard lock(mutex_);
        auto found = maps_.find(id);
        if (found == maps_.end()) return std::unexpected(operation_error{"ESDF derivation submap is absent"});
        auto& map = *found->second;
        const auto state = map.authority.snapshot();
        if (!state.content[0] || state.gpu[0] != state.content[0])
            return std::unexpected(operation_error{"ESDF derivation requires the current GPU TSDF"});
        auto source = gpu_.template field<storage::schema::tsdf>(id);
        if (!source) return std::unexpected(source.error());
        auto read_view = (*source)->payload.view();
        auto read_index = (*source)->index.view();
        if (!read_view) return std::unexpected(read_view.error());
        if (!read_index) return std::unexpected(read_index.error());
        auto metadata = (*source)->source_identity.lock();
        if (!metadata) return std::unexpected(operation_error{"ESDF source topology is unavailable"});
        std::vector<block_key> keys;
        keys.reserve(metadata->index.size());

        for (const auto& slot : metadata->index.records())
            if (slot.occupied) keys.push_back(slot.key);
        std::sort(keys.begin(), keys.end(), [](auto a, auto b) {
            return a.z < b.z || (a.z == b.z && (a.y < b.y || (a.y == b.y && a.x < b.x)));
        });
        auto writer = map.authority.acquire(writer_kind::gpu, 2U);
        if (!writer) return std::unexpected(writer.error());
        return writer->publish(state.content[0], [&](const auto& versions) {
            return gpu_.template edit<es>(id, versions[1], state.content[0], [&](auto target, auto index, auto stream) {
                return consumer(*read_view, *read_index, target, index, stream, std::span<const block_key>(keys), *budget_);
            }, preserve_values);
        });
    }
    template<storage::schema s, class function> operation_status edit_gpu(std::uint64_t id, function&& consumer,
        std::uint64_t esdf_source = 0)
    {
        std::lock_guard lock(mutex_);
        auto found = maps_.find(id);
        if (found == maps_.end()) return std::unexpected(operation_error{"GPU writer submap is absent"});
        auto writer = found->second->authority.acquire(writer_kind::gpu, s == storage::schema::tsdf ? 1U : 2U);
        if (!writer) return std::unexpected(writer.error());
        return writer->publish(esdf_source, [&](const auto& versions) {
            return gpu_.template edit<s>(id, versions[s == storage::schema::tsdf ? 0 : 1], esdf_source,
                std::forward<function>(consumer));
        });
    }
private:
    template<storage::schema s> operation_status synchronize_cpu_locked(std::uint64_t id)
    {
        auto found = maps_.find(id);
        if (found == maps_.end()) return std::unexpected(operation_error{"CPU mirror submap is absent"});
        auto& map = *found->second;
        auto ticket = map.authority.transfer(writer_kind::gpu, writer_kind::cpu, s == storage::schema::tsdf ? 1U : 2U);
        if (!ticket) return std::unexpected(ticket.error());
        const auto source = map.authority.snapshot().esdf_source;
        auto resident = gpu_.template field<s>(id);
        if (!resident) return std::unexpected(resident.error());
        auto copied = gpu_.template download<s>(id);
        if (!copied) return std::unexpected(copied.error());
        return ticket->acknowledge([&]() -> operation_status {
            auto checked = map.cpu.template install_mirror<s>(id, *copied, source);
            if (checked) (*resident)->source_identity = *copied;
            return checked;
        });
    }
public:
    template<storage::schema s> operation_status synchronize_cpu(std::uint64_t id)
    {
        std::lock_guard lock(mutex_);
        return synchronize_cpu_locked<s>(id);
    }
    operation_result<immutable_snapshot<l, es>> lease_readback(std::span<const std::uint64_t> ids,
        std::uint32_t fields, std::size_t payload_budget, std::size_t readback_budget)
    {
        std::lock_guard lock(mutex_);
        if (ids.empty() || !fields || (fields & ~3U)) return std::unexpected(operation_error{"invalid geometry selection"});
        std::unordered_set<std::uint64_t> unique;
        std::size_t download_bytes{}, payload_bytes{};

        for (auto id : ids) {
            auto found = maps_.find(id);
            if (found == maps_.end() || !unique.insert(id).second)
                return std::unexpected(operation_error{"geometry submap is absent or duplicated"});
            const auto state = found->second->authority.snapshot();
            auto checked = require_current_esdf(state, fields);
            if (!checked) return std::unexpected(checked.error());
            if (state.state == lifecycle::retired || state.writer != writer_kind::none)
                return std::unexpected(operation_error{"geometry submap is unavailable"});
            const auto* cpu = found->second->cpu.find(id);
            auto admit = [&]<storage::schema s>(unsigned field) -> operation_status {
                if (!(fields & (1U << field))) return {};
                if (!state.content[field]) return std::unexpected(operation_error{"geometry field is absent"});
                constexpr auto stride = storage::payload_tile_bytes<s, l>;
                const auto& host = select_field<s, es>(*cpu);
                std::size_t capacity = host ? host->payload.capacity() : 0;
                if (state.cpu[field] != state.content[field]) {
                    if (state.gpu[field] != state.content[field]) return std::unexpected(operation_error{"geometry has no current replica"});
                    auto resident = gpu_.template field<s>(id);
                    if (!resident) return std::unexpected(resident.error());
                    capacity = (*resident)->payload.size();
                    constexpr auto record_bytes = storage::canonical_tile_bytes<s>;
                    if (capacity > (readback_budget - download_bytes) / record_bytes)
                        return std::unexpected(operation_error{"geometry readback limit exceeded"});
                    download_bytes += capacity * record_bytes;
                }
                if (capacity > (payload_budget - payload_bytes) / stride)
                    return std::unexpected(operation_error{"geometry snapshot limit exceeded"});
                payload_bytes += capacity * stride;
                return {};
            };
            checked = admit.template operator()<storage::schema::tsdf>(0);
            if (!checked) return std::unexpected(checked.error());
            checked = admit.template operator()<es>(1);
            if (!checked) return std::unexpected(checked.error());
        }

        for (auto id : ids) {
            const auto state = maps_.find(id)->second->authority.snapshot();
            if ((fields & 1U) && state.cpu[0] != state.content[0]) {
                auto checked = synchronize_cpu_locked<storage::schema::tsdf>(id);
                if (!checked) return std::unexpected(checked.error());
            }
            if ((fields & 2U) && state.cpu[1] != state.content[1]) {
                auto checked = synchronize_cpu_locked<es>(id);
                if (!checked) return std::unexpected(checked.error());
            }
        }
        return lease_locked(ids, fields, payload_budget);
    }
    operation_status evict_gpu(std::uint64_t id)
    {
        std::lock_guard lock(mutex_);
        auto found = maps_.find(id);
        if (found == maps_.end()) return std::unexpected(operation_error{"submap identity is absent"});
        auto& map = *found->second;
        const auto state = map.authority.snapshot();
        return map.authority.evict(writer_kind::gpu, [&] { return gpu_.evict(id, state); });
    }
    operation_status synchronize_gpu(std::uint64_t id, std::uint32_t fields, std::size_t budget)
    {
        std::lock_guard lock(mutex_);
        auto found = maps_.find(id);
        if (found == maps_.end()) return std::unexpected(operation_error{"submap identity is absent"});
        auto& map = *found->second;
        auto ticket = map.authority.transfer(writer_kind::cpu, writer_kind::gpu, fields);
        if (!ticket) return std::unexpected(ticket.error());
        const std::array<std::uint64_t, 1> ids{id};
        auto source = map.cpu.snapshot(ids, fields, budget);
        if (!source) return std::unexpected(source.error());
        source->front().record.versions.esdf_source = map.authority.snapshot().esdf_source;
        return ticket->acknowledge([&] { return gpu_.publish(source->front(), fields); });
    }
    template<storage::schema s, class function> operation_result<storage::completion_ticket> read_gpu(
        std::uint64_t id, cudaStream_t stream, function&& consumer) const
    {
        std::lock_guard lock(mutex_);
        auto found = maps_.find(id);
        if (found == maps_.end()) return std::unexpected(operation_error{"submap identity is absent"});
        const auto state = found->second->authority.snapshot();
        const unsigned field = s == storage::schema::tsdf ? 0 : 1;
        auto dependency = require_current_esdf(state, 1U << field);
        if (!dependency) return std::unexpected(dependency.error());
        if (!state.content[field] || state.gpu[field] != state.content[field])
            return std::unexpected(operation_error{"GPU mirror is not current"});
        return gpu_.template read_on<s>(id, stream, std::forward<function>(consumer));
    }
#endif
};
}
