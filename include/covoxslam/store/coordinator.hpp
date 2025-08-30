#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/map_store/coordinator.hpp>
#include <covoxslam/map_store/authority.hpp>
#include <covoxslam/map_store/snapshot.hpp>
#include <covoxslam/map_store/checkpoint.hpp>
#if defined(covoxslam_has_cuda)
#include <covoxslam/storage/device_buffer.hpp>
#include <covoxslam/store/gpu_store.hpp>
#endif

namespace covoxslam::store
{
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
#include <covoxslam/map_store/detail/coordinator_state.inl>
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
