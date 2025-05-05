#pragma once
#include <covoxslam/store/store.hpp>
#include <covoxslam/store/authority.hpp>
#include <covoxslam/store/snapshot.hpp>
#include <covoxslam/store/checkpoint.hpp>
#if defined(covoxslam_has_cuda)
#include <covoxslam/store/gpu_store.hpp>
#endif

namespace covoxslam::store
{
struct residency_diagnostics
{
    std::size_t active{}, finalized{}, retired{}, cpu_stale_fields{}, gpu_stale_fields{};
    std::size_t pending_reclamation_bytes{}, current_cpu_payload_bytes{};
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
    operation_result<immutable_snapshot<l, es>> lease(std::span<const std::uint64_t> ids, std::uint32_t fields, std::size_t budget) const
    {
        std::lock_guard lock(mutex_);
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
            constexpr auto stride = storage::storage_traits<s, l>::tile_stride * (l == storage::layout::pool_soa ? storage::field_count<s> : 1);
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
    template<class function> operation_status derive_gpu(std::uint64_t id, function&& consumer)
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
        auto writer = map.authority.acquire(writer_kind::gpu, 2U);
        if (!writer) return std::unexpected(writer.error());
        return writer->publish(state.content[0], [&](const auto& versions) {
            return gpu_.template edit<es>(id, versions[1], state.content[0], [&](auto target, auto index, auto stream) {
                return consumer(*read_view, *read_index, target, index, stream);
            });
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
    template<storage::schema s> operation_status synchronize_cpu(std::uint64_t id)
    {
        std::lock_guard lock(mutex_);
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
