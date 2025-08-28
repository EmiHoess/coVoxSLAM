#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/storage/device_buffer.hpp>
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/store/store.hpp>
#include <covoxslam/store/gpu_index.hpp>
#include <covoxslam/store/snapshot.hpp>
#include <covoxslam/map_store/authority.hpp>
#include <covoxslam/storage/transfers.hpp>

namespace covoxslam::store
{
template<storage::schema s, storage::layout l> struct gpu_field
{
    std::shared_ptr<budget_claim> memory;
    storage::device_buffer<s, l> payload;
    gpu_block_index index;
    std::uint64_t content_epoch{}, allocation_generation{};
    mutable std::weak_ptr<const field_pool<s, l>> source_identity;
};
template<storage::layout l, storage::schema es = storage::schema::esdf> struct resident_entry
{
    submap_record record;
    std::shared_ptr<const gpu_field<storage::schema::tsdf, l>> tsdf;
    std::shared_ptr<const gpu_field<es, l>> esdf;
};
template<storage::layout l, storage::schema es = storage::schema::esdf> class gpu_map_store
{
    template<storage::layout, storage::schema> friend class coordinated_store;
    std::unordered_map<std::uint64_t, resident_entry<l, es>> maps_;
    retirement_queue retired_;
    std::shared_ptr<memory_budget> budget_;
    template<storage::schema s> operation_result<retired_allocation> retained_field(const std::shared_ptr<const gpu_field<s, l>>& field) const
    {
        auto payload = field->payload.descriptor();
        if (!payload) return std::unexpected(payload.error());
        auto index = field->index.view();
        if (!index) return std::unexpected(index.error());
        const auto payload_bytes = payload->capacity_tiles * payload->tile_stride * (l == storage::layout::pool_soa ? storage::field_count<s> : 1);
        const auto index_bytes = index->capacity * sizeof(device_hash_entry);
        if (payload_bytes > std::numeric_limits<std::size_t>::max() - index_bytes)
            return std::unexpected(operation_error{"resident byte accounting overflows"});
        return retired_allocation{field, payload_bytes + index_bytes};
    }
    template<storage::schema s> operation_result<std::shared_ptr<const gpu_field<s, l>>> upload(
        const std::shared_ptr<const field_pool<s, l>>& source, std::uint64_t generation, bool copy_values = true,
        const storage::device_buffer<s, l>* device_source = nullptr)
    {
        if (!source || source->index.slot_count() != source->payload.size())
            return std::unexpected(operation_error{"resident field has invalid initialized slots"});
        if (device_source && (copy_values || device_source->size() < source->payload.size()))
            return std::unexpected(operation_error{"device source extent or transfer mode is invalid"});
        if (device_source) {
            auto descriptor = device_source->descriptor();
            if (!descriptor) return std::unexpected(descriptor.error());

            if (descriptor->record_schema != s || descriptor->storage_layout != l)
                return std::unexpected(operation_error{"device source schema differs from destination"});
        }
        constexpr auto stride = storage::payload_tile_bytes<s, l>;
        std::size_t buckets{2};
        if (source->index.size() > std::size_t(std::numeric_limits<std::uint32_t>::max()) / 2 ||
            source->payload.size() > std::numeric_limits<std::size_t>::max() / stride)
            return std::unexpected(operation_error{"resident capacity is unsupported"});
        while (buckets / 2 < source->index.size()) buckets *= 2;
        const auto index_bytes = buckets * sizeof(device_hash_entry);
        auto bytes = add_memory({0, 0, source->payload.size() * stride}, {0, 0, index_bytes});
        if (!bytes) return std::unexpected(bytes.error());
        auto retained = budget_->reserve(*bytes);
        if (!retained) return std::unexpected(retained.error());
        const auto input_bytes = source->index.size() * sizeof(device_hash_entry);
        auto transient = budget_->reserve({input_bytes, 0, input_bytes + sizeof(unsigned)});
        if (!transient) return std::unexpected(transient.error());
        auto field = std::make_shared<gpu_field<s, l>>();
        field->memory = std::move(*retained);
        auto checked = field->payload.initialize(source->payload.size());
        if (!checked) return std::unexpected(checked.error());
        storage::transfer_measurement measured;
        if (copy_values) checked = field->payload.upload_range(source->payload, 0, 0, source->payload.size() * 512,
            (1U << storage::field_count<s>) - 1, storage::transfer_method::bulk_same_layout, measured);
        if (device_source)
            checked = field->payload.copy_from(*device_source, 0, 0, source->payload.size() * 512,
                (1U << storage::field_count<s>) - 1, storage::transfer_method::bulk_same_layout, measured);
        if (!checked) return std::unexpected(checked.error());
        checked = field->index.initialize(source->index, generation);
        if (!checked) return std::unexpected(checked.error());
        field->content_epoch = source->content_epoch;
        field->allocation_generation = generation;
        field->source_identity = source;
        return field;
    }
    operation_result<std::vector<retired_allocation>> prepare_retirement(std::uint64_t id) const
    {
        auto found = maps_.find(id);
        std::vector<retired_allocation> entries;
        if (found == maps_.end()) return entries;
        if (found->second.tsdf) {
            auto retained = retained_field(found->second.tsdf);
            if (!retained) return std::unexpected(retained.error());
            entries.push_back(std::move(*retained));
        }
        if (found->second.esdf) {
            auto retained = retained_field(found->second.esdf);
            if (!retained) return std::unexpected(retained.error());
            entries.push_back(std::move(*retained));
        }
        return entries;
    }
    void finish_retirement(std::uint64_t id)
    {
        auto found = maps_.find(id);
        if (found == maps_.end()) return;
        if (found->second.tsdf && found->second.tsdf->memory) found->second.tsdf->memory->retire();
        if (found->second.esdf && found->second.esdf->memory) found->second.esdf->memory->retire();
        maps_.erase(found);
    }
public:
    explicit gpu_map_store(std::shared_ptr<memory_budget> budget = {})
        : budget_(budget ? std::move(budget) : std::make_shared<memory_budget>(residency_policy{}.limits)) {}
    budget_snapshot memory() const { return budget_->snapshot(); }
    template<storage::schema s> operation_result<std::shared_ptr<const field_pool<s, l>>> download(std::uint64_t id)
    {
        auto resident = field<s>(id);
        if (!resident) return std::unexpected(resident.error());
        auto metadata = (*resident)->source_identity.lock();
        if (!metadata) return std::unexpected(operation_error{"GPU readback topology is unavailable"});
        constexpr auto stride = storage::payload_tile_bytes<s, l>;
        auto memory = budget_->reserve({metadata->payload.size() * stride, 0, 0});
        if (!memory) return std::unexpected(memory.error());
        auto candidate = std::make_shared<field_pool<s, l>>();
        candidate->memory = std::move(*memory);
        auto checked = candidate->payload.initialize(metadata->payload.size());
        if (!checked) return std::unexpected(checked.error());
        candidate->index = metadata->index;
        storage::transfer_measurement measured;
        checked = (*resident)->payload.download_range(candidate->payload, 0, 0, candidate->payload.size() * 512,
            (1U << storage::field_count<s>) - 1, storage::transfer_method::bulk_same_layout, measured);
        if (!checked) return std::unexpected(checked.error());
        candidate->content_epoch = (*resident)->content_epoch;
        return candidate;
    }
    template<storage::schema s, class function> operation_status edit(std::uint64_t id,
        std::uint64_t content_epoch, std::uint64_t esdf_source, function&& consumer, bool preserve_values = true)
    {
        auto previous = field<s>(id);
        if (!previous) return std::unexpected(previous.error());
        auto& entry = maps_.at(id);
        auto source = (*previous)->source_identity.lock();
        auto generation = next_epoch(entry.record.versions.allocation);
        auto descriptor = next_epoch(entry.record.versions.descriptor);
        if (!source || entry.record.state != lifecycle::active || !generation || !descriptor ||
            content_epoch <= (*previous)->content_epoch)
            return std::unexpected(operation_error{"GPU writer source or epoch is invalid"});
        auto replacement = upload<s>(source, *generation, false);
        if (!replacement) return std::unexpected(replacement.error());
        auto candidate = std::const_pointer_cast<gpu_field<s, l>>(*replacement);
        storage::transfer_measurement copied;
        operation_status checked;
        if (preserve_values)
            checked = candidate->payload.copy_from((*previous)->payload, 0, 0, source->payload.size() * 512,
                (1U << storage::field_count<s>) - 1, storage::transfer_method::bulk_same_layout, copied);
        if (!checked) return checked;
        auto view = candidate->payload.view();
        auto index = candidate->index.view();
        if (!view) return std::unexpected(view.error());
        if (!index) return std::unexpected(index.error());
        storage::stream_drain drain{candidate->payload.stream(), index->device};
        checked = consumer(*view, *index, candidate->payload.stream());
        if (!checked) return checked;
        checked = storage::cuda_storage_status(cudaStreamSynchronize(candidate->payload.stream()));
        if (!checked) return checked;
        drain.armed = false;
        auto retired = retained_field(*previous);
        if (!retired) return std::unexpected(retired.error());
        checked = retired_.retain(retired->owner, retired->bytes);
        if (!checked) return checked;
        candidate->content_epoch = content_epoch;
        if ((*previous)->memory) (*previous)->memory->retire();
        if constexpr (s == storage::schema::tsdf) entry.tsdf = std::move(candidate);
        else entry.esdf = std::move(candidate);
        (s == storage::schema::tsdf ? entry.record.versions.tsdf : entry.record.versions.esdf) = content_epoch;
        if constexpr (s != storage::schema::tsdf) entry.record.versions.esdf_source = esdf_source;
        entry.record.versions.allocation = *generation; entry.record.versions.descriptor = *descriptor;
        return {};
    }
    std::size_t size() const noexcept { return maps_.size(); }
    std::size_t collect() { return retired_.collect(); }
    std::size_t pending_bytes() const noexcept { return retired_.pending_bytes(); }
    operation_status evict(std::uint64_t id, const authority_snapshot& authority)
    {
        auto found = maps_.find(id);
        if (authority.state != lifecycle::finalized || authority.writer != writer_kind::none)
            return std::unexpected(operation_error{"GPU eviction requires a finalized submap"});
        if (found == maps_.end()) return {};
        auto available = [&]<storage::schema s>(const std::shared_ptr<const gpu_field<s, l>>& field, unsigned i) -> operation_status {
            if (!field) return {};
            if (field.use_count() != 1) return std::unexpected(operation_error{"GPU allocation is pinned"});
            if (authority.cpu[i] != authority.content[i] || field->content_epoch != authority.content[i])
                return std::unexpected(operation_error{"GPU eviction has no current CPU replica"});
            auto recovery = field->source_identity.lock();
            if (!recovery || recovery->content_epoch != authority.content[i] ||
                recovery->index.submap() != id || recovery->index.slot_count() != recovery->payload.size())
                return std::unexpected(operation_error{"GPU eviction has no retained lossless recovery field"});
            if (recovery->index.field() != (i ? field_kind::esdf : field_kind::tsdf))
                return std::unexpected(operation_error{"GPU recovery field identity conflicts"});
            auto descriptor = field->payload.descriptor();
            if (!descriptor) return std::unexpected(descriptor.error());
            if (descriptor->capacity_tiles < recovery->payload.size())
                return std::unexpected(operation_error{"GPU recovery field size conflicts"});
            return {};
        };
        auto checked = available(found->second.tsdf, 0);
        if (!checked) return checked;
        checked = available(found->second.esdf, 1);
        if (!checked) return checked;
        found->second.tsdf.reset();
        found->second.esdf.reset();
        found->second.record.gpu = residency::evicted;
        return {};
    }
    operation_status retire(std::uint64_t id)
    {
        auto entries = prepare_retirement(id);
        if (!entries) return std::unexpected(entries.error());
        auto checked = retired_.retain_many(*entries);
        if (!checked) return checked;
        finish_retirement(id);
        return {};
    }
    operation_status publish(const map_snapshot<l, es>& source, std::uint32_t fields)
    {
        if (!fields || (fields & ~3U) || source.record.state == lifecycle::retired)
            return std::unexpected(operation_error{"invalid resident publication"});
        resident_entry<l, es> candidate;
        auto old = maps_.find(source.record.id);
        if (old != maps_.end()) candidate = old->second;
        if (old != maps_.end() && (source.record.versions.pose < old->second.record.versions.pose ||
            source.record.versions.descriptor < old->second.record.versions.descriptor))
            return std::unexpected(operation_error{"resident metadata publication is stale"});
        const auto prepare = [&]<storage::schema s>(const auto& input, auto& output) -> operation_status {
            if (!input) return std::unexpected(operation_error{"resident publication field is absent"});
            const auto expected = s == storage::schema::tsdf ? source.record.versions.tsdf : source.record.versions.esdf;
            const auto kind = s == storage::schema::tsdf ? field_kind::tsdf : field_kind::esdf;
            if (!expected || input->content_epoch != expected || input->index.submap() != source.record.id || input->index.field() != kind)
                return std::unexpected(operation_error{"resident publication identity or version differs"});
            if (output && output->content_epoch > input->content_epoch)
                return std::unexpected(operation_error{"resident publication is stale"});
            if (output && output->content_epoch == input->content_epoch) {
                if (output->source_identity.lock() != input)
                    return std::unexpected(operation_error{"resident epoch refers to different source storage"});
                auto device = output->payload.descriptor();
                if (!device) return std::unexpected(device.error());
                return {};
            }
            auto replacement = upload<s>(input, source.record.versions.allocation);
            if (!replacement) return std::unexpected(replacement.error());
            output = std::move(*replacement);
            return {};
        };
        if (fields & 1U) {
            auto checked = prepare.template operator()<storage::schema::tsdf>(source.tsdf, candidate.tsdf);
            if (!checked) return checked;
        }
        if (fields & 2U) {
            auto checked = prepare.template operator()<es>(source.esdf, candidate.esdf);
            if (!checked) return checked;
        }
        candidate.record = source.record;
        candidate.record.versions.tsdf = candidate.tsdf ? candidate.tsdf->content_epoch : 0;
        candidate.record.versions.esdf = candidate.esdf ? candidate.esdf->content_epoch : 0;
        if (!(fields & 2U)) candidate.record.versions.esdf_source = old == maps_.end() ? 0 : old->second.record.versions.esdf_source;
        candidate.record.gpu = residency::resident;
        std::vector<retired_allocation> previous;
        if (old != maps_.end()) {
            if (old->second.tsdf && old->second.tsdf != candidate.tsdf) {
                auto retained = retained_field(old->second.tsdf);
                if (!retained) return std::unexpected(retained.error());
                previous.push_back(std::move(*retained));
            }
            if (old->second.esdf && old->second.esdf != candidate.esdf) {
                auto retained = retained_field(old->second.esdf);
                if (!retained) return std::unexpected(retained.error());
                previous.push_back(std::move(*retained));
            }
        }
        auto retained = retired_.retain_many(previous);
        if (!retained) return retained;
        if (old != maps_.end()) {
            if (old->second.tsdf && old->second.tsdf != candidate.tsdf && old->second.tsdf->memory)
                old->second.tsdf->memory->retire();
            if (old->second.esdf && old->second.esdf != candidate.esdf && old->second.esdf->memory)
                old->second.esdf->memory->retire();
        }
        maps_.insert_or_assign(source.record.id, std::move(candidate));
        return {};
    }
    template<storage::schema s> operation_result<std::shared_ptr<const gpu_field<s, l>>> field(std::uint64_t id) const
    {
        const auto found = maps_.find(id);
        if (found == maps_.end()) return std::unexpected(operation_error{"GPU submap is absent"});
        auto selected = select_field<s, es>(found->second);
        if (!selected) return std::unexpected(operation_error{"GPU field is absent"});
        auto descriptor = selected->payload.descriptor();
        if (!descriptor) return std::unexpected(descriptor.error());
        return selected;
    }
    template<storage::schema s, class function> operation_result<storage::completion_ticket> read_on(
        std::uint64_t id, cudaStream_t stream, function&& consumer) const
    {
        auto selected = field<s>(id);
        if (!selected) return std::unexpected(selected.error());
        auto view = (*selected)->payload.view();
        auto index = (*selected)->index.view();
        if (!view) return std::unexpected(view.error());
        if (!index) return std::unexpected(index.error());
        auto retained = std::make_shared<std::shared_ptr<const gpu_field<s, l>>>(*selected);
        storage::stream_drain drain{stream, index->device};
        auto checked = consumer(*view, *index, stream);
        if (!checked) return std::unexpected(checked.error());
        auto completion = storage::completion_ticket::record(stream, retained);
        if (!completion) return std::unexpected(completion.error());
        drain.armed = false;
        return completion;
    }
};
}
