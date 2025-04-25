#pragma once
#include <covoxslam/store/store.hpp>
#include <covoxslam/store/gpu_index.hpp>
#include <covoxslam/store/snapshot.hpp>
#include <covoxslam/store/authority.hpp>
#include <covoxslam/storage/completion.hpp>

namespace covoxslam::store
{
template<storage::schema s, storage::layout l> struct gpu_field
{
    storage::cuda_draft<s, l> payload;
    gpu_block_index index;
    std::uint64_t content_epoch{}, allocation_generation{};
    std::weak_ptr<const field_pool<s, l>> source_identity;
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
        const std::shared_ptr<const field_pool<s, l>>& source, std::uint64_t generation)
    {
        if (!source || source->index.slot_count() != source->payload.size())
            return std::unexpected(operation_error{"resident field has invalid initialized slots"});
        auto field = std::make_shared<gpu_field<s, l>>();
        auto checked = field->payload.initialize(source->payload.size());
        if (!checked) return std::unexpected(checked.error());
        storage::transfer_measurement measured;
        checked = field->payload.upload_range(source->payload, 0, 0, source->payload.size() * 512,
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
    void finish_retirement(std::uint64_t id) { maps_.erase(id); }
public:
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
        maps_.insert_or_assign(source.record.id, std::move(candidate));
        return {};
    }
    template<storage::schema s> operation_result<std::shared_ptr<const gpu_field<s, l>>> field(std::uint64_t id) const
    {
        const auto found = maps_.find(id);
        if (found == maps_.end()) return std::unexpected(operation_error{"GPU submap is absent"});
        auto selected = [&]() { if constexpr (s == storage::schema::tsdf) return found->second.tsdf; else return found->second.esdf; }();
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
