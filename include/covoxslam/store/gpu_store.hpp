#pragma once
#include <covoxslam/store/store.hpp>
#include <covoxslam/store/gpu_index.hpp>
#include <covoxslam/storage/completion.hpp>

namespace covoxslam::store
{
template<storage::schema s, storage::layout l> struct gpu_field
{
    storage::cuda_draft<s, l> payload;
    gpu_block_index index;
    std::uint64_t content_epoch{}, allocation_generation{};
};
template<storage::layout l, storage::schema es = storage::schema::esdf> struct resident_entry
{
    submap_record record;
    std::shared_ptr<const gpu_field<storage::schema::tsdf, l>> tsdf;
    std::shared_ptr<const gpu_field<es, l>> esdf;
};
template<storage::layout l, storage::schema es = storage::schema::esdf> class gpu_map_store
{
    std::unordered_map<std::uint64_t, resident_entry<l, es>> maps_;
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
        return field;
    }
public:
    std::size_t size() const noexcept { return maps_.size(); }
    operation_status publish(const map_snapshot<l, es>& source, std::uint32_t fields)
    {
        if (!fields || (fields & ~3U) || source.record.state == lifecycle::retired)
            return std::unexpected(operation_error{"invalid resident publication"});
        resident_entry<l, es> candidate;
        auto old = maps_.find(source.record.id);
        if (old != maps_.end()) candidate = old->second;
        const auto prepare = [&]<storage::schema s>(const auto& input, auto& output) -> operation_status {
            if (!input) return std::unexpected(operation_error{"resident publication field is absent"});
            if (output && output->content_epoch > input->content_epoch)
                return std::unexpected(operation_error{"resident publication is stale"});
            if (output && output->content_epoch == input->content_epoch) return {};
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
        candidate.record.gpu = residency::resident;
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
