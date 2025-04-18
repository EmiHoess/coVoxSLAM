#pragma once
#include <covoxslam/store/dirty.hpp>
#include <covoxslam/storage/cuda_draft.hpp>

namespace covoxslam::store
{
struct dirty_measurement
{
    std::size_t logical_bytes{}, transferred_bytes{}, calls{}, waits{}, ranges{};
};
template<storage::schema s> operation_result<std::vector<dirty_range>> validate_dirty(
    const dirty_batch& batch, const cpu_block_index& index, std::uint64_t target_epoch)
{
    constexpr auto fields = (1U << storage::field_count<s>) - 1;
    if (!index.initialized() || batch.submap != index.submap() || batch.field != index.field() || !target_epoch)
        return std::unexpected(operation_error{"dirty transfer identity differs"});
    auto ranges = dirty_ranges(batch, index.slot_count(), fields);
    if (!ranges) return std::unexpected(ranges.error());

    for (const auto& record : batch.records) {
        if (index.records()[record.slot].generation != record.state.generation)
            return std::unexpected(operation_error{"dirty transfer slot was recycled"});

        for (unsigned i = 0; i < storage::field_count<s>; ++i)
            if ((record.state.fields & (1U << i)) && (!record.state.versions[i] || record.state.versions[i] > target_epoch))
                return std::unexpected(operation_error{"dirty transfer content version differs"});
    }
    return ranges;
}
template<storage::schema s, storage::layout h, storage::layout d>
operation_result<dirty_measurement> upload_dirty(const storage::draft<s, h>& source, storage::cuda_draft<s, d>& destination,
    const cpu_block_index& index, std::uint64_t epoch, const dirty_batch& batch)
{
    auto ranges = validate_dirty<s>(batch, index, epoch);
    if (!ranges) return std::unexpected(ranges.error());
    auto descriptor = destination.descriptor();
    if (!descriptor) return std::unexpected(descriptor.error());
    if (source.size() != index.slot_count() || descriptor->size_tiles != source.size())
        return std::unexpected(operation_error{"dirty transfer payload extent differs"});
    dirty_measurement measured;

    for (const auto& range : *ranges) {
        storage::transfer_measurement transfer;
        auto checked = destination.upload_range(source, range.first_slot * 512, range.first_slot * 512, range.slots * 512,
            range.fields, storage::transfer_method::bulk_same_layout, transfer);
        if (!checked) return std::unexpected(checked.error());
        measured.logical_bytes += transfer.useful_bytes;
        measured.transferred_bytes += transfer.useful_bytes;
        measured.calls += transfer.calls;
        measured.waits += transfer.waits;
        ++measured.ranges;
    }
    return measured;
}
template<storage::schema s, storage::layout d, storage::layout h>
operation_result<dirty_measurement> download_dirty(const storage::cuda_draft<s, d>& source, storage::draft<s, h>& destination,
    const cpu_block_index& index, std::uint64_t epoch, const dirty_batch& batch)
{
    auto ranges = validate_dirty<s>(batch, index, epoch);
    if (!ranges) return std::unexpected(ranges.error());
    auto descriptor = source.descriptor();
    if (!descriptor) return std::unexpected(descriptor.error());
    if (destination.size() != index.slot_count() || descriptor->size_tiles != destination.size())
        return std::unexpected(operation_error{"dirty transfer payload extent differs"});
    dirty_measurement measured;

    for (const auto& range : *ranges) {
        storage::transfer_measurement transfer;
        auto checked = source.download_range(destination, range.first_slot * 512, range.first_slot * 512, range.slots * 512,
            range.fields, storage::transfer_method::bulk_same_layout, transfer);
        if (!checked) return std::unexpected(checked.error());
        measured.logical_bytes += transfer.useful_bytes;
        measured.transferred_bytes += transfer.useful_bytes;
        measured.calls += transfer.calls;
        measured.waits += transfer.waits;
        ++measured.ranges;
    }
    return measured;
}
}
