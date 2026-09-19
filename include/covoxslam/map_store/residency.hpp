#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/map_store/cpu_index.hpp>
#if defined(covoxslam_has_cuda)
#include <covoxslam/storage/device_buffer.hpp>
#include <covoxslam/storage/host_buffer.hpp>
#include <covoxslam/storage/layout.hpp>
#endif
#include <array>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

namespace covoxslam::store
{
enum class memory_space : std::size_t { host, pinned, device };
using memory_bytes = std::array<std::size_t, 3>;
struct residency_policy
{
    memory_bytes limits{1024U * 1024U * 1024U, 128U * 1024U * 1024U, 1024U * 1024U * 1024U};
    std::size_t maximum_tiles{1000000};
};
struct budget_snapshot
{
    memory_bytes limit{}, used{}, peak{}, pending{};
};
struct budget_state
{
    std::mutex mutex;
    budget_snapshot counters;
};
class budget_claim
{
    std::shared_ptr<budget_state> state_;
    memory_bytes bytes_{};
    bool pending_{};
    friend class memory_budget;
public:
    budget_claim() = default;
    budget_claim(const budget_claim&) = delete;
    budget_claim& operator=(const budget_claim&) = delete;
    ~budget_claim()
    {
        if (!state_) return;
        std::lock_guard lock(state_->mutex);

        for (unsigned i = 0; i < 3; ++i) {
            state_->counters.used[i] -= bytes_[i];
            if (pending_) state_->counters.pending[i] -= bytes_[i];
        }
    }
    void retire()
    {
        if (!state_) return;
        std::lock_guard lock(state_->mutex);
        if (pending_) return;

        for (unsigned i = 0; i < 3; ++i) state_->counters.pending[i] += bytes_[i];
        pending_ = true;
    }
    const memory_bytes& bytes() const noexcept { return bytes_; }
};
class memory_budget
{
    std::shared_ptr<budget_state> state_{std::make_shared<budget_state>()};
public:
    explicit memory_budget(memory_bytes limits)
    {
        state_->counters.limit = limits;
    }
    bool owns(const std::shared_ptr<budget_claim>& claim) const noexcept
    {
        return claim && claim->state_ == state_;
    }
    operation_result<std::shared_ptr<budget_claim>> reserve(memory_bytes bytes)
    {
        auto claim = std::make_shared<budget_claim>();
        std::lock_guard lock(state_->mutex);
        auto& current = state_->counters;

        for (unsigned i = 0; i < 3; ++i)
            if (bytes[i] > current.limit[i] - current.used[i])
                return std::unexpected(operation_error{"map store memory budget exhausted"});

        for (unsigned i = 0; i < 3; ++i) {
            current.used[i] += bytes[i];
            current.peak[i] = std::max(current.peak[i], current.used[i]);
        }
        claim->bytes_ = bytes;
        claim->state_ = state_;
        return claim;
    }
    operation_status configure(memory_bytes limits)
    {
        std::lock_guard lock(state_->mutex);

        for (unsigned i = 0; i < 3; ++i)
            if (limits[i] < state_->counters.used[i]) return std::unexpected(operation_error{"new budget is below retained allocations"});
        state_->counters.limit = limits;
        return {};
    }
    budget_snapshot snapshot() const
    {
        std::lock_guard lock(state_->mutex);
        return state_->counters;
    }
};
inline operation_result<memory_bytes> add_memory(memory_bytes first, memory_bytes second)
{
    for (unsigned i = 0; i < 3; ++i) {
        if (second[i] > std::numeric_limits<std::size_t>::max() - first[i])
            return std::unexpected(operation_error{"memory accounting overflows"});
        first[i] += second[i];
    }
    return first;
}
struct dirty_slot
{
    std::uint64_t generation{};
    std::array<std::uint64_t, 5> versions{};
    std::uint32_t fields{};
};
struct dirty_record { std::uint32_t slot{}; dirty_slot state; };
struct dirty_batch
{
    std::uint64_t submap{};
    field_kind field{};
    std::vector<dirty_record> records;
};
struct dirty_range { std::size_t first_slot{}, slots{}; std::uint32_t fields{}; };
template<storage::schema s> class dirty_tracker
{
    static constexpr auto mask = (1U << storage::field_count<s>) - 1;
    std::uint64_t submap_{};
    std::vector<dirty_slot> slots_;
    static constexpr auto field = s == storage::schema::tsdf ? field_kind::tsdf : field_kind::esdf;
public:
    explicit dirty_tracker(std::uint64_t submap) : submap_(submap) {}
    operation_status reserve(std::size_t slots)
    {
        if (slots > std::numeric_limits<std::uint32_t>::max() || slots > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(dirty_slot))
            return std::unexpected(operation_error{"dirty slot capacity overflows"});
        if (slots > slots_.size()) slots_.resize(slots);
        return {};
    }
    operation_status mark(tile_handle handle, std::uint32_t fields, std::uint64_t content_epoch)
    {
        if (handle.submap != submap_ || handle.field != field || !handle.reuse_generation ||
            !fields || (fields & ~mask) || !content_epoch || handle.slot >= slots_.size())
            return std::unexpected(operation_error{"dirty update has invalid identity fields or version"});
        auto& target = slots_[handle.slot];
        if (handle.reuse_generation < target.generation)
            return std::unexpected(operation_error{"dirty handle is stale"});
        auto next = target;
        if (handle.reuse_generation != target.generation) {
            if (fields != mask) return std::unexpected(operation_error{"reused slot requires all fields"});
            if (content_epoch <= *std::max_element(target.versions.begin(), target.versions.end()))
                return std::unexpected(operation_error{"reused slot content epoch did not advance"});
            next = {}; next.generation = handle.reuse_generation;
        }

        for (unsigned i = 0; i < storage::field_count<s>; ++i)
            if (fields & (1U << i)) {
                if (content_epoch <= next.versions[i]) return std::unexpected(operation_error{"dirty content version did not advance"});
                next.versions[i] = content_epoch;
            }
        next.fields |= fields;
        target = next;
        return {};
    }
    dirty_batch capture(std::uint32_t fields = mask) const
    {
        dirty_batch batch{submap_, field};

        for (std::size_t slot = 0; slot < slots_.size(); ++slot) {
            auto state = slots_[slot];
            state.fields &= fields & mask;
            if (state.fields) batch.records.push_back({std::uint32_t(slot), state});
        }
        return batch;
    }
    operation_status acknowledge(const dirty_batch& batch, operation_status completed)
    {
        if (!completed) return completed;
        if (batch.submap != submap_ || batch.field != field)
            return std::unexpected(operation_error{"dirty acknowledgement belongs to another field"});

        std::uint64_t previous{};
        bool first = true;

        for (const auto& record : batch.records) {
            if (record.slot >= slots_.size() || !record.state.generation || !record.state.fields ||
                (record.state.fields & ~mask) || (!first && record.slot <= previous))
                return std::unexpected(operation_error{"dirty acknowledgement is out of bounds"});
            first = false; previous = record.slot;

            for (unsigned i = 0; i < storage::field_count<s>; ++i)
                if ((record.state.fields & (1U << i)) && !record.state.versions[i])
                    return std::unexpected(operation_error{"dirty acknowledgement has no source version"});
        }

        for (const auto& record : batch.records) {
            auto& current = slots_[record.slot];
            if (current.generation != record.state.generation) continue;

            for (unsigned i = 0; i < storage::field_count<s>; ++i)
                if ((record.state.fields & (1U << i)) && current.versions[i] == record.state.versions[i])
                    current.fields &= ~(1U << i);
        }
        return {};
    }
};
inline operation_result<std::vector<dirty_range>> dirty_ranges(const dirty_batch& batch, std::size_t slots, std::uint32_t allowed_fields)
{
    std::vector<dirty_range> result;
    std::size_t previous{};
    bool first = true;

    for (const auto& record : batch.records) {
        if (record.slot >= slots || (!first && record.slot <= previous) || !record.state.generation ||
            !record.state.fields || (record.state.fields & ~allowed_fields))
            return std::unexpected(operation_error{"dirty batch is not a valid ordered slot set"});
        first = false; previous = record.slot;
        if (!result.empty() && result.back().fields == record.state.fields && result.back().first_slot + result.back().slots == record.slot)
            ++result.back().slots;
        else result.push_back({record.slot, 1, record.state.fields});
    }
    return result;
}
#if defined(covoxslam_has_cuda)
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
operation_result<dirty_measurement> upload_dirty(const storage::host_buffer<s, h>& source, storage::device_buffer<s, d>& destination,
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
        auto checked = destination.upload_range(source, range.first_slot * tile_voxels,
            range.first_slot * tile_voxels, range.slots * tile_voxels,
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
operation_result<dirty_measurement> download_dirty(const storage::device_buffer<s, d>& source, storage::host_buffer<s, h>& destination,
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
        auto checked = source.download_range(destination, range.first_slot * tile_voxels,
            range.first_slot * tile_voxels, range.slots * tile_voxels,
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
#endif
}
