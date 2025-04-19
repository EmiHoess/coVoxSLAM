#pragma once
#include <covoxslam/store/cpu_index.hpp>
#include <algorithm>
#include <array>

namespace covoxslam::store
{
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
}
