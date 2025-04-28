#pragma once
#include <covoxslam/store/handle.hpp>
#include <span>
#include <unordered_map>
#include <vector>

namespace covoxslam::store
{
struct slot_record
{
    block_key key;
    std::uint64_t generation{1};
    bool occupied{true};
};
class cpu_block_index
{
    std::unordered_map<block_key, std::uint32_t, key_hash> buckets_;
    std::vector<slot_record> slots_;
    std::uint64_t submap_{};
    field_kind field_{};
    bool initialized_{};
public:
    operation_status restore(std::uint64_t submap, field_kind field, std::span<const slot_record> records)
    {
        if (initialized_ || (field != field_kind::tsdf && field != field_kind::esdf) ||
            records.size() > std::numeric_limits<std::uint32_t>::max())
            return std::unexpected(operation_error{"invalid restored block index"});
        decltype(buckets_) buckets;
        buckets.reserve(records.size());

        for (std::size_t slot = 0; slot < records.size(); ++slot) {
            const auto& record = records[slot];
            if (!record.generation || record.key.submap != submap ||
                (record.occupied && !buckets.emplace(record.key, std::uint32_t(slot)).second))
                return std::unexpected(operation_error{"restored slot identities conflict"});
        }
        std::vector<slot_record> slots(records.begin(), records.end());
        buckets_ = std::move(buckets); slots_ = std::move(slots);
        submap_ = submap; field_ = field; initialized_ = true;
        return {};
    }
    operation_status initialize(std::uint64_t submap, field_kind field, std::span<const block_key> keys)
    {
        if (initialized_ || (field != field_kind::tsdf && field != field_kind::esdf) ||
            keys.size() > std::numeric_limits<std::uint32_t>::max())
            return std::unexpected(operation_error{"invalid block index initialization"});
        decltype(buckets_) buckets;
        std::vector<slot_record> slots;
        slots.reserve(keys.size());
        buckets.reserve(keys.size());

        for (const auto& key : keys) {
            if (key.submap != submap || !buckets.emplace(key, std::uint32_t(slots.size())).second)
                return std::unexpected(operation_error{"block index keys conflict"});
            slots.push_back({key});
        }
        buckets_ = std::move(buckets);
        slots_ = std::move(slots);
        submap_ = submap;
        field_ = field;
        initialized_ = true;
        return {};
    }
    std::size_t size() const noexcept { return buckets_.size(); }
    bool initialized() const noexcept { return initialized_; }
    std::uint64_t submap() const noexcept { return submap_; }
    field_kind field() const noexcept { return field_; }
    std::size_t slot_count() const noexcept { return slots_.size(); }
    std::span<const slot_record> records() const noexcept { return slots_; }
    operation_result<tile_handle> insert(block_key key, std::size_t maximum_slots)
    {
        if (!initialized_ || key.submap != submap_)
            return std::unexpected(operation_error{"block insertion has invalid submap"});
        const auto existing = buckets_.find(key);
        if (existing != buckets_.end()) return find(key);
        std::size_t slot{};
        while (slot < slots_.size() && slots_[slot].occupied) ++slot;

        if (slot >= maximum_slots || slot >= std::numeric_limits<std::uint32_t>::max())
            return std::unexpected(operation_error{"block pool capacity is exhausted"});
        if (slot == slots_.size()) slots_.reserve(slot + 1);
        buckets_.emplace(key, std::uint32_t(slot));
        if (slot == slots_.size()) slots_.push_back({key});
        else { slots_[slot].key = key; slots_[slot].occupied = true; }
        return tile_handle{submap_, field_, std::uint32_t(slot), slots_[slot].generation};
    }
    operation_status erase(tile_handle handle)
    {
        auto key = resolve(handle);
        if (!key) return std::unexpected(key.error());
        auto generation = next_epoch(slots_[handle.slot].generation);
        if (!generation) return std::unexpected(generation.error());
        buckets_.erase(*key);
        slots_[handle.slot].occupied = false;
        slots_[handle.slot].generation = *generation;
        return {};
    }
    operation_result<tile_handle> find(block_key key) const
    {
        const auto found = buckets_.find(key);
        if (!initialized_ || found == buckets_.end()) return std::unexpected(operation_error{"block key is absent"});
        const auto slot = found->second;
        return tile_handle{submap_, field_, slot, slots_[slot].generation};
    }
    operation_result<block_key> resolve(tile_handle handle) const
    {
        if (!initialized_ || handle.submap != submap_ || handle.field != field_ || handle.slot >= slots_.size())
            return std::unexpected(operation_error{"tile handle belongs to another field"});
        const auto& slot = slots_[handle.slot];
        if (!slot.occupied || handle.reuse_generation != slot.generation)
            return std::unexpected(operation_error{"tile handle is stale"});
        return slot.key;
    }
};
}
