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
    std::size_t slot_count() const noexcept { return slots_.size(); }
    std::span<const slot_record> records() const noexcept { return slots_; }
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
