#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/map_store/authority.hpp>
#include <covoxslam/map_store/cpu_index.hpp>
#include <covoxslam/map_store/residency.hpp>
#include <covoxslam/storage/host_buffer.hpp>
#include <atomic>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <span>
#include <unordered_set>
#include <vector>

namespace covoxslam::store
{
template<storage::schema s, storage::layout l> struct field_pool
{
    std::shared_ptr<budget_claim> memory;
    storage::host_buffer<s, l> payload;
    cpu_block_index index;
    std::uint64_t content_epoch{};
};
inline bool finite_pose(const pose_record& pose) noexcept
{
    return std::isfinite(pose.x) && std::isfinite(pose.y) && std::isfinite(pose.z) && std::isfinite(pose.yaw);
}
template<storage::layout l, storage::schema es = storage::schema::esdf> struct map_snapshot
{
    submap_record record;
    std::shared_ptr<const field_pool<storage::schema::tsdf, l>> tsdf;
    std::shared_ptr<const field_pool<es, l>> esdf;
};
inline operation_result<std::uint64_t> allocate_snapshot_id()
{
    static std::atomic<std::uint64_t> next{};
    auto previous = next.load(std::memory_order_relaxed);
    do {
        if (previous == std::numeric_limits<std::uint64_t>::max())
            return std::unexpected(operation_error{"snapshot identity exhausted"});
    } while (!next.compare_exchange_weak(previous, previous + 1, std::memory_order_relaxed));
    return previous + 1;
}
template<storage::layout l, storage::schema es = storage::schema::esdf> class immutable_snapshot
{
    std::uint64_t id_{};
    std::size_t bytes_{};
    std::vector<map_snapshot<l, es>> maps_;
public:
    static operation_result<immutable_snapshot> create(std::vector<map_snapshot<l, es>> maps, std::size_t byte_budget)
    {
        immutable_snapshot result;
        std::unordered_set<std::uint64_t> ids;

        for (const auto& map : maps) {
            if (!ids.insert(map.record.id).second || map.record.state == lifecycle::retired ||
                !finite_pose(map.record.initial_pose) || !finite_pose(map.record.optimized_pose) || (!map.tsdf && !map.esdf))
                return std::unexpected(operation_error{"snapshot map identity or pose is invalid"});
            auto admit = [&]<storage::schema s>(const std::shared_ptr<const field_pool<s, l>>& field) -> operation_status {
                if (!field) return {};
                const auto epoch = s == storage::schema::tsdf ? map.record.versions.tsdf : map.record.versions.esdf;
                if (field->index.submap() != map.record.id || !epoch || field->content_epoch != epoch ||
                    field->index.slot_count() != field->payload.size())
                    return std::unexpected(operation_error{"snapshot field is inconsistent"});
                constexpr auto bytes_per_tile = storage::payload_tile_bytes<s, l>;
                if (field->payload.capacity() > (byte_budget - result.bytes_) / bytes_per_tile)
                    return std::unexpected(operation_error{"snapshot payload budget exceeded"});
                result.bytes_ += field->payload.capacity() * bytes_per_tile;
                return {};
            };
            auto checked = admit(map.tsdf);
            if (!checked) return std::unexpected(checked.error());
            checked = admit(map.esdf);
            if (!checked) return std::unexpected(checked.error());
            if (map.esdf && (!map.record.versions.tsdf || map.record.versions.esdf_source != map.record.versions.tsdf))
                return std::unexpected(operation_error{"snapshot ESDF source is stale"});
        }
        auto id = allocate_snapshot_id();
        if (!id) return std::unexpected(id.error());
        result.id_ = *id;
        result.maps_ = std::move(maps);
        return result;
    }
    std::uint64_t id() const noexcept { return id_; }
    std::size_t payload_bytes() const noexcept { return bytes_; }
    std::span<const map_snapshot<l, es>> maps() const noexcept { return maps_; }
    void cancel() noexcept { maps_.clear(); bytes_ = 0; id_ = 0; }
};
struct retired_allocation
{
    std::shared_ptr<const void> owner;
    std::size_t bytes{};
};
class retirement_queue
{
    std::vector<retired_allocation> entries_;
    std::size_t bytes_{};
public:
    operation_status retain(std::shared_ptr<const void> owner, std::size_t bytes)
    {
        const std::array<retired_allocation, 1> items{{{std::move(owner), bytes}}};
        return retain_many(items);
    }
    operation_status retain_many(std::span<const retired_allocation> items)
    {
        if (items.size() > entries_.max_size() - entries_.size())
            return std::unexpected(operation_error{"retirement queue capacity exceeded"});
        auto bytes = bytes_;

        for (std::size_t i = 0; i < items.size(); ++i) {
            const auto& owner = items[i].owner;
            if (!owner || items[i].bytes > std::numeric_limits<std::size_t>::max() - bytes)
                return std::unexpected(operation_error{"retirement accounting overflows"});

            for (const auto& entry : entries_)
                if (!owner.owner_before(entry.owner) && !entry.owner.owner_before(owner))
                    return std::unexpected(operation_error{"allocation is already retired"});

            for (std::size_t j = 0; j < i; ++j)
                if (!owner.owner_before(items[j].owner) && !items[j].owner.owner_before(owner))
                    return std::unexpected(operation_error{"retirement batch repeats an allocation"});
            bytes += items[i].bytes;
        }
        entries_.reserve(entries_.size() + items.size());
        entries_.insert(entries_.end(), items.begin(), items.end());
        bytes_ = bytes;
        return {};
    }
    std::size_t collect()
    {
        std::size_t released{};
        auto current = entries_.begin();
        while (current != entries_.end()) {
            if (current->owner.use_count() == 1) {
                released += current->bytes;
                bytes_ -= current->bytes;
                current = entries_.erase(current);
            }
            else ++current;
        }
        return released;
    }
    std::size_t pending_bytes() const noexcept { return bytes_; }
    std::size_t pending_allocations() const noexcept { return entries_.size(); }
};
}
