#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/map_store/authority.hpp>
#include <covoxslam/map_store/cpu_index.hpp>
#include <covoxslam/map_store/residency.hpp>
#include <covoxslam/map_store/snapshot.hpp>
#include <covoxslam/map_store/checkpoint.hpp>
#include <covoxslam/map_store/detail/field_access.hpp>
#include <covoxslam/storage/host_buffer.hpp>
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/storage/record_pack.hpp>
#if defined(covoxslam_has_cuda)
#include <covoxslam/storage/device_buffer.hpp>
#include <covoxslam/map_store/detail/gpu_store.hpp>
#endif
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace covoxslam::store
{
template<storage::schema s, storage::layout l> class field_editor
{
    field_pool<s, l>& field_;
    memory_budget& budget_;
    std::size_t maximum_tiles_;
    std::optional<operation_error> failure_;
public:
    struct cached_tile
    {
        tile_handle handle;
        const void* owner{};
        std::uint64_t generation{};
    };
    using voxel_reference = decltype(std::declval<storage::host_buffer<s, l>&>().view()[0].voxels[0]);
    field_editor(field_pool<s, l>& field, memory_budget& budget, std::size_t maximum_tiles)
        : field_(field), budget_(budget), maximum_tiles_(maximum_tiles) {}
    field_editor(const field_editor&) = delete;
    field_editor& operator=(const field_editor&) = delete;
    std::size_t size() const { return field_.index.size(); }
    std::span<const slot_record> slots() const { return field_.index.records(); }
    operation_result<tile_handle> find(block_key key) const { return field_.index.find(key); }
private:
    operation_result<tile_handle> allocate_impl(block_key key)
    {
        auto present = find(key);
        if (present) return present;
        const auto count = field_.payload.size();
        auto handle = field_.index.insert(key, maximum_tiles_);
        if (!handle) return std::unexpected(handle.error());
        const auto capacity = field_.index.slot_count();
        std::shared_ptr<budget_claim> replacement;
        if (capacity > field_.payload.capacity()) {
            constexpr auto stride = storage::payload_tile_bytes<s, l>;
            if (capacity > std::numeric_limits<std::size_t>::max() / stride)
                return std::unexpected(operation_error{"mapping field growth overflows"});
            auto memory = budget_.reserve({capacity * stride, 0, 0});
            if (!memory) return std::unexpected(memory.error());
            replacement = std::move(*memory);
        }
        auto checked = field_.payload.resize(capacity);
        if (!checked) return std::unexpected(checked.error());
        if (replacement) field_.memory = std::move(replacement);
        if (handle->slot < count) {
            auto view = field_.payload.view();

            for (std::size_t voxel = 0; voxel < 512; ++voxel)
                storage::scatter_record(view[handle->slot].voxels[voxel], storage::value<s>{});
        }
        return handle;
    }
public:
    operation_status status() const
    {
        if (failure_) return std::unexpected(*failure_);
        if (field_.payload.size() != field_.index.slot_count())
            return std::unexpected(operation_error{"mapping payload and index sizes differ"});
        return {};
    }
    operation_result<tile_handle> allocate(block_key key)
    {
        if (failure_) return std::unexpected(*failure_);
        auto result = allocate_impl(key);
        if (!result) failure_ = result.error();
        return result;
    }
    operation_result<cached_tile> cache(block_key key)
    {
        auto handle = allocate(key);
        if (!handle) return std::unexpected(handle.error());
        return cached_tile{*handle, &field_, field_.payload.generation()};
    }
    operation_result<voxel_reference> voxel(tile_handle handle, std::size_t local)
    {
        if (failure_) return std::unexpected(*failure_);
        auto checked = field_.index.resolve(handle);
        if (!checked) { failure_ = checked.error(); return std::unexpected(*failure_); }
        if (local >= 512) {
            failure_ = operation_error{"local voxel index is invalid"};
            return std::unexpected(*failure_);
        }
        return field_.payload.view()[handle.slot].voxels[local];
    }
    operation_result<voxel_reference> voxel(cached_tile cached, std::size_t local)
    {
        if (cached.owner != &field_ || cached.generation != field_.payload.generation()) {
            failure_ = operation_error{"cached mapping view is stale"};
            return std::unexpected(*failure_);
        }
        return voxel(cached.handle, local);
    }
};
template<storage::layout l, storage::schema es = storage::schema::esdf> struct map_entry
{
    submap_record record;
    std::shared_ptr<const field_pool<storage::schema::tsdf, l>> tsdf;
    std::shared_ptr<const field_pool<es, l>> esdf;
};
struct store_statistics
{
    std::size_t submaps{}, retired{}, live_tiles{}, allocated_slots{}, payload_capacity_bytes{}, index_entries{};
};
template<storage::layout l, storage::schema es = storage::schema::esdf> class map_store
{
#include <covoxslam/map_store/detail/cpu_store_state.inl>
#include <covoxslam/map_store/detail/cpu_store_snapshot.inl>
#include <covoxslam/map_store/detail/cpu_store_fields.inl>
};
struct residency_diagnostics
{
    std::size_t active{}, finalized{}, retired{}, cpu_stale_fields{}, gpu_stale_fields{};
    std::size_t pending_reclamation_bytes{}, current_cpu_payload_bytes{};
};
struct pose_publication
{
    std::uint64_t submap{};
    epochs expected;
    pose_record optimized;
};
template<storage::layout l, storage::schema es = storage::schema::esdf> class coordinated_store
{
#include <covoxslam/map_store/detail/coordinator_state.inl>
#include <covoxslam/map_store/detail/coordinator_cpu.inl>
#if defined(covoxslam_has_cuda)
#include <covoxslam/map_store/detail/coordinator_gpu_publish.inl>
#include <covoxslam/map_store/detail/coordinator_gpu_access.inl>
#endif
};
}
