#pragma once
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/storage/urd_storage.hpp>
#include <covoxslam/storage/views.hpp>
#include <covoxslam/core/result.hpp>
#include <algorithm>
#include <limits>
#include <memory>
#include <span>

namespace covoxslam::storage
{
template<schema s, layout l> class host_buffer
{
    using storage_type = collection_storage<s, l>;
    using traits = storage_traits<s, l>;
    std::unique_ptr<storage_type> storage_;
    std::size_t size_{}, capacity_{};
     std::uint64_t generation_{1};
    std::shared_ptr<std::uint64_t> epoch_{std::make_shared<std::uint64_t>(generation_)};
    void advance_generation() noexcept { *epoch_ = ++generation_; }
    template<std::size_t... i> static void initialize_fields(storage_type& storage,
        std::size_t count, std::index_sequence<i...>)
    {
        (storage.template get<i>().resize(count * voxels_per_tile), ...);
    }
    static operation_status check_capacity(std::size_t count)
    {
        constexpr auto bound = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
        constexpr auto bytes = std::max(sizeof(value<s>) * voxels_per_tile, traits::tile_stride);
        if (count > bound / bytes)
            return std::unexpected(operation_error{"collection capacity exceeds addressable storage"});
        return {};
    }
    template<std::size_t... i> void copy_live(storage_type& destination, std::index_sequence<i...>) const
    {
        for (std::size_t tile = 0; tile < size_; ++tile)
            for (std::size_t voxel = 0; voxel < voxels_per_tile; ++voxel)
                ((storage_field<s, l, i>(destination, tile, voxel) =
                    storage_field<s, l, i>(std::as_const(*storage_), tile, voxel)), ...);
    }
    template<std::size_t... i> void zero_range(std::size_t first, std::size_t last, std::index_sequence<i...>) noexcept
    {
        for (std::size_t tile = first; tile < last; ++tile)
            for (std::size_t voxel = 0; voxel < voxels_per_tile; ++voxel)
                ((storage_field<s, l, i>(*storage_, tile, voxel) = field_type<s, i>{}), ...);
    }
public:
    template<bool constant> class host_accessor
    {
        using owner_type = std::conditional_t<constant, const host_buffer, host_buffer>;
        owner_type* owner_{};
        std::weak_ptr<std::uint64_t> epoch_;
        std::uint64_t generation_{};
        std::size_t size_{}, capacity_{};
        friend class host_buffer;
        explicit host_accessor(owner_type& owner) noexcept : owner_(&owner), epoch_(owner.epoch_),
            generation_(owner.generation()), size_(owner.size()), capacity_(owner.capacity()) {}
    public:
        static constexpr bool read_only = constant;
        static constexpr layout storage_layout = l;
        const void* owner_identity() const noexcept { return owner_; }
        template<std::size_t field> auto field_segment(std::size_t first) const noexcept
        {
            using scalar_type = std::conditional_t<constant, const field_type<s, field>, field_type<s, field>>;
            const auto count = size_ * voxels_per_tile - first;
            return std::span<scalar_type>(&owner_->template scalar<field>(first / voxels_per_tile, first % voxels_per_tile), count);
        }
        std::size_t size() const noexcept { return size_; }
        std::size_t capacity() const noexcept { return capacity_; }
        bool valid() const noexcept
        {
            const auto epoch = epoch_.lock();
            return epoch && *epoch == generation_;
        }
        template<std::size_t field> decltype(auto) scalar(std::size_t tile, std::size_t voxel) const noexcept
        {
            return owner_->template scalar<field>(tile, voxel);
        }
    };
    auto view() noexcept { return map_view<s, host_accessor<false>>{host_accessor<false>{*this}}; }
    auto view() const noexcept { return map_view<s, host_accessor<true>>{host_accessor<true>{*this}}; }
    host_buffer() = default;
    host_buffer(const host_buffer&) = delete;
    host_buffer& operator=(const host_buffer&) = delete;
    host_buffer(host_buffer&&) = delete;
    host_buffer& operator=(host_buffer&&) = delete;
    operation_status initialize(std::size_t count)
    {
        if (storage_) return std::unexpected(operation_error{"collection is already initialized"});
        auto valid = check_capacity(count);
        if (!valid) return valid;
        auto replacement = std::make_unique<storage_type>();
        initialize_fields(*replacement, count, std::make_index_sequence<field_count<s>>{});
        storage_ = std::move(replacement);
        size_ = capacity_ = count;
        advance_generation();
        return {};
    }
    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return capacity_; }
    std::uint64_t generation() const noexcept { return generation_; }
    operation_status reserve(std::size_t count)
    {
        if (count <= capacity_) return {};
        auto valid = check_capacity(count);
        if (!valid) return valid;

        if (generation_ == std::numeric_limits<std::uint64_t>::max())
            return std::unexpected(operation_error{"collection generation exhausted"});
        auto replacement = std::make_unique<storage_type>();
        initialize_fields(*replacement, count, std::make_index_sequence<field_count<s>>{});
        copy_live(*replacement, std::make_index_sequence<field_count<s>>{});
        storage_ = std::move(replacement);
        capacity_ = count;
        advance_generation();
        return {};
    }
    operation_status resize(std::size_t count)
    {
        if (count == size_) return {};
        const auto increments = std::uint64_t(1 + (count > capacity_));
        if (generation_ > std::numeric_limits<std::uint64_t>::max() - increments)
            return std::unexpected(operation_error{"collection generation exhausted"});
        auto allocated = reserve(count);
        if (!allocated) return allocated;

        if (count > size_) zero_range(size_, count, std::make_index_sequence<field_count<s>>{});
        size_ = count;
         advance_generation();
        return {};
    }
    operation_status clear() { return resize(0); }
    template<std::size_t field> decltype(auto) scalar(std::size_t tile, std::size_t voxel) noexcept
    {
        return storage_field<s, l, field>(*storage_, tile, voxel);
    }
    template<std::size_t field> decltype(auto) scalar(std::size_t tile, std::size_t voxel) const noexcept
    {
        return storage_field<s, l, field>(std::as_const(*storage_), tile, voxel);
    }
};
}
