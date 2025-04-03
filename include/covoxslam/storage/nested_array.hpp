#pragma once
#include <covoxslam/storage/draft.hpp>
#include <covoxslam/storage/nested_device_view.hpp>
#include <cstring>

#if defined(covoxslam_has_urd)
#include <urd/types/array.hpp>

namespace covoxslam::storage
{
enum class nested_order { parent_major, lane_major };
enum class nested_array_policy { aos, soa, standard };
template<schema s> struct nested_value;
template<> struct nested_value<schema::tsdf>
{
    using _memory_layout = urd::layout::soa;
    float distance{}, weight{};
    std::uint32_t sample_count{}, flags{};
};
template<> struct nested_value<schema::esdf> {
    using _memory_layout = urd::layout::soa;
    float signed_distance_m{};
    std::int32_t parent_dx{}, parent_dy{}, parent_dz{};
    std::uint32_t flags{};
};
template<> struct nested_value<schema::blocks_esdf>
{
    using _memory_layout = urd::layout::soa;
    float distance_m{};
    std::int32_t parent_dx{}, parent_dy{}, parent_dz{};
    std::uint32_t flags{};
};
template<> struct nested_value<schema::projective_esdf>
{
    using _memory_layout = urd::layout::soa;
    float squared_distance_voxels{};
    std::int32_t parent_x{}, parent_y{}, parent_z{};
    std::uint32_t flags{};
};
template<schema s, nested_array_policy policy, bool tagged> struct nested_tile
{
    using element = std::conditional_t<tagged, nested_value<s>, value<s>>;
    using array_type = std::conditional_t<policy == nested_array_policy::standard, std::array<element, 512>,
        urd::array<element, 512, std::conditional_t<policy == nested_array_policy::aos, urd::layout::aos, urd::layout::soa>>>;
    array_type voxels;
};
template<schema s, std::size_t lanes, nested_array_policy policy, bool tagged> struct nested_traits
{
    static_assert(lanes == 8 || lanes == 16 || lanes == 32);
    using group_type = urd::v2::struct_container<nested_tile<s, policy, tagged>, lanes,
        urd::v2::soa_layout, urd::v2::stack_storage>;
    static constexpr bool split_members = tagged && policy != nested_array_policy::aos;
    static constexpr std::size_t member_stride = split_members ? 4 : sizeof(value<s>);
    static_assert(sizeof(typename nested_tile<s, policy, tagged>::element) == sizeof(value<s>));

    template<std::size_t field, class group> static decltype(auto) scalar(group& storage, std::size_t lane,
        std::size_t voxel, nested_order order)
    {
        static_assert(field < field_count<s>);
        const auto index = order == nested_order::lane_major ? voxel * lanes + lane : lane * 512 + voxel;
        auto& array = storage.template get<0>();
        if constexpr (split_members) return urd::v2::get<field>(array)[index];
        else return std::get<field>(scalar_fields<s>(array[index]));
    }
};
template<schema s, std::size_t lanes, nested_array_policy policy, bool tagged>
class nested_draft
{
    using traits = nested_traits<s, lanes, policy, tagged>;
    using group = typename traits::group_type;
    std::vector<group> groups_;
    std::size_t size_{};
    nested_order order_{nested_order::lane_major};
    std::shared_ptr<std::uint64_t> generation_{std::make_shared<std::uint64_t>(1)};
    bool initialized_{};
    operation_status replace(std::size_t tiles, std::size_t size, nested_order order)
    {
        constexpr auto maximum = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
        if (tiles > maximum - lanes + 1 || (tiles + lanes - 1) / lanes > maximum / sizeof(group) || size > tiles ||
            *generation_ == std::numeric_limits<std::uint64_t>::max())
            return std::unexpected(operation_error{"nested replacement extent or generation is invalid"});
        std::vector<group> replacement((tiles + lanes - 1) / lanes);

        for (std::size_t tile = 0; tile < std::min(size_, size); ++tile)
            for (std::size_t voxel = 0; voxel < 512; ++voxel) {
                [&]<std::size_t... field>(std::index_sequence<field...>) {
                    (std::memcpy(&traits::template scalar<field>(replacement[tile / lanes], tile % lanes, voxel, order),
                        &scalar<field>(tile, voxel), 4), ...);
                }(std::make_index_sequence<field_count<s>>{});
            }
        groups_.swap(replacement);
        size_ = size;
        order_ = order;
        ++*generation_;
        return {};
    }
public:
    nested_draft() = default;
    nested_draft(const nested_draft&) = delete;
    nested_draft& operator=(const nested_draft&) = delete;
    operation_status initialize(std::size_t tiles, nested_order order = nested_order::lane_major)
    {
        if (initialized_) return std::unexpected(operation_error{"nested collection is already initialized"});
        if (order != nested_order::lane_major && order != nested_order::parent_major)
            return std::unexpected(operation_error{"invalid nested element order"});
        constexpr auto maximum = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
        if (tiles > maximum - lanes + 1 || (tiles + lanes - 1) / lanes > maximum / sizeof(group))
            return std::unexpected(operation_error{"nested collection size overflow"});
        groups_.resize((tiles + lanes - 1) / lanes);
        size_ = tiles;
        order_ = order;
        initialized_ = true;
        return {};
    }
    operation_status reserve(std::size_t tiles)
    {
        if (!initialized_) return std::unexpected(operation_error{"nested collection is uninitialized"});
        if (tiles <= capacity()) return {};
        return replace(tiles, size_, order_);
    }
    operation_status resize(std::size_t tiles)
    {
        if (!initialized_) return std::unexpected(operation_error{"nested collection is uninitialized"});
        if (tiles == size_) return {};
        if (tiles > capacity()) return replace(tiles, tiles, order_);
        if (*generation_ == std::numeric_limits<std::uint64_t>::max())
            return std::unexpected(operation_error{"nested generation exhausted"});

        for (std::size_t tile = tiles; tile < size_; ++tile)
            for (std::size_t voxel = 0; voxel < 512; ++voxel) {
                [&]<std::size_t... field>(std::index_sequence<field...>) {
                    ((scalar<field>(tile, voxel) = field_type<s, field>{}), ...);
                }(std::make_index_sequence<field_count<s>>{});
            }
        size_ = tiles;
        ++*generation_;
        return {};
    }
    operation_status reorder(nested_order order)
    {
        if (!initialized_ || (order != nested_order::parent_major && order != nested_order::lane_major))
            return std::unexpected(operation_error{"invalid nested reorder"});
        if (order == order_) return {};
        return replace(capacity(), size_, order);
    }
    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return groups_.size() * lanes; }
    std::size_t allocated_bytes() const noexcept { return groups_.size() * sizeof(group); }
    template<std::size_t field> auto& scalar(std::size_t tile, std::size_t voxel) noexcept
    {
        assert(tile < size_ && voxel < 512);
        return traits::template scalar<field>(groups_[tile / lanes], tile % lanes, voxel, order_);
    }
    template<std::size_t field> const auto& scalar(std::size_t tile, std::size_t voxel) const noexcept
    {
        assert(tile < size_ && voxel < 512);
        return traits::template scalar<field>(groups_[tile / lanes], tile % lanes, voxel, order_);
    }
    template<bool constant> struct host_accessor
    {
        using owner_type = std::conditional_t<constant, const nested_draft, nested_draft>;
        owner_type* owner{};
        std::weak_ptr<std::uint64_t> epoch;
        std::uint64_t captured{};
        std::size_t count{}, reserved{};
        static constexpr bool read_only = constant;
        std::size_t size() const noexcept { return count; }
        std::size_t capacity() const noexcept { return reserved; }
        const void* owner_identity() const noexcept { return owner; }
        bool valid() const noexcept { auto current = epoch.lock(); return current && *current == captured && owner->initialized_; }
        template<std::size_t field> decltype(auto) scalar(std::size_t tile, std::size_t voxel) const noexcept
        {
            assert(valid());
            return owner->template scalar<field>(tile, voxel);
        }
    };
    auto view() noexcept
    {
        return map_view<s, host_accessor<false>>{host_accessor<false>{this, generation_, *generation_, size_, capacity()}};
    }
    auto view() const noexcept
    {
        return map_view<s, host_accessor<true>>{host_accessor<true>{this, generation_, *generation_, size_, capacity()}};
    }
    nested_descriptor descriptor() const noexcept
    {
        nested_descriptor result;
        result.size_tiles = size_;
        result.capacity_tiles = capacity();
        result.lanes = lanes;
        result.group_stride = sizeof(group);
        result.member_stride = traits::member_stride;
        result.tile_stride = traits::member_stride * (order_ == nested_order::lane_major ? 1 : 512);
        result.voxel_stride = traits::member_stride * (order_ == nested_order::lane_major ? lanes : 1);
        result.generation = *generation_;
        result.record_schema = s;

        if (capacity()) [&]<std::size_t... field>(std::index_sequence<field...>) {
            ((result.fields[field] = reinterpret_cast<std::byte*>(const_cast<field_type<s, field>*>(
                &traits::template scalar<field>(groups_[0], 0, 0, order_)))), ...);
        }(std::make_index_sequence<field_count<s>>{});
        return result;
    }
};
}
#endif
