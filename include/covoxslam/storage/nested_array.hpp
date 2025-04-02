#pragma once
#include <covoxslam/storage/draft.hpp>

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
public:
    nested_draft() = default;
    nested_draft(const nested_draft&) = delete;
    nested_draft& operator=(const nested_draft&) = delete;
    operation_status initialize(std::size_t tiles, nested_order order = nested_order::lane_major)
    {
        if (!groups_.empty() || size_) return std::unexpected(operation_error{"nested collection is already initialized"});
        if (order != nested_order::lane_major && order != nested_order::parent_major)
            return std::unexpected(operation_error{"invalid nested element order"});
        constexpr auto maximum = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
        if (tiles > maximum - lanes + 1 || (tiles + lanes - 1) / lanes > maximum / sizeof(group))
            return std::unexpected(operation_error{"nested collection size overflow"});
        groups_.resize((tiles + lanes - 1) / lanes);
        size_ = tiles;
        order_ = order;
        return {};
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
};
}
#endif
