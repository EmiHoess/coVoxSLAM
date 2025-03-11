#pragma once
#include <covoxslam/storage/schema.hpp>
#include <tuple>
#include <utility>
#include <vector>

#if defined(covoxslam_has_urd)
#include <urd/types/struct_container.hpp>
#include <urd/types/struct_container/soa_offset.hpp>
#endif

namespace covoxslam::storage
{
template<schema s, class t> auto scalar_fields(t& record) noexcept
{
    if constexpr (s == schema::tsdf)
        return std::tie(record.distance, record.weight, record.sample_count, record.flags);
    else if constexpr (s == schema::esdf)
        return std::tie(record.signed_distance_m, record.parent_dx, record.parent_dy, record.parent_dz, record.flags);
    else if constexpr (s == schema::blocks_esdf)
        return std::tie(record.distance_m, record.parent_dx, record.parent_dy, record.parent_dz, record.flags);
    else return std::tie(record.squared_distance_voxels, record.parent_x, record.parent_y, record.parent_z, record.flags);
}
template<schema s, layout l> struct storage_traits;
template<schema s> struct storage_traits<s, layout::pooled_aos>
{
    using type = std::vector<aos_tile<s>>;
    static constexpr std::size_t voxel_stride = sizeof(value<s>);
    static constexpr std::size_t tile_stride = sizeof(aos_tile<s>);
    static constexpr auto offsets = [] {
        std::array<std::size_t, field_count<s>> result{};

        for (std::size_t i = 0; i < result.size(); ++i) result[i] = i * 4;
        return result;
    }();
};
#if defined(covoxslam_has_urd)
template<schema s> using urd_tile = urd::v2::struct_container<value<s>, voxels_per_tile,
    urd::v2::soa_layout, urd::v2::stack_storage>;
template<schema s> using urd_pool = urd::v2::struct_container<value<s>, 0,
    urd::v2::soa_layout, urd::v2::heap_storage>;
template<schema s> struct storage_traits<s, layout::tile_aosoa>
{
    using type = std::vector<urd_tile<s>>;
    static_assert(std::is_standard_layout_v<urd_tile<s>>);
    static_assert(sizeof(urd_tile<s>) == sizeof(typename urd_tile<s>::storage_type));
    static constexpr std::size_t voxel_stride = 4;
    static constexpr std::size_t tile_stride = sizeof(urd_tile<s>);
    static constexpr auto offsets = urd::v2::detail::storage_offsets<typename urd_tile<s>::storage_type>::value;
};
template<schema s> struct storage_traits<s, layout::pool_soa>
{
    using type = urd_pool<s>;
    static constexpr std::size_t voxel_stride = 4;
     static constexpr std::size_t tile_stride = voxels_per_tile * 4;
    static constexpr std::array<std::size_t, field_count<s>> offsets{};
};
#endif
template<schema s, layout l> using collection_storage = typename storage_traits<s, l>::type;
template<schema s, layout l, std::size_t field, class t>
decltype(auto) storage_field(t& storage, std::size_t tile, std::size_t voxel) noexcept
{
    if constexpr (l == layout::pooled_aos)
        return std::get<field>(scalar_fields<s>(storage[tile].voxels[voxel]));
    else if constexpr (l == layout::tile_aosoa)
        return storage[tile].template get<field>()[voxel];
    else return storage.template get<field>()[tile * voxels_per_tile + voxel];
}
}
