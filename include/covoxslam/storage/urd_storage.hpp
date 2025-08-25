#pragma once
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/storage/schema.hpp>
#include <covoxslam/storage/views.hpp>
#include <array>
#include <cstddef>
#include <vector>

#if defined(covoxslam_has_urd)
#include <urd/types/struct_container.hpp>
#include <urd/types/struct_container/soa_offset.hpp>
#endif

namespace covoxslam::storage
{
template<schema s, layout l> struct storage_traits;
#if defined(covoxslam_has_urd)
template<schema s> using urd_pool = urd::v2::struct_container<value<s>, 0,
    urd::v2::soa_layout, urd::v2::heap_storage>;
template<schema s> struct storage_traits<s, layout::pool_soa>
{
    using type = urd_pool<s>;
    static constexpr std::size_t voxel_stride = 4;
     static constexpr std::size_t tile_stride = voxels_per_tile * 4;
    static constexpr std::array<std::size_t, field_count<s>> offsets{};
};
#endif
template<schema s, layout l> using collection_storage = typename storage_traits<s, l>::type;
template<schema s, layout l> inline constexpr std::size_t payload_tile_bytes =
    storage_traits<s, l>::tile_stride * (l == layout::pool_soa ? field_count<s> : 1);
template<schema s, layout l, std::size_t field, class t>
decltype(auto) storage_field(t& storage, std::size_t tile, std::size_t voxel) noexcept
{
    return storage.template get<field>()[tile * voxels_per_tile + voxel];
}
}
