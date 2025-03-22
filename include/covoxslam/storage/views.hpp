#pragma once
#include <covoxslam/storage/schema.hpp>
#include <cassert>
#include <span>

#if defined(__CUDACC__)
#define covoxslam_storage_hd __host__ __device__
#else
#define covoxslam_storage_hd
#endif

namespace covoxslam::storage
{
template<bool read_only, class t> using scalar_reference = std::conditional_t<read_only, const t&, t&>;
template<schema s, bool read_only> struct voxel_reference;
template<bool read_only> struct voxel_reference<schema::tsdf, read_only>
{
    scalar_reference<read_only, float> distance, weight;
    scalar_reference<read_only, std::uint32_t> sample_count, flags;
};
template<bool read_only> struct voxel_reference<schema::esdf, read_only> {
    scalar_reference<read_only, float> signed_distance_m;
    scalar_reference<read_only, std::int32_t> parent_dx, parent_dy, parent_dz;
    scalar_reference<read_only, std::uint32_t> flags;
};
template<bool read_only> struct voxel_reference<schema::blocks_esdf, read_only>
{
    scalar_reference<read_only, float> distance_m;
    scalar_reference<read_only, std::int32_t> parent_dx, parent_dy, parent_dz;
    scalar_reference<read_only, std::uint32_t> flags;
};
template<bool read_only> struct voxel_reference<schema::projective_esdf, read_only>
{
    scalar_reference<read_only, float> squared_distance_voxels;
    scalar_reference<read_only, std::int32_t> parent_x, parent_y, parent_z;
     scalar_reference<read_only, std::uint32_t> flags;
};
template<schema s, class accessor>
covoxslam_storage_hd auto bind_voxel(const accessor& access, std::size_t tile, std::size_t voxel) noexcept
{
    if constexpr (s == schema::tsdf)
        return voxel_reference<s, accessor::read_only>{access.template scalar<0>(tile, voxel),
            access.template scalar<1>(tile, voxel), access.template scalar<2>(tile, voxel),
            access.template scalar<3>(tile, voxel)};
    else
        return voxel_reference<s, accessor::read_only>{access.template scalar<0>(tile, voxel),
            access.template scalar<1>(tile, voxel), access.template scalar<2>(tile, voxel),
            access.template scalar<3>(tile, voxel), access.template scalar<4>(tile, voxel)};
}
template<schema s, class accessor> struct voxel_array_view
{
    accessor access;
    std::size_t tile{};
    covoxslam_storage_hd constexpr std::size_t size() const noexcept { return voxels_per_tile; }
    covoxslam_storage_hd auto operator[](std::size_t voxel) const noexcept
    {
        return bind_voxel<s>(access, tile, voxel);
    }
};
template<schema s, class accessor> struct tile_view { voxel_array_view<s, accessor> voxels; };
template<class accessor> struct readonly_accessor
{
    accessor source;
    static constexpr bool read_only = true;
    covoxslam_storage_hd std::size_t size() const noexcept { return source.size(); }
    covoxslam_storage_hd std::size_t capacity() const noexcept { return source.capacity(); }
    covoxslam_storage_hd bool valid() const noexcept { return source.valid(); }
    const void* owner_identity() const noexcept requires requires { source.owner_identity(); }
    {
        return source.owner_identity();
    }
    auto record_segment(std::size_t first) const noexcept requires requires { source.record_segment(first); }
    {
        const auto segment = source.record_segment(first);
        return std::span<const typename decltype(segment)::element_type>(segment);
    }
    template<std::size_t field> auto field_segment(std::size_t first) const noexcept
        requires requires { source.template field_segment<field>(first); }
    {
        const auto segment = source.template field_segment<field>(first);
        return std::span<const typename decltype(segment)::element_type>(segment);
    }
    template<std::size_t field>
    covoxslam_storage_hd const auto& scalar(std::size_t tile, std::size_t voxel) const noexcept
    {
        return source.template scalar<field>(tile, voxel);
    }
};
template<schema s, class accessor> class map_view
{
    accessor access_;
public:
    covoxslam_storage_hd explicit map_view(accessor access) noexcept : access_(access) {}
    covoxslam_storage_hd std::size_t size() const noexcept { return access_.size(); }
    covoxslam_storage_hd std::size_t capacity() const noexcept { return access_.capacity(); }
    covoxslam_storage_hd bool empty() const noexcept { return size() == 0; }
    covoxslam_storage_hd bool valid() const noexcept { return access_.valid(); }
    const accessor& native_accessor() const noexcept { return access_; }
    covoxslam_storage_hd auto as_const() const noexcept
    {
        return map_view<s, readonly_accessor<accessor>>{readonly_accessor<accessor>{access_}};
    }
    covoxslam_storage_hd auto operator[](std::size_t tile) const noexcept
    {
        return tile_view<s, accessor>{{access_, tile}};
    }
};
template<schema s, bool constant>
covoxslam_storage_hd value<s> gather_record(const voxel_reference<s, constant>& record) noexcept
{
    if constexpr (s == schema::tsdf)
        return {record.distance, record.weight, record.sample_count, record.flags};
    else if constexpr (s == schema::esdf)
        return {record.signed_distance_m, record.parent_dx, record.parent_dy, record.parent_dz, record.flags};
    else if constexpr (s == schema::blocks_esdf)
        return {record.distance_m, record.parent_dx, record.parent_dy, record.parent_dz, record.flags};
    else return {record.squared_distance_voxels, record.parent_x, record.parent_y, record.parent_z, record.flags};
}
template<schema s>
covoxslam_storage_hd void scatter_record(voxel_reference<s, false> target, value<s> source) noexcept
{
    if constexpr (s == schema::tsdf)
    {
        target.distance = source.distance;
        target.weight = source.weight;
        target.sample_count = source.sample_count;
    }
    else if constexpr (s == schema::projective_esdf) {
        target.squared_distance_voxels = source.squared_distance_voxels;
        target.parent_x = source.parent_x;
        target.parent_y = source.parent_y;
        target.parent_z = source.parent_z;
    }
    else
    {
        if constexpr (s == schema::esdf) target.signed_distance_m = source.signed_distance_m;
        else target.distance_m = source.distance_m;
        target.parent_dx = source.parent_dx;
        target.parent_dy = source.parent_dy;
        target.parent_dz = source.parent_dz;
    }
     target.flags = source.flags;
}
}
