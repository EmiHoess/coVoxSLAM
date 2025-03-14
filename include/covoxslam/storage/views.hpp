#pragma once
#include <covoxslam/storage/schema.hpp>
#include <cassert>

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
template<schema s, class accessor> class map_view
{
    accessor access_;
public:
    covoxslam_storage_hd explicit map_view(accessor access) noexcept : access_(access) {}
    covoxslam_storage_hd std::size_t size() const noexcept { return access_.size(); }
    covoxslam_storage_hd std::size_t capacity() const noexcept { return access_.capacity(); }
    covoxslam_storage_hd bool empty() const noexcept { return size() == 0; }
    covoxslam_storage_hd bool valid() const noexcept { return access_.valid(); }
    covoxslam_storage_hd auto operator[](std::size_t tile) const noexcept
    {
        return tile_view<s, accessor>{{access_, tile}};
    }
};
}
