#pragma once
#include <covoxslam/storage/views.hpp>

namespace covoxslam::storage
{
template<schema s, class accessor> struct device_voxel_array
{
    accessor access;
    std::size_t tile{};
    covoxslam_storage_hd constexpr std::size_t size() const noexcept { return 512; }
#if defined(__CUDACC__)
    __device__ auto operator[](std::size_t voxel) const noexcept
    {
        if constexpr (s == schema::tsdf)
            return voxel_reference<s, accessor::read_only>{access.template scalar<0>(tile, voxel),
                access.template scalar<1>(tile, voxel), access.template scalar<2>(tile, voxel), access.template scalar<3>(tile, voxel)};
        else
            return voxel_reference<s, accessor::read_only>{access.template scalar<0>(tile, voxel),
                access.template scalar<1>(tile, voxel), access.template scalar<2>(tile, voxel),
                access.template scalar<3>(tile, voxel), access.template scalar<4>(tile, voxel)};
    }
#endif
};
template<schema s, class accessor> struct device_tile_view { device_voxel_array<s, accessor> voxels; };
template<schema s, class accessor> class device_collection
{
    accessor access_;
public:
    covoxslam_storage_hd explicit device_collection(accessor access) noexcept : access_(access) {}
    covoxslam_storage_hd std::size_t size() const noexcept { return access_.size(); }
    covoxslam_storage_hd std::size_t capacity() const noexcept { return access_.capacity(); }
    covoxslam_storage_hd bool empty() const noexcept { return size() == 0; }
    covoxslam_storage_hd bool valid() const noexcept { return access_.valid(); }
    covoxslam_storage_hd const accessor& native_accessor() const noexcept { return access_; }
#if defined(__CUDACC__)
    __device__ auto operator[](std::size_t tile) const noexcept
    {
        return device_tile_view<s, accessor>{{access_, tile}};
    }
#endif
};
}
